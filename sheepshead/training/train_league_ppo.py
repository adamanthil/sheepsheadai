#!/usr/bin/env python3
"""League trainer: interleaved main/exploiter generations (plan P3).

The single training driver (notebooks/Exploiter_League_Plan_202606.md §3.4),
built on the shared game primitives in pfsp_runtime.py (play_population_game,
make_game_summary) plus its own versioned-weight worker pool. Per generation:

  MAIN PHASE:      train the main agent for --main-episodes vs tables drawn by
                   League.sample_table (PFSP past-mains / hot exploiters /
                   self-play; the 3-component mixture replaces the old
                   anchor-block + pressure + support slot scheduling).
  EXPLOITER PHASE: freeze the main, subprocess exploiter.py against it, gate,
                   auto-insert on pass (--league-dir), and append the edge to
                   exploitability.csv — the empirical-exploitability headline
                   metric (success = the trend declines across generations).

Generation boundaries are keyed to the ABSOLUTE episode count: generation g
ends at g * --main-episodes episodes and the exploiter is numbered g. Stopping
and resuming from a mid-run checkpoint therefore keeps the same cadence and
generation indices rather than resetting them to the resume point — a resume
partway through a phase trains only the episodes remaining to the next boundary.

Terminal reward, optionally with the always-on CE search teacher (--teacher,
CE_Teacher_Design_202608.md): a frozen generation-start expert's ISMCTS
committee labels a subsample of PLAY decisions with shrink-and-tilt CE
targets, distilled in supervised passes after each PPO update. The teacher
runs the WHOLE generation (no phases — any teacher-off window is a measured
reversion window, §13.4); the expert refreezes at each generation boundary.
The bidding-head KL anchor is available (--anchor-coeff) for warm-start
safety but defaults OFF: without a distillation yank there is nothing it is
known to guard against, and it caps bidding improvement.

Bootstrap an empty league one of three ways: --seed-checkpoints <glob|dir> to
seed past_mains from PPO checkpoints (e.g. the selfplay snapshots that seeded
the original pfsp run), --migrate-from <old population dir> to ingest a legacy
dual population, or neither to cold-start from pure self-play.

Usage (from-scratch reproduction matching the 30M starting point — resume the
selfplay-100k policy and seed the league from the selfplay snapshots):
  PYTHONPATH=. .venv/bin/python train_league_ppo.py \
      --resume runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
      --seed-checkpoints 'runs/reference_selfplay_ppo/checkpoints/*.pt' \
      --league-dir runs/repro_league/league --run-name repro_league \
      --generations 6 --main-episodes 5000000 --schedule-horizon 20000000
"""

from __future__ import annotations

import copy
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import deque
from multiprocessing import get_context

import numpy as np
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.config import LeagueConfig, PFSPHyperparams
from sheepshead.training.entropy_controller import (
    EntropyControllerConfig,
    EntropyTargetController,
)
from sheepshead.training.league import ROLE_PAST_MAIN, SELF_PLAY, League

# Re-exported for compatibility: build_arg_parser moved to league_cli.py.
from sheepshead.training.league_cli import build_arg_parser  # noqa: F401

# Re-exported for compatibility: moved to league_streams.py.
from sheepshead.training.league_streams import (  # noqa: F401
    AVG_TX_PER_GAME,
    MainPhaseContext,
    _TxCounter,
    parallel_stream,
    sequential_stream,
    setup_episode,
)

# Re-exported for compatibility: moved to league_teacher.py.
from sheepshead.training.league_teacher import (  # noqa: F401
    TeacherSettings,
    _build_frozen_expert,
    _teacher_kwargs,
)

# Re-exported for compatibility: moved to league_worker.py.
from sheepshead.training.league_worker import (  # noqa: F401
    _LWORKER,
    OpponentAdapter,
    _Job,
    _league_worker_get_member,
    _league_worker_init,
    _league_worker_play,
    _Seat,
    publish_weights,
)
from sheepshead.training.leaster_watchdog import LeasterWatchdog
from sheepshead.training.pfsp_runtime import interpolated_weight
from sheepshead.training.training_utils import (
    append_csv_row,
    ensure_csv_columns,
    greedy_health_probe,
    paired_edge,
    set_all_seeds,
    truncate_csv_rows_past_episode,
)

# league_training_progress.csv schema (append-only: add at the end, never
# rename; ensure_csv_columns migrates pre-existing files on resume).
PROGRESS_CSV_HEADER = [
    "episode",
    "picker_avg",
    "pick_rate",
    "leaster_rate",
    "exploiter_share",
    "mu_jd",
    "mu_ca",
    "adv_std_all",
    "adv_std_pick",
    "adv_std_play",
    "ev_oracle",
    "ev_limited",
    "anchor_kl",
    "opt_steps",
    "gns_global",
    "gns_lead",
    "lead_rows",
    "lead_adv_mean",
    "lead_adv_std",
    "lead_trump_mass",
    # Adaptive-entropy Phase 1 (2026-07-28): theta_old per-node H/ln(n_legal)
    # means per head (forced moves excluded). Instrumentation for a future
    # SAC-style target-entropy controller (Haarnoja et al., arXiv:1812.05905;
    # discrete: Christodoulou, arXiv:1910.07207) with bumpless initialization
    # from measured values.
    "ent_norm_pick",
    "ent_norm_partner",
    "ent_norm_bury",
    "ent_norm_play",
    # Soft-band fractions (share of eligible nodes with H_norm > 0.3, see
    # ppo.SOFTBAND_HNORM): boundary-band health per head — the mean H_norm
    # hides band collapse in a tail.
    "softband_pick",
    "softband_partner",
    "softband_bury",
    "softband_play",
    # LR diagnostics (2026-07-28): per-update approx KL (computed since
    # forever, never logged; target_kl is None so nothing acts on it) and
    # the actor LR actually in force. Prerequisite evidence for any future
    # KL-targeted LR adaptation (rl_games-style banded control) — the LR
    # lever stays clock-scheduled until a generation of this data says the
    # trust region binds, and any LR plateau-lever is sequenced strictly
    # after the entropy ladder floors (single-lever attributability).
    "approx_kl",
    "lr_actor",
    # CE search-teacher telemetry (CE_Teacher_Design §2): per update window,
    # nodes searched / fraction with shrink w > 0 (material) / mean label-
    # time KL(target||policy) (the self-retirement readout — decays as the
    # policy conforms) / mean CE loss over the teacher passes.
    "teacher_searched",
    "teacher_material_frac",
    "teacher_kl",
    "teacher_ce",
]

# greedy_health.csv schema (append-only; migrated on resume like the
# progress CSV).
GREEDY_CSV_HEADER = [
    "episode",
    "pick_rate",
    "alone_rate",
    "leaster_rate",
    "t0_trump_lead_rate",
    "t0_def_leads",
    "play_logit_spread_med",
    "play_nodes",
    "games",
    "partner_trump_lead_rate",
    "partner_leads",
    "called_suit_lead_rate",
    "called_leads",
]

PFSP_HYPERPARAMS = PFSPHyperparams()  # entropy/LR decay schedules + greedy-health gates

# Fixed deal-set seed for the anchored strength probe: every probe replays the
# SAME deals, so consecutive probe values are paired and the trend line is
# policy movement, not deal luck.
LEAGUE_ANCHOR_EVAL_SEED = 20260701
# Fixed seed for the n=1000 adherence guard probe: successive guard readings
# share the deal stream, so probe-to-probe deltas are paired (policy-driven,
# not deal-luck). Same seed as the offline monitoring probes (§12.17).
ADHERENCE_GUARD_SEED = 98765
# PPO epochs per update in run_main_phase's training_agent.update() call.
PPO_EPOCHS = 4


def checkpoint_path(checkpoint_dir: str, args, episode: int) -> str:
    """Standard main-agent checkpoint filename, arch-tagged."""
    return os.path.join(
        checkpoint_dir, f"pfsp_{getattr(args, 'arch', 'full')}_checkpoint_{episode}.pt"
    )


def _inherited_ratings(league: League, training_ratings: dict) -> dict:
    """Per-mode ratings for a new snapshot, seeded from the training agent's
    current rating rather than the mu=25 prior. The population's mu scale
    drifts over a long run, so a fresh prior outranks every rated member and
    turns skill-based pruning into newest-wins (the run-review F1 failure:
    the roster degenerated to a sliding window of recent snapshots). Sigma is
    floored at half the prior so the copy can still be re-rated as the field
    evolves around it."""
    min_sigma = league.rating_model.rating().sigma / 2.0
    return {
        mode: league.rating_model.rating(mu=r.mu, sigma=max(r.sigma, min_sigma))
        for mode, r in training_ratings.items()
    }


# ----------------------------------------------------------------------------
# Main phase
# ----------------------------------------------------------------------------
def apply_schedules(episode: int, ctx: MainPhaseContext):
    pct = min(100.0, 100.0 * episode / max(ctx.args.schedule_horizon, 1))
    decay = 1.0 - pct / 100.0
    ctx.training_agent.entropy_coeff_pick = (
        PFSP_HYPERPARAMS.entropy_pick_end
        + (PFSP_HYPERPARAMS.entropy_pick_start - PFSP_HYPERPARAMS.entropy_pick_end)
        * decay
    )
    ctx.training_agent.entropy_coeff_partner = (
        PFSP_HYPERPARAMS.entropy_partner_end
        + (
            PFSP_HYPERPARAMS.entropy_partner_start
            - PFSP_HYPERPARAMS.entropy_partner_end
        )
        * decay
    )
    ctx.training_agent.entropy_coeff_bury = (
        PFSP_HYPERPARAMS.entropy_bury_end
        + (PFSP_HYPERPARAMS.entropy_bury_start - PFSP_HYPERPARAMS.entropy_bury_end)
        * decay
    )
    ctx.training_agent.entropy_coeff_play = (
        PFSP_HYPERPARAMS.entropy_play_end
        + (PFSP_HYPERPARAMS.entropy_play_start - PFSP_HYPERPARAMS.entropy_play_end)
        * decay
    )
    ctx.training_agent.set_learning_rates(
        interpolated_weight(PFSP_HYPERPARAMS.lr_schedule_actor, pct),
        interpolated_weight(PFSP_HYPERPARAMS.lr_schedule_critic, pct),
    )


def store_events_by_seat(agent: PPOAgent, events: list) -> int:
    """Store one episode's events as one coherent stream PER COLLECTING SEAT.

    play_population_game returns all collecting players' events in a single
    temporally-interleaved list. Storing that list whole produced ONE
    braided multi-perspective segment per self-seat episode (done is set
    only on the final action), so the recurrent update forward ran a single
    memory across perspective switches — a train/act mismatch for every
    SELF-seat row (act-time memories are per-player), non-unit PPO ratios
    at theta_old, and the max-length padding blowups behind the 2026-07
    OOM. The selfplay trainer that produced the seeds always stored
    per-player streams; this restores the same semantics on the league
    path (bug fix, 2026-07-24 operator directive; a sibling of the
    pre-30M interleaving bug). Hero-only episodes are unaffected (single
    group == the historical call). Returns the number of action rows
    stored.
    """
    by_player: dict[int, list] = {}
    for ev in events:
        by_player.setdefault(ev["player_id"], []).append(ev)
    n_actions = 0
    for pid in sorted(by_player):
        agent.store_episode_events(by_player[pid])
        n_actions += sum(1 for ev in by_player[pid] if ev["kind"] == "action")
    return n_actions


def fresh_teacher_window() -> dict:
    """Zeroed CE-teacher telemetry accumulator (reset after each
    progress-CSV row)."""
    return {
        "searched": 0,
        "labeled": 0,
        "material": 0,
        "w_sum": 0.0,
        "spread_sum": 0.0,
        "kl_sum": 0.0,
        "kl_n": 0,
    }


def fresh_entropy_targets(args) -> dict:
    """Initial targets for a FRESH entropy controller (no sidecar yet):
    the explicit --entropy-target-* flags, if any. The default (empty) is
    bumpless adoption of the first update's measurement — correct because
    the orchestrator only enables target mode from generation 2, at a
    settled operating point (seed-transient entropy levels are never
    captured as targets; see run_extended_league.trainer_cmd)."""
    return {
        h: v
        for h in ("pick", "partner", "bury", "play")
        if (v := getattr(args, f"entropy_target_{h}", None)) is not None
    }


def run_main_phase(
    training_agent: PPOAgent,
    league: League,
    training_ratings: dict,
    args,
    start_episode: int,
    n_episodes: int,
    checkpoint_dir: str,
    anchor_eval: dict | None = None,
) -> int:
    """Train the main agent for ``n_episodes`` vs league tables; returns the
    final episode index. Mutates league ratings/EMAs and training_ratings.

    ``anchor_eval`` (optional): {"agent", "label", "interval", "deals"} — a
    frozen reference for the periodic paired CRN greedy probe, the run's only
    absolute-strength signal (run-review F7). The deal set is fixed across
    probes, so successive probe values are paired with each other and the
    trend is policy-driven, not deal-luck."""
    rng = random.Random(args.seed + start_episode)
    end_episode = start_episode + n_episodes
    # Oracle critic (critic_mode="oracle"): collection attaches full-information
    # oracle_state to every training-agent event; the learner uses it as the
    # GAE baseline (asymmetric actor-critic; see oracle.py). getattr keeps the
    # exploiter's SimpleNamespace args (no critic_mode field) on the limited path.
    collect_oracle = getattr(args, "critic_mode", "limited") == "oracle"
    picker_scores = deque(maxlen=3000)
    pick_window = deque(maxlen=3000)
    leaster_window = deque(maxlen=3000)
    # getattr: the exploiter's SimpleNamespace args has no leaster_watchdog
    # field, so best-response training always runs without the kick.
    watchdog = LeasterWatchdog() if getattr(args, "leaster_watchdog", False) else None
    # CE-teacher telemetry window (reset after each progress-CSV row).
    teacher_window = fresh_teacher_window()
    t0 = time.time()

    progress_csv = os.path.join(checkpoint_dir, "league_training_progress.csv")
    greedy_csv = os.path.join(checkpoint_dir, "greedy_health.csv")
    anchored_csv = os.path.join(checkpoint_dir, "anchored_eval.csv")
    # Crash-resume dedupe: drop telemetry a crashed run wrote past the
    # resume episode, or the replayed episodes would duplicate rows.
    if ensure_csv_columns(progress_csv, PROGRESS_CSV_HEADER):
        print("📊 Migrated league_training_progress.csv to wider schema")
    if ensure_csv_columns(greedy_csv, GREEDY_CSV_HEADER):
        print("📊 Migrated greedy_health.csv to wider schema")
    for _csv in (progress_csv, greedy_csv, anchored_csv):
        _n = truncate_csv_rows_past_episode(_csv, start_episode)
        if _n:
            print(
                f"🧹 Trimmed {_n} stale rows past episode {start_episode:,} "
                f"from {os.path.basename(_csv)}"
            )

    weight_sync = {
        "version": 0,
        "base": os.path.join("runs", args.run_name, "_league_worker_weights"),
    }
    tx_counter = _TxCounter()
    ctx = MainPhaseContext(
        training_agent=training_agent,
        league=league,
        rng=rng,
        args=args,
        collect_oracle=collect_oracle,
        weight_sync=weight_sync,
        tx_counter=tx_counter,
        start_episode=start_episode,
        end_episode=end_episode,
    )

    # Entropy controller v2 (always on for the league trainer,
    # CE_Teacher_Design §4): the signed target-entropy controller owns the
    # entropy coefficients (LR keeps its clock schedule). Bumpless handoff:
    # seed alpha from the legacy schedule's value at the current episode,
    # and let un-set targets adopt the first update's measurement. The
    # exploiter's SimpleNamespace args carries no entropy_controller
    # attribute, so best-response training stays on the plain schedule
    # (disposable, exploration-hungry — a controller would fight its
    # intentionally hot coefficients).
    entropy_ctrl = None
    entropy_ctrl_path = os.path.join(checkpoint_dir, "entropy_controller.json")
    if getattr(args, "entropy_controller", False):
        if os.path.exists(entropy_ctrl_path):
            entropy_ctrl = EntropyTargetController.load(entropy_ctrl_path)
            print(
                f"🎯 Entropy controller resumed: targets "
                f"{entropy_ctrl.targets}  alphas {entropy_ctrl.alphas}"
            )
        else:
            targets = fresh_entropy_targets(args)
            entropy_ctrl = EntropyTargetController(
                config=EntropyControllerConfig(
                    floors={"play": getattr(args, "entropy_play_floor", 0.28)}
                ),
                targets=targets,
            )
            if targets:
                print(
                    "🎯 Entropy controller fresh, explicit targets: "
                    + "  ".join(f"{h} {v:.3f}" for h, v in sorted(targets.items()))
                )
            else:
                print("🎯 Entropy controller fresh (bumpless targets pending)")
        apply_schedules(start_episode, ctx)
        entropy_ctrl.attach(training_agent)

    pool = None
    if args.num_workers > 1:
        mp_ctx = get_context("spawn")
        teacher_settings = TeacherSettings.from_args(args)
        pool = mp_ctx.Pool(
            processes=args.num_workers,
            initializer=_league_worker_init,
            initargs=(
                {
                    "arch": getattr(args, "arch", "full"),
                    "members_dir": str(league.members_dir),
                    "weight_path_base": weight_sync["base"],
                    "base_seed": args.seed,
                    # CE search teacher in workers: oracle-mode agents so
                    # the payload's oracle head loads (calibrated leaves) and
                    # gamma rides the payload (search discounting).
                    "critic_mode": getattr(args, "critic_mode", "limited"),
                    "oracle_aux_heads": bool(getattr(args, "oracle_aux_heads", False)),
                    "teacher": teacher_settings.enabled,
                    "teacher_prob": teacher_settings.prob,
                    "teacher_replicates": teacher_settings.replicates,
                    "teacher_iters": teacher_settings.iters,
                    # Stationary expert: workers rebuild the frozen
                    # generation-start policy from these paths; weight
                    # refreshes touch only the live agent. --teacher-ckpt
                    # pins the expert independently of --resume so a mid-run
                    # continuation doesn't silently refreeze to student weights.
                    "teacher_resume": teacher_settings.ckpt,
                    "teacher_oracle_init": teacher_settings.oracle_init,
                    "teacher_gamma": float(training_agent.gamma),
                },
            ),
        )
        stream = parallel_stream(ctx, pool, args.num_workers)
    else:
        stream = sequential_stream(ctx)

    last_episode = start_episode
    try:
        for (
            episode,
            mode,
            position,
            events,
            scores,
            training_data_single,
            summary,
            seat_to_id,
        ) in stream:
            last_episode = episode
            tx_counter.count += store_events_by_seat(training_agent, events)
            sd = (training_data_single.get("search_diagnostics") or {}).get("play")
            if sd:
                teacher_window["searched"] += sd["count"]
                teacher_window["labeled"] += sd["labeled"]
                teacher_window["material"] += sd["material"]
                teacher_window["w_sum"] += sd["w_sum"]
                teacher_window["spread_sum"] += sd["spread_sum"]
                teacher_window["kl_sum"] += sd["kl_sum"]
                teacher_window["kl_n"] += sd["kl_n"]
            if training_data_single["was_picker"]:
                picker_scores.append(training_data_single["score"])
            pick_window.append(1 if training_data_single["was_picker"] else 0)
            leaster_window.append(1 if summary["is_leaster"] else 0)

            members_by_pos = {
                pos: league.get(mid)
                for pos, mid in seat_to_id.items()
                if mid != SELF_PLAY and league.get(mid) is not None
            }
            training_ratings[mode] = league.update_ratings_with_training(
                partner_mode=mode,
                training_rating=training_ratings[mode],
                final_scores=scores,
                training_position=position,
                opponents_by_position=members_by_pos,
                picker_seat=summary["picker"],
                partner_seat=summary["partner"],
                is_leaster=summary["is_leaster"],
            )

            if tx_counter.count >= args.update_interval:
                apply_schedules(episode, ctx)
                if entropy_ctrl is not None:
                    # Controller owns the entropy coefficients (overrides the
                    # schedule's); the watchdog kick below still multiplies
                    # on top — it stays the upward override.
                    entropy_ctrl.apply(training_agent)
                if watchdog is not None:
                    watchdog.tick(training_agent, leaster_window)
                stats = training_agent.update(
                    oracle_extra_epochs=getattr(args, "oracle_extra_epochs", 0),
                    epochs=PPO_EPOCHS,
                    batch_size=getattr(args, "minibatch_episodes", 256),
                    grad_accum=getattr(args, "grad_accum", False),
                    teacher_epochs=(
                        int(getattr(args, "teacher_epochs", 0))
                        if getattr(args, "teacher", False)
                        else 0
                    ),
                )
                tx_counter.count = 0
                if entropy_ctrl is not None and stats:
                    had_pending = any(
                        entropy_ctrl.targets[h] is None
                        for h in ("pick", "partner", "bury", "play")
                    )
                    entropy_ctrl.observe(stats.get("head_entropy_norm") or {})
                    if had_pending and not any(
                        entropy_ctrl.targets[h] is None
                        for h in ("pick", "partner", "bury", "play")
                    ):
                        print(
                            "🎯 Entropy targets initialized (bumpless): "
                            + "  ".join(
                                f"{h} {entropy_ctrl.targets[h]:.3f}"
                                for h in ("pick", "partner", "bury", "play")
                            )
                        )
                    entropy_ctrl.save(entropy_ctrl_path)
                for mid in league.retire_patched_exploiters():
                    print(f"🩹 Exploiter {mid} patched (EMA collapsed); retired")
                if pool is not None:
                    publish_weights(ctx)
                if stats:
                    eps_s = (episode - start_episode) / max(time.time() - t0, 1e-9)
                    picker_avg = float(np.mean(picker_scores)) if picker_scores else 0.0
                    anchor = stats.get("anchor", {})
                    anchor_str = (
                        f"  anchor_kl={anchor.get('kl', 0.0):.4f}"
                        if anchor.get("active")
                        else ""
                    )
                    astats = stats.get("advantage_stats", {})
                    hstd = astats.get("head_std", {})
                    adv_std_all = astats.get("std", 0.0)
                    adv_std_play = hstd.get("play", 0.0)
                    adv_std_pick = hstd.get("pick", 0.0)
                    # Oracle mode: explained variance of each critic vs the
                    # empirical return — the variance-reduction headline.
                    ostats = stats.get("oracle") or {}
                    oracle_str = (
                        f"  ev O/L {ostats['ev_oracle']:.2f}/{ostats['ev_limited']:.2f}"
                        if ostats
                        else ""
                    )
                    hnorm = stats.get("head_entropy_norm") or {}
                    hsoft = stats.get("head_softband") or {}

                    def _hn(head):
                        v = hnorm.get(head)
                        return f"{v:.2f}" if v is not None else "-"

                    hnorm_str = (
                        f" | Hn {_hn('pick')}/{_hn('partner')}/"
                        f"{_hn('bury')}/{_hn('play')}"
                        if hnorm
                        else ""
                    )
                    print(
                        f"Ep {episode:,} | picker_avg {picker_avg:+.2f} | "
                        f"pick {100 * np.mean(pick_window):.0f}% | "
                        f"leaster {100 * np.mean(leaster_window):.1f}% | "
                        f"x-share {league.exploiter_share():.2f} | "
                        f"advσ all/pick/play "
                        f"{adv_std_all:.3f}/{adv_std_pick:.3f}/{adv_std_play:.3f} | "
                        f"{eps_s:.1f} eps/s{anchor_str}{oracle_str}{hnorm_str}",
                        flush=True,
                    )
                    gns = stats.get("gns") or {}
                    write_header = not os.path.exists(progress_csv)
                    with open(progress_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(PROGRESS_CSV_HEADER)
                        w.writerow(
                            [
                                episode,
                                f"{picker_avg:.3f}",
                                f"{np.mean(pick_window):.3f}",
                                f"{np.mean(leaster_window):.3f}",
                                f"{league.exploiter_share():.3f}",
                                f"{training_ratings[0].mu:.2f}",
                                f"{training_ratings[1].mu:.2f}",
                                f"{adv_std_all:.4f}",
                                f"{adv_std_pick:.4f}",
                                f"{adv_std_play:.4f}",
                                f"{ostats['ev_oracle']:.4f}" if ostats else "",
                                f"{ostats['ev_limited']:.4f}" if ostats else "",
                                f"{anchor.get('kl', 0.0):.5f}"
                                if anchor.get("active")
                                else "",
                                stats.get("optimizer_steps_total", ""),
                                f"{gns['global']:.0f}"
                                if gns.get("global") is not None
                                else "",
                                f"{gns['lead']:.0f}"
                                if gns.get("lead") is not None
                                else "",
                                gns.get("lead_rows", ""),
                                f"{gns['lead_adv_mean']:.4f}"
                                if "lead_adv_mean" in gns
                                else "",
                                f"{gns['lead_adv_std']:.4f}"
                                if "lead_adv_std" in gns
                                else "",
                                f"{gns['lead_trump_mass']:.4f}"
                                if "lead_trump_mass" in gns
                                else "",
                                *[
                                    f"{hnorm[head]:.4f}"
                                    if hnorm.get(head) is not None
                                    else ""
                                    for head in ("pick", "partner", "bury", "play")
                                ],
                                *[
                                    f"{hsoft[head]:.4f}"
                                    if hsoft.get(head) is not None
                                    else ""
                                    for head in ("pick", "partner", "bury", "play")
                                ],
                                f"{stats.get('approx_kl', 0.0):.6f}",
                                f"{training_agent.actor_optimizer.param_groups[0]['lr']:.2e}",
                                teacher_window["searched"],
                                f"{teacher_window['material'] / teacher_window['labeled']:.3f}"
                                if teacher_window["labeled"]
                                else "",
                                f"{teacher_window['kl_sum'] / teacher_window['kl_n']:.4f}"
                                if teacher_window["kl_n"]
                                else "",
                                f"{stats['teacher']['ce']:.4f}"
                                if stats.get("teacher")
                                else "",
                            ]
                        )
                    if teacher_window["searched"]:
                        tstats = stats.get("teacher") or {}
                        w = teacher_window
                        # Mean label-time KL(target||policy) is the
                        # self-retirement readout: it decays toward 0 as
                        # the student conforms to the committee.
                        kl = f", KL {w['kl_sum'] / w['kl_n']:.3f}" if w["kl_n"] else ""
                        print(
                            f"🎓 teacher: {w['searched']} searched, "
                            f"{w['material']} material "
                            f"({100 * w['material'] / max(w['labeled'], 1):.0f}%), "
                            f"mean w {w['w_sum'] / max(w['labeled'], 1):.2f}, "
                            f"spread {w['spread_sum'] / max(w['labeled'], 1):.3f}"
                            f"{kl} | CE {tstats.get('ce', 0.0):.4f} "
                            f"x{tstats.get('epochs', 0)} epochs "
                            f"({tstats.get('rows', 0)} rows)",
                            flush=True,
                        )
                    teacher_window = fresh_teacher_window()

            # League snapshot of the main (replaces population_add_interval)
            if episode % args.snapshot_interval == 0:
                snap = copy.deepcopy(training_agent)
                snap.set_anchor(None, 0.0)
                # League members are inference-only: drop the privileged critic
                # so it isn't persisted into every member checkpoint.
                snap.strip_oracle()
                league.add_member(
                    snap,
                    ROLE_PAST_MAIN,
                    training_episodes=episode,
                    initial_ratings=_inherited_ratings(league, training_ratings),
                )
                print(f"👥 League snapshot at ep {episode:,}; {league.summary()}")

            # Greedy health probe + gates (collapse guard, unchanged semantics)
            if (
                args.greedy_eval_interval > 0
                and episode % args.greedy_eval_interval == 0
            ):
                probe = greedy_health_probe(
                    training_agent, n_games=args.greedy_eval_games, seed=episode
                )
                print(
                    f"🩺 Greedy health ({probe['games']} games): "
                    f"PICK {probe['pick_rate']:.1f}%, ALONE {probe['alone_rate']:.1f}%, "
                    f"leaster {probe['leaster_rate']:.1f}%, "
                    f"t0 trump-lead {probe['t0_trump_lead_rate']:.1f}% "
                    f"(n={probe['t0_def_leads']}), "
                    f"partner trump-lead {probe['partner_trump_lead_rate']:.1f}% "
                    f"(n={probe['partner_leads']}), "
                    f"called-suit lead {probe['called_suit_lead_rate']:.1f}% "
                    f"(n={probe['called_leads']}), "
                    f"play-spread {probe['play_logit_spread_med']:.2f}",
                    flush=True,
                )
                if probe["pick_rate"] < PFSP_HYPERPARAMS.greedy_gate_min_pick:
                    print(
                        f"🚨 GREEDY GATE VIOLATION: PICK rate < "
                        f"{PFSP_HYPERPARAMS.greedy_gate_min_pick:.0f}%",
                        flush=True,
                    )
                if probe["alone_rate"] > PFSP_HYPERPARAMS.greedy_gate_max_alone:
                    print(
                        f"🚨 GREEDY GATE VIOLATION: ALONE rate > "
                        f"{PFSP_HYPERPARAMS.greedy_gate_max_alone:.0f}%",
                        flush=True,
                    )
                if (
                    probe["t0_trump_lead_rate"]
                    > PFSP_HYPERPARAMS.greedy_gate_max_trump_lead
                ):
                    print(
                        f"🚨 GREEDY GATE VIOLATION: trump-lead > "
                        f"{PFSP_HYPERPARAMS.greedy_gate_max_trump_lead:.0f}%",
                        flush=True,
                    )
                if (
                    probe["play_logit_spread_med"]
                    < PFSP_HYPERPARAMS.greedy_gate_min_play_spread
                ):
                    print(
                        "🚨 GREEDY GATE VIOLATION: play-head logit spread < "
                        f"{PFSP_HYPERPARAMS.greedy_gate_min_play_spread} "
                        "(play head collapsing toward uniform)",
                        flush=True,
                    )
                write_header = not os.path.exists(greedy_csv)
                with open(greedy_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(GREEDY_CSV_HEADER)
                    w.writerow(
                        [
                            episode,
                            f"{probe['pick_rate']:.2f}",
                            f"{probe['alone_rate']:.2f}",
                            f"{probe['leaster_rate']:.2f}",
                            f"{probe['t0_trump_lead_rate']:.2f}",
                            probe["t0_def_leads"],
                            f"{probe['play_logit_spread_med']:.3f}",
                            probe["play_nodes"],
                            probe["games"],
                            f"{probe['partner_trump_lead_rate']:.2f}",
                            probe["partner_leads"],
                            f"{probe['called_suit_lead_rate']:.2f}",
                            probe["called_leads"],
                        ]
                    )

            # Convention adherence guard, two-tier (CE_Teacher_Design §3;
            # §12.17/§12.21 protocol): the 200-300-game greedy probe cannot
            # resolve sub-5-point convention regressions (it masked an
            # 8-point partner-trump deficit for a full teacher run), so the
            # guard reruns the probe at n=1000 on a FIXED seed (successive
            # probes are paired). HARD tier (partner below the floor, or t0
            # trump-lead above the ceiling — the scramble signature) stops
            # the run for operator review; the NOTIFY tier (partner below
            # the notify line) only prints, because §12.20 showed partner
            # dips during teaching can be oscillation with a restoring
            # force, not collapse.
            if (
                getattr(args, "adherence_guard_interval", 0) > 0
                and episode % args.adherence_guard_interval == 0
            ):
                gp = greedy_health_probe(
                    training_agent,
                    n_games=int(getattr(args, "adherence_guard_games", 1000)),
                    seed=ADHERENCE_GUARD_SEED,
                )
                print(
                    f"🛡️ Adherence guard (n={gp['games']}): "
                    f"called-suit {gp['called_suit_lead_rate']:.1f}% "
                    f"t0-trump {gp['t0_trump_lead_rate']:.1f}% "
                    f"partner-trump {gp['partner_trump_lead_rate']:.1f}%",
                    flush=True,
                )
                violations = []
                floor = getattr(args, "guard_partner_floor", None)
                if floor is not None and gp["partner_trump_lead_rate"] < float(floor):
                    violations.append(
                        f"partner trump-lead {gp['partner_trump_lead_rate']:.1f}% "
                        f"< hard floor {float(floor):.1f}%"
                    )
                ceil = getattr(args, "guard_t0_ceiling", None)
                if ceil is not None and gp["t0_trump_lead_rate"] > float(ceil):
                    violations.append(
                        f"t0 trump-lead {gp['t0_trump_lead_rate']:.1f}% "
                        f"> ceiling {float(ceil):.1f}%"
                    )
                notify = getattr(args, "guard_partner_notify", None)
                if (
                    not violations
                    and notify is not None
                    and gp["partner_trump_lead_rate"] < float(notify)
                ):
                    print(
                        f"🛡️⚠️ Adherence NOTIFY: partner trump-lead "
                        f"{gp['partner_trump_lead_rate']:.1f}% < notify line "
                        f"{float(notify):.1f}% (hard floor "
                        f"{float(floor):.1f}%) — continuing",
                        flush=True,
                    )
                if violations:
                    stop_ckpt = checkpoint_path(checkpoint_dir, args, episode)
                    training_agent.save(stop_ckpt)
                    league.save()
                    print(
                        "🚨 ADHERENCE GUARD STOP: "
                        + "; ".join(violations)
                        + f" — checkpoint saved to {stop_ckpt}; run halted for "
                        "operator review",
                        flush=True,
                    )
                    raise SystemExit(3)

            # Anchored strength probe: paired CRN greedy edge vs the frozen
            # reference (fixed deal set => probe-to-probe diffs are paired).
            if anchor_eval is not None and episode % anchor_eval["interval"] == 0:
                saved_mem = training_agent.snapshot_player_memories()
                probe = paired_edge(
                    training_agent,
                    anchor_eval["agent"],
                    anchor_eval["agent"],
                    n_deals=anchor_eval["deals"],
                    seed=LEAGUE_ANCHOR_EVAL_SEED,
                    log_every=0,
                )
                training_agent.restore_player_memories(saved_mem)
                print(
                    f"⚓ Anchored eval vs {anchor_eval['label']}: "
                    f"{probe['edge']:+.3f} ± {probe['se']:.3f} score/deal "
                    f"(win {probe['win_frac']:.3f}, n={probe['n_deals']})",
                    flush=True,
                )
                append_csv_row(
                    anchored_csv,
                    ["episode", "edge", "se", "win_frac", "n_deals"],
                    {
                        "episode": episode,
                        "edge": f"{probe['edge']:.4f}",
                        "se": f"{probe['se']:.4f}",
                        "win_frac": f"{probe['win_frac']:.4f}",
                        "n_deals": probe["n_deals"],
                    },
                )

            if episode % args.save_interval == 0:
                training_agent.save(checkpoint_path(checkpoint_dir, args, episode))
                league.save()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return last_episode


# ----------------------------------------------------------------------------
def run_boundary_cert(
    training_agent: PPOAgent, args, generation: int, checkpoint_dir: str
) -> dict:
    """Absolute-anchor expert-refresh cert (CE_Teacher_Design §3), run on the
    generation-boundary candidate before it may become the next generation's
    frozen expert.

    Two components, both against FIXED absolute bars (never relative to the
    previous generation — a relative cert lets a refresh chain ratchet drift
    into the certified regime):

    * n=--cert-games adherence battery at --cert-seeds distinct fixed deal
      seeds, judged on the ACROSS-SEED MEAN (single reads are luck-of-phase
      — the §12.22 lesson: consolidation called-suit swung 39.9-51.0 across
      reads): mean partner trump-lead >= --cert-partner-floor AND mean t0
      trump-lead <= --cert-t0-ceiling.
    * Paired CRN h2h vs the run's fixed cert anchor (--cert-anchor-ckpt,
      default the ORIGINAL expert checkpoint): edge must not be
      significantly negative (edge + 2*SE >= 0).

    The exploiter gate that follows every boundary is the third cert
    component and keeps its existing flow. Result is persisted to
    boundary_cert_gen<g>.json for the run record."""
    seeds = [ADHERENCE_GUARD_SEED + i for i in range(int(args.cert_seeds))]
    probes = [
        greedy_health_probe(training_agent, n_games=int(args.cert_games), seed=s)
        for s in seeds
    ]
    partner_mean = float(np.mean([p["partner_trump_lead_rate"] for p in probes]))
    t0_mean = float(np.mean([p["t0_trump_lead_rate"] for p in probes]))
    called_mean = float(np.mean([p["called_suit_lead_rate"] for p in probes]))

    anchor_path = args.cert_anchor_resolved
    anchor_agent = load_agent(anchor_path)
    saved_mem = training_agent.snapshot_player_memories()
    h2h = paired_edge(
        training_agent,
        anchor_agent,
        anchor_agent,
        n_deals=int(args.cert_h2h_deals),
        seed=LEAGUE_ANCHOR_EVAL_SEED,
        log_every=0,
    )
    training_agent.restore_player_memories(saved_mem)

    failures = []
    if partner_mean < float(args.cert_partner_floor):
        failures.append(
            f"partner trump-lead mean {partner_mean:.1f}% "
            f"< cert floor {float(args.cert_partner_floor):.1f}%"
        )
    if t0_mean > float(args.cert_t0_ceiling):
        failures.append(
            f"t0 trump-lead mean {t0_mean:.1f}% "
            f"> cert ceiling {float(args.cert_t0_ceiling):.1f}%"
        )
    if h2h["edge"] + 2.0 * h2h["se"] < 0.0:
        failures.append(
            f"h2h vs {os.path.basename(anchor_path)} significantly negative "
            f"({h2h['edge']:+.3f} ± {h2h['se']:.3f})"
        )
    result = {
        "generation": generation,
        "passed": not failures,
        "failures": failures,
        "adherence": {
            "seeds": seeds,
            "games_per_seed": int(args.cert_games),
            "partner_trump_by_seed": [p["partner_trump_lead_rate"] for p in probes],
            "t0_trump_by_seed": [p["t0_trump_lead_rate"] for p in probes],
            "called_suit_by_seed": [p["called_suit_lead_rate"] for p in probes],
            "partner_trump_mean": partner_mean,
            "t0_trump_mean": t0_mean,
            "called_suit_mean": called_mean,
        },
        "h2h": {
            "anchor": anchor_path,
            "edge": h2h["edge"],
            "se": h2h["se"],
            "n_deals": h2h["n_deals"],
        },
    }
    cert_path = os.path.join(checkpoint_dir, f"boundary_cert_gen{generation}.json")
    with open(cert_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"📜 Boundary cert gen {generation}: partner {partner_mean:.1f}% "
        f"t0 {t0_mean:.1f}% called-suit {called_mean:.1f}% "
        f"(means over {len(seeds)} seeds x {int(args.cert_games)} games) | "
        f"h2h vs anchor {h2h['edge']:+.3f} ± {h2h['se']:.3f} -> "
        f"{'PASS' if result['passed'] else 'FAIL'}",
        flush=True,
    )
    return result


def run_exploiter_generation(args, generation: int, main_ckpt: str) -> dict:
    """Subprocess the exploiter module vs the frozen main; returns the gate result."""
    exp_run = f"{args.run_name}_exploiter_gen{generation}"
    cmd = [
        sys.executable,
        "-m",
        "sheepshead.training.exploiter",
        "--main-ckpt",
        main_ckpt,
        "--run-name",
        exp_run,
        "--episodes",
        str(args.exploiter_episodes),
        "--gate-deals",
        str(args.gate_deals),
        "--screen-deals",
        str(args.screen_deals),
        "--generation",
        str(generation),
        "--league-dir",
        args.league_dir,
        "--seed",
        str(args.seed + generation),
        "--arch",
        getattr(args, "arch", "full"),
        # Exploiters inherit the main run's critic mode: an oracle-trained
        # main must face oracle-trained best responses, or the gate's
        # "certified robust" verdicts are weakened by a noise-limited
        # adversary (the exploiter is pure training scaffolding — CTDE
        # applies with no deployment constraint).
        "--critic-mode",
        getattr(args, "critic_mode", "limited"),
    ]
    if args.num_workers:
        cmd += ["--num-workers", str(args.num_workers)]
    print(f"🥷 Generation {generation} exploiter phase: {' '.join(cmd)}", flush=True)
    env = dict(os.environ, PYTHONPATH=".")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"exploiter phase failed (rc={proc.returncode})")
    with open(os.path.join("runs", exp_run, "gate_result.json")) as f:
        return json.load(f)


def _seed_league_from_checkpoints(league: League, spec: str) -> None:
    """Seed an empty league with PPO checkpoints as past_mains. ``spec`` is a
    glob (``.../*.pt``) or a directory (all ``*.pt`` within). Mirrors how the
    original pfsp run bootstrapped its population from the selfplay snapshots."""
    paths = (
        sorted(glob.glob(spec))
        if any(c in spec for c in "*?[")
        else sorted(glob.glob(os.path.join(spec, "*.pt")))
    )
    if not paths:
        raise SystemExit(f"--seed-checkpoints matched no .pt files: {spec}")
    for p in paths:
        agent = load_agent(p)
        episodes = 0
        if "checkpoint_" in p:
            try:
                episodes = int(os.path.basename(p).split("_")[-1].split(".")[0])
            except ValueError:
                episodes = 0
        league.add_member(agent, ROLE_PAST_MAIN, training_episodes=episodes)
    print(f"🌱 Seeded league with {len(paths)} checkpoints as past_mains")


def main():
    args = build_arg_parser().parse_args()

    set_all_seeds(args.seed)

    run_dir = os.path.join("runs", args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    league_config = LeagueConfig()
    if args.exploiter_patched_ema is not None:
        league_config.exploiter_patched_ema = args.exploiter_patched_ema
        print(
            f"🩹 Exploit-patched retirement ON: EMA < {args.exploiter_patched_ema} "
            f"with ≥ {league_config.exploiter_patched_min_samples} samples "
            "demotes to past_main"
        )
    league = League(args.league_dir, league_config)
    if len(league) == 0 and args.migrate_from:
        print(f"🏗️  Empty league; migrating legacy population from {args.migrate_from}")
        league = League.migrate_legacy(args.migrate_from, args.league_dir)
    if len(league) == 0 and args.seed_checkpoints:
        _seed_league_from_checkpoints(league, args.seed_checkpoints)
    if len(league) == 0:
        print(
            "⚠️  Empty league: bootstrapping from pure self-play until past_main "
            f"snapshots accumulate (first at +{args.snapshot_interval:,} episodes)."
        )
    print(league.summary())

    training_agent = PPOAgent(
        len(ACTIONS),
        critic_mode=args.critic_mode,
        arch=args.arch,
        oracle_aux_heads=getattr(args, "oracle_aux_heads", False),
    )
    training_agent.load(args.resume, load_optimizers=True)
    if args.gae_lambda is not None:
        training_agent.gae_lambda = float(args.gae_lambda)
        print(f"λ  GAE lambda override: {training_agent.gae_lambda}")
    if getattr(args, "gamma", None) is not None:
        training_agent.gamma = float(args.gamma)
        print(f"γ  discount override: {training_agent.gamma}")
    if getattr(args, "teacher", False):
        training_agent.teacher_coeff = float(getattr(args, "teacher_coeff", 1.0))
        print(
            f"🎓 CE search teacher ON (always-on): "
            f"prob={getattr(args, 'teacher_prob', 0.1)}, "
            f"R={int(getattr(args, 'teacher_replicates', 3))}, "
            f"iters={int(getattr(args, 'teacher_iters', 1024))}, "
            f"coeff={training_agent.teacher_coeff}, "
            f"epochs={int(getattr(args, 'teacher_epochs', 4))}"
        )
    if getattr(args, "oracle_init", None):
        sd = torch.load(args.oracle_init, map_location="cpu", weights_only=True)
        training_agent.oracle_critic.load_state_dict(sd, strict=True)
        print(f"🔮⚡ Oracle warm-started from {args.oracle_init}")
    if getattr(args, "oracle_aux_heads", False):
        print("🔮🧩 Oracle aux heads ON (team membership + team points)")
    if getattr(args, "seat_rotation", False):
        print("🔄 Seat rotation ON: each deal played once per hero seat")
    if args.arch != "full":
        print(f"🧬 Architecture: {args.arch}")
    if args.critic_mode == "oracle":
        print("🔮 Oracle critic ON: privileged full-information GAE baseline")
    if getattr(args, "gns_log", False):
        training_agent.gns_log = True
        print("📡 GNS logging ON (global + partner-lead, rows)")
    if getattr(args, "oracle_extra_epochs", 0) > 0:
        print(
            f"🔮+ Oracle extra epochs: {args.oracle_extra_epochs} "
            "regression-only passes per update"
        )
    if args.leaster_watchdog:
        print("🐶 Leaster watchdog ON (main phases)")
    start_episode = 0
    if "checkpoint_" in args.resume:
        start_episode = int(args.resume.split("_")[-1].split(".")[0])
    print(f"📍 Main resumed from {args.resume} (episode {start_episode:,})")

    if args.anchor_coeff > 0.0:
        ref = load_agent(args.anchor_ref or args.resume)
        training_agent.set_anchor(ref, args.anchor_coeff)
        print(f"⚓ Bidding anchor ON (coeff={args.anchor_coeff})")

    anchor_eval = None
    if args.anchor_eval_ckpt and args.anchor_eval_interval > 0:
        if os.path.exists(args.anchor_eval_ckpt):
            ref_agent = load_agent(args.anchor_eval_ckpt)
            anchor_eval = {
                "agent": ref_agent,
                "label": os.path.basename(args.anchor_eval_ckpt),
                "interval": args.anchor_eval_interval,
                "deals": args.anchor_eval_deals,
            }
            print(
                f"⚓ Anchored strength probe vs {args.anchor_eval_ckpt} every "
                f"{args.anchor_eval_interval:,} eps "
                f"({args.anchor_eval_deals} paired deals)"
            )
        else:
            print(
                f"⚠️  --anchor-eval-ckpt not found ({args.anchor_eval_ckpt}); "
                "anchored probe disabled"
            )

    training_ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
    exploitability_csv = os.path.join(checkpoint_dir, "exploitability.csv")

    episode = start_episode
    main_ep = args.main_episodes
    # Generation index and phase boundary are derived from the ABSOLUTE episode
    # count: gen g ends at g * main_ep. The first generation to run is the one
    # whose boundary lies past the resumed episode, so a mid-run restart picks up
    # the same cadence/numbering and only trains the remainder to the next
    # boundary (rather than resetting the counter to the resume point).
    first_gen = episode // main_ep + 1
    # Always-on teacher generations (CE_Teacher_Design §3): NO phases — a
    # teacher-off window is a measured reversion window (§13.4). The frozen
    # expert is pinned per generation to the generation-start checkpoint and
    # refreshed at each boundary only through the absolute-anchor cert
    # (run_boundary_cert): fixed bars, never relative-to-previous, so an
    # expert refresh chain cannot ratchet drift into the certified regime.
    gen_start_ckpt = getattr(args, "teacher_ckpt", None) or args.resume
    # The cert h2h anchor is resolved ONCE at launch and never follows the
    # refresh chain (absolute anchoring, §3).
    args.cert_anchor_resolved = (
        getattr(args, "cert_anchor_ckpt", None) or gen_start_ckpt
    )
    for generation in range(first_gen, first_gen + args.generations):
        boundary = generation * main_ep
        print(
            f"\n{'=' * 70}\n🏁 GENERATION {generation}: main phase "
            f"({episode:,} -> {boundary:,})"
            + (
                f"  [teacher expert: {os.path.basename(gen_start_ckpt)}]"
                if getattr(args, "teacher", False)
                else ""
            )
            + f"\n{'=' * 70}"
        )
        args.teacher_ckpt = gen_start_ckpt
        episode = run_main_phase(
            training_agent,
            league,
            training_ratings,
            args,
            episode,
            boundary - episode,
            checkpoint_dir,
            anchor_eval=anchor_eval,
        )
        main_ckpt = checkpoint_path(checkpoint_dir, args, episode)
        if not os.path.exists(main_ckpt):
            training_agent.save(main_ckpt)
        # Expert refresh (CE_Teacher_Design §3): the boundary checkpoint
        # becomes the next generation's frozen expert only if it passes the
        # absolute-anchor cert; a failed cert halts for operator review.
        if getattr(args, "teacher", False):
            cert = run_boundary_cert(training_agent, args, generation, checkpoint_dir)
            if cert["passed"]:
                gen_start_ckpt = main_ckpt
                print(
                    f"🧊 Boundary cert PASSED — gen {generation + 1} expert "
                    f"refreezes to {os.path.basename(main_ckpt)}"
                )
            else:
                league.save()
                print(
                    f"🚨 BOUNDARY CERT FAILED at gen {generation}: "
                    + "; ".join(cert["failures"])
                    + f" — expert stays {os.path.basename(gen_start_ckpt)}; "
                    "run halted for operator review "
                    f"(checkpoint: {main_ckpt})",
                    flush=True,
                )
                raise SystemExit(4)
        else:
            gen_start_ckpt = main_ckpt

        gate = run_exploiter_generation(args, generation, main_ckpt)
        write_header = not os.path.exists(exploitability_csv)
        with open(exploitability_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "generation",
                        "main_episode",
                        "edge",
                        "se",
                        "win_frac",
                        "passed",
                        "exploiter_ckpt",
                    ]
                )
            w.writerow(
                [
                    generation,
                    episode,
                    f"{gate['edge']:.4f}",
                    f"{gate['se']:.4f}",
                    f"{gate['win_frac']:.3f}",
                    gate["passed"],
                    gate["exploiter_ckpt"],
                ]
            )
        print(
            f"📊 Exploitability gen {generation}: edge {gate['edge']:+.3f} ± {gate['se']:.3f} "
            f"({'inserted' if gate['passed'] else 'below gate'})"
        )
        # Reload league to pick up the subprocess insertion
        league = League(args.league_dir, league.config)
        # Advance the generation clock (pass or fail) so exploiter
        # retirement runs on elapsed generations, not on insertions.
        league.note_generation(generation)
        if not gate["passed"]:
            # Certified robust: no best response cleared the gate against this
            # main, so its boundary snapshot becomes a HOF anchor (the
            # anti-forgetting floor; quota enforced by promote_to_hof).
            snaps = [
                m
                for m in league.by_role(ROLE_PAST_MAIN)
                if m.meta.training_episodes == episode
            ]
            if snaps:
                league.promote_to_hof(snaps[-1].member_id)
                print(
                    f"🏛️  Gen {generation} main survived its exploiter gate; "
                    f"{snaps[-1].member_id} promoted to HOF anchor"
                )

    training_agent.save(os.path.join(run_dir, f"final_{args.arch}.pt"))
    print(f"\n✅ League run complete at episode {episode:,}")
    print(league.summary())


if __name__ == "__main__":
    main()
