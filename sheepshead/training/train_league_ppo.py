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
CE_Teacher_Design_202608.md): a closed-loop ISMCTS committee — running on
the training network itself (§15a) — labels a subsample of PLAY decisions
with shrink-and-tilt CE targets, distilled in supervised passes after each
PPO update. The teacher runs the WHOLE generation (no phases — any
teacher-off window is a measured reversion window, §13.4); the boundary
cert's absolute anchors are its certification.
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
from dataclasses import dataclass
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
from sheepshead.training.league_cli import build_arg_parser
from sheepshead.training.league_gates import (
    LEAGUE_ANCHOR_EVAL_SEED,
    GateExit,
    check_adherence_guard,
    run_boundary_cert,
)
from sheepshead.training.league_streams import (
    MainPhaseContext,
    TransitionCounter,
    parallel_stream,
    sequential_stream,
)
from sheepshead.training.league_teacher import TeacherSettings, warn_if_oracle_overwrite
from sheepshead.training.league_worker import league_worker_init, publish_weights
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
def apply_schedules(episode: int, context: MainPhaseContext):
    hyperparams = context.hyperparams or PFSP_HYPERPARAMS
    pct = min(100.0, 100.0 * episode / max(context.args.schedule_horizon, 1))
    decay = 1.0 - pct / 100.0
    context.training_agent.entropy_coeff_pick = (
        hyperparams.entropy_pick_end
        + (hyperparams.entropy_pick_start - hyperparams.entropy_pick_end) * decay
    )
    context.training_agent.entropy_coeff_partner = (
        hyperparams.entropy_partner_end
        + (hyperparams.entropy_partner_start - hyperparams.entropy_partner_end) * decay
    )
    context.training_agent.entropy_coeff_bury = (
        hyperparams.entropy_bury_end
        + (hyperparams.entropy_bury_start - hyperparams.entropy_bury_end) * decay
    )
    context.training_agent.entropy_coeff_play = (
        hyperparams.entropy_play_end
        + (hyperparams.entropy_play_start - hyperparams.entropy_play_end) * decay
    )
    context.training_agent.set_learning_rates(
        interpolated_weight(hyperparams.lr_schedule_actor, pct),
        interpolated_weight(hyperparams.lr_schedule_critic, pct),
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
        head: target
        for head in ("pick", "partner", "bury", "play")
        if (target := getattr(args, f"entropy_target_{head}", None)) is not None
    }


@dataclass
class _PhaseState:
    """Loop-carried state of one run_main_phase call, threaded through the
    per-episode phase helpers (_ingest_episode, _ppo_update,
    _run_interval_probes). One instance per phase; never outlives it."""

    context: MainPhaseContext
    checkpoint_dir: str
    training_ratings: dict
    anchor_eval: dict | None
    entropy_controller: EntropyTargetController | None
    entropy_controller_path: str
    watchdog: LeasterWatchdog | None
    pool: object | None
    progress_csv: str
    greedy_csv: str
    anchored_csv: str
    picker_scores: deque
    pick_window: deque
    leaster_window: deque
    teacher_window: dict
    start_time: float


def _setup_telemetry_csvs(checkpoint_dir: str, start_episode: int):
    """Resolve the three phase CSVs, migrating pre-existing files to wider
    schemas and trimming rows a crashed run wrote past the resume episode
    (the replayed episodes would otherwise duplicate rows)."""
    progress_csv = os.path.join(checkpoint_dir, "league_training_progress.csv")
    greedy_csv = os.path.join(checkpoint_dir, "greedy_health.csv")
    anchored_csv = os.path.join(checkpoint_dir, "anchored_eval.csv")
    if ensure_csv_columns(progress_csv, PROGRESS_CSV_HEADER):
        print("📊 Migrated league_training_progress.csv to wider schema")
    if ensure_csv_columns(greedy_csv, GREEDY_CSV_HEADER):
        print("📊 Migrated greedy_health.csv to wider schema")
    for csv_file in (progress_csv, greedy_csv, anchored_csv):
        n_trimmed = truncate_csv_rows_past_episode(csv_file, start_episode)
        if n_trimmed:
            print(
                f"🧹 Trimmed {n_trimmed} stale rows past episode {start_episode:,} "
                f"from {os.path.basename(csv_file)}"
            )
    return progress_csv, greedy_csv, anchored_csv


def _setup_entropy_controller(args, checkpoint_dir: str, context: MainPhaseContext):
    """Entropy controller v2 (always on for the league trainer,
    CE_Teacher_Design §4): the signed target-entropy controller owns the
    entropy coefficients (LR keeps its clock schedule). Bumpless handoff:
    seed alpha from the legacy schedule's value at the current episode,
    and let un-set targets adopt the first update's measurement. The
    exploiter's SimpleNamespace args carries no entropy_controller
    attribute, so best-response training stays on the plain schedule
    (disposable, exploration-hungry — a controller would fight its
    intentionally hot coefficients). Returns (controller | None, sidecar
    path)."""
    entropy_controller_path = os.path.join(checkpoint_dir, "entropy_controller.json")
    if not getattr(args, "entropy_controller", False):
        return None, entropy_controller_path
    if os.path.exists(entropy_controller_path):
        entropy_controller = EntropyTargetController.load(entropy_controller_path)
        print(
            f"🎯 Entropy controller resumed: targets "
            f"{entropy_controller.targets}  alphas {entropy_controller.alphas}"
        )
    else:
        targets = fresh_entropy_targets(args)
        entropy_controller = EntropyTargetController(
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
    apply_schedules(context.start_episode, context)
    entropy_controller.attach(context.training_agent)
    return entropy_controller, entropy_controller_path


def _spawn_worker_pool(args, league: League, context: MainPhaseContext):
    """Spawn the versioned-weights worker pool (league_worker protocol), or
    return None for the in-process sequential path (num_workers <= 1)."""
    inference_flags = (
        getattr(args, "worker_compile", None),
        getattr(args, "worker_device", None),
        getattr(args, "worker_routed_encoder", None),
    )
    if args.num_workers <= 1:
        # These only ever apply to worker processes, so with no pool they do
        # nothing at all. Say so: a flag that is silently inert reads exactly
        # like an optimization that did not help.
        if any(inference_flags):
            print(
                "⚠️  --worker-compile/--worker-device/--worker-routed-encoder "
                "ignored: they configure the worker pool, and "
                "--num-workers <= 1 runs in-process"
            )
        return None
    spawn_context = get_context("spawn")
    teacher_settings = TeacherSettings.from_args(args)
    if any(inference_flags):
        # Into the run log, because it changes the last bits of every episode
        # these workers produce: a later reader comparing two runs needs to
        # know which one was on.
        routed = getattr(args, "worker_routed_encoder", None)
        print(
            "⚡ worker inference: "
            + (
                f"routed->{routed} (>=16-row encodes on a compiled shadow, "
                "rest eager CPU) "
                if routed
                else f"device={getattr(args, 'worker_device', None) or 'process default'}, "
                f"compile={getattr(args, 'worker_compile', None) or 'off'} "
            )
            + "(throughput only; not bit-comparable with an eager run)"
        )
    return spawn_context.Pool(
        processes=args.num_workers,
        initializer=league_worker_init,
        initargs=(
            {
                "arch": getattr(args, "arch", "full"),
                "members_dir": str(league.members_dir),
                "weight_path_base": context.weight_sync["base"],
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
                "teacher_gamma": float(context.training_agent.gamma),
                # Opt-in worker throughput options; see
                # league_worker._apply_inference_options.
                "worker_device": getattr(args, "worker_device", None),
                "worker_compile": getattr(args, "worker_compile", None),
                "worker_compile_granularity": getattr(
                    args, "worker_compile_granularity", 32
                ),
                "worker_routed_encoder": getattr(args, "worker_routed_encoder", None),
            },
        ),
    )


def _ingest_episode(
    state: _PhaseState,
    mode: int,
    position: int,
    events: list,
    scores: dict,
    training_data_single: dict,
    summary: dict,
    seat_to_id: dict,
) -> None:
    """Fold one finished episode into the phase state: store its events for
    the next PPO update, accumulate telemetry windows, and update ratings."""
    league = state.context.league
    state.context.tx_counter.count += store_events_by_seat(
        state.context.training_agent, events
    )
    diagnostics = (training_data_single.get("search_diagnostics") or {}).get("play")
    if diagnostics:
        state.teacher_window["searched"] += diagnostics["count"]
        state.teacher_window["labeled"] += diagnostics["labeled"]
        state.teacher_window["material"] += diagnostics["material"]
        state.teacher_window["w_sum"] += diagnostics["w_sum"]
        state.teacher_window["spread_sum"] += diagnostics["spread_sum"]
        state.teacher_window["kl_sum"] += diagnostics["kl_sum"]
        state.teacher_window["kl_n"] += diagnostics["kl_n"]
    if training_data_single["was_picker"]:
        state.picker_scores.append(training_data_single["score"])
    state.pick_window.append(1 if training_data_single["was_picker"] else 0)
    state.leaster_window.append(1 if summary["is_leaster"] else 0)

    members_by_pos = {
        pos: league.get(member_id)
        for pos, member_id in seat_to_id.items()
        if member_id != SELF_PLAY and league.get(member_id) is not None
    }
    state.training_ratings[mode] = league.update_ratings_with_training(
        partner_mode=mode,
        training_rating=state.training_ratings[mode],
        final_scores=scores,
        training_position=position,
        opponents_by_position=members_by_pos,
        picker_seat=summary["picker"],
        partner_seat=summary["partner"],
        is_leaster=summary["is_leaster"],
    )


def _emit_progress(state: _PhaseState, episode: int, stats: dict) -> None:
    """Post-update reporting: the progress print, the progress-CSV row, and
    the teacher-window print, then reset the teacher window."""
    training_agent = state.context.training_agent
    league = state.context.league
    episodes_per_sec = (episode - state.context.start_episode) / max(
        time.time() - state.start_time, 1e-9
    )
    picker_avg = float(np.mean(state.picker_scores)) if state.picker_scores else 0.0
    anchor = stats.get("anchor", {})
    anchor_str = (
        f"  anchor_kl={anchor.get('kl', 0.0):.4f}" if anchor.get("active") else ""
    )
    advantage_stats = stats.get("advantage_stats", {})
    head_std = advantage_stats.get("head_std", {})
    adv_std_all = advantage_stats.get("std", 0.0)
    adv_std_play = head_std.get("play", 0.0)
    adv_std_pick = head_std.get("pick", 0.0)
    # Oracle mode: explained variance of each critic vs the
    # empirical return — the variance-reduction headline.
    oracle_stats = stats.get("oracle") or {}
    oracle_str = (
        f"  ev O/L {oracle_stats['ev_oracle']:.2f}/{oracle_stats['ev_limited']:.2f}"
        if oracle_stats
        else ""
    )
    head_entropy = stats.get("head_entropy_norm") or {}
    head_softband = stats.get("head_softband") or {}

    def _format_head_entropy(head):
        v = head_entropy.get(head)
        return f"{v:.2f}" if v is not None else "-"

    hnorm_str = (
        f" | Hn {_format_head_entropy('pick')}/{_format_head_entropy('partner')}/{_format_head_entropy('bury')}/{_format_head_entropy('play')}"
        if head_entropy
        else ""
    )
    print(
        f"Ep {episode:,} | picker_avg {picker_avg:+.2f} | "
        f"pick {100 * np.mean(state.pick_window):.0f}% | "
        f"leaster {100 * np.mean(state.leaster_window):.1f}% | "
        f"x-share {league.exploiter_share():.2f} | "
        f"advσ all/pick/play "
        f"{adv_std_all:.3f}/{adv_std_pick:.3f}/{adv_std_play:.3f} | "
        f"{episodes_per_sec:.1f} eps/s{anchor_str}{oracle_str}{hnorm_str}",
        flush=True,
    )
    gns_stats = stats.get("gns") or {}
    write_header = not os.path.exists(state.progress_csv)
    with open(state.progress_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(PROGRESS_CSV_HEADER)
        writer.writerow(
            [
                episode,
                f"{picker_avg:.3f}",
                f"{np.mean(state.pick_window):.3f}",
                f"{np.mean(state.leaster_window):.3f}",
                f"{league.exploiter_share():.3f}",
                f"{state.training_ratings[0].mu:.2f}",
                f"{state.training_ratings[1].mu:.2f}",
                f"{adv_std_all:.4f}",
                f"{adv_std_pick:.4f}",
                f"{adv_std_play:.4f}",
                f"{oracle_stats['ev_oracle']:.4f}" if oracle_stats else "",
                f"{oracle_stats['ev_limited']:.4f}" if oracle_stats else "",
                f"{anchor.get('kl', 0.0):.5f}" if anchor.get("active") else "",
                stats.get("optimizer_steps_total", ""),
                f"{gns_stats['global']:.0f}"
                if gns_stats.get("global") is not None
                else "",
                f"{gns_stats['lead']:.0f}" if gns_stats.get("lead") is not None else "",
                gns_stats.get("lead_rows", ""),
                f"{gns_stats['lead_adv_mean']:.4f}"
                if "lead_adv_mean" in gns_stats
                else "",
                f"{gns_stats['lead_adv_std']:.4f}"
                if "lead_adv_std" in gns_stats
                else "",
                f"{gns_stats['lead_trump_mass']:.4f}"
                if "lead_trump_mass" in gns_stats
                else "",
                *[
                    f"{head_entropy[head]:.4f}"
                    if head_entropy.get(head) is not None
                    else ""
                    for head in ("pick", "partner", "bury", "play")
                ],
                *[
                    f"{head_softband[head]:.4f}"
                    if head_softband.get(head) is not None
                    else ""
                    for head in ("pick", "partner", "bury", "play")
                ],
                f"{stats.get('approx_kl', 0.0):.6f}",
                f"{training_agent.actor_optimizer.param_groups[0]['lr']:.2e}",
                state.teacher_window["searched"],
                f"{state.teacher_window['material'] / state.teacher_window['labeled']:.3f}"
                if state.teacher_window["labeled"]
                else "",
                f"{state.teacher_window['kl_sum'] / state.teacher_window['kl_n']:.4f}"
                if state.teacher_window["kl_n"]
                else "",
                f"{stats['teacher']['ce']:.4f}" if stats.get("teacher") else "",
            ]
        )
    if state.teacher_window["searched"]:
        teacher_stats = stats.get("teacher") or {}
        teacher_window = state.teacher_window
        # Mean label-time KL(target||policy) is the self-retirement
        # readout: it decays toward 0 as the student conforms to the
        # committee.
        kl_str = (
            f", KL {teacher_window['kl_sum'] / teacher_window['kl_n']:.3f}"
            if teacher_window["kl_n"]
            else ""
        )
        print(
            f"🎓 teacher: {teacher_window['searched']} searched, "
            f"{teacher_window['material']} material "
            f"({100 * teacher_window['material'] / max(teacher_window['labeled'], 1):.0f}%), "
            f"mean w {teacher_window['w_sum'] / max(teacher_window['labeled'], 1):.2f}, "
            f"spread {teacher_window['spread_sum'] / max(teacher_window['labeled'], 1):.3f}"
            f"{kl_str} | CE {teacher_stats.get('ce', 0.0):.4f} "
            f"x{teacher_stats.get('epochs', 0)} epochs "
            f"({teacher_stats.get('rows', 0)} rows)",
            flush=True,
        )
    state.teacher_window = fresh_teacher_window()


def _ppo_update(state: _PhaseState, episode: int) -> None:
    """One PPO update at an update-interval boundary: schedules, entropy
    controller, watchdog kick, the gradient update (+ CE teacher passes),
    exploiter retirement, weight republish, and progress reporting."""
    args = state.context.args
    training_agent = state.context.training_agent
    apply_schedules(episode, state.context)
    if state.entropy_controller is not None:
        # Controller owns the entropy coefficients (overrides the
        # schedule's); the watchdog kick below still multiplies on top —
        # it stays the upward override.
        state.entropy_controller.apply(training_agent)
    if state.watchdog is not None:
        state.watchdog.tick(training_agent, state.leaster_window)
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
    state.context.tx_counter.count = 0
    if state.entropy_controller is not None and stats:
        had_pending = any(
            state.entropy_controller.targets[h] is None
            for h in ("pick", "partner", "bury", "play")
        )
        state.entropy_controller.observe(stats.get("head_entropy_norm") or {})
        if had_pending and not any(
            state.entropy_controller.targets[h] is None
            for h in ("pick", "partner", "bury", "play")
        ):
            print(
                "🎯 Entropy targets initialized (bumpless): "
                + "  ".join(
                    f"{h} {state.entropy_controller.targets[h]:.3f}"
                    for h in ("pick", "partner", "bury", "play")
                )
            )
        state.entropy_controller.save(state.entropy_controller_path)
    for member_id in state.context.league.retire_patched_exploiters():
        print(f"🩹 Exploiter {member_id} patched (EMA collapsed); retired")
    if state.pool is not None:
        publish_weights(state.context)
    if stats:
        _emit_progress(state, episode, stats)


def _run_interval_probes(state: _PhaseState, episode: int) -> None:
    """Interval-keyed side effects after each episode: league snapshot,
    greedy-health probe + gates, adherence guard, anchored strength probe,
    and the periodic checkpoint save."""
    args = state.context.args
    training_agent = state.context.training_agent
    league = state.context.league

    # League snapshot of the main (replaces population_add_interval)
    if episode % args.snapshot_interval == 0:
        snapshot = copy.deepcopy(training_agent)
        snapshot.set_anchor(None, 0.0)
        # League members are inference-only: drop the privileged critic
        # so it isn't persisted into every member checkpoint.
        snapshot.strip_oracle()
        league.add_member(
            snapshot,
            ROLE_PAST_MAIN,
            training_episodes=episode,
            initial_ratings=_inherited_ratings(league, state.training_ratings),
        )
        print(f"👥 League snapshot at ep {episode:,}; {league.summary()}")

    # Greedy health probe + gates (collapse guard, unchanged semantics)
    if args.greedy_eval_interval > 0 and episode % args.greedy_eval_interval == 0:
        hyperparams = state.context.hyperparams or PFSP_HYPERPARAMS
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
        if probe["pick_rate"] < hyperparams.greedy_gate_min_pick:
            print(
                f"🚨 GREEDY GATE VIOLATION: PICK rate < {hyperparams.greedy_gate_min_pick:.0f}%",
                flush=True,
            )
        if probe["alone_rate"] > hyperparams.greedy_gate_max_alone:
            print(
                f"🚨 GREEDY GATE VIOLATION: ALONE rate > "
                f"{hyperparams.greedy_gate_max_alone:.0f}%",
                flush=True,
            )
        if probe["t0_trump_lead_rate"] > hyperparams.greedy_gate_max_trump_lead:
            print(
                f"🚨 GREEDY GATE VIOLATION: trump-lead > "
                f"{hyperparams.greedy_gate_max_trump_lead:.0f}%",
                flush=True,
            )
        if probe["play_logit_spread_med"] < hyperparams.greedy_gate_min_play_spread:
            print(
                "🚨 GREEDY GATE VIOLATION: play-head logit spread < "
                f"{hyperparams.greedy_gate_min_play_spread} "
                "(play head collapsing toward uniform)",
                flush=True,
            )
        write_header = not os.path.exists(state.greedy_csv)
        with open(state.greedy_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(GREEDY_CSV_HEADER)
            writer.writerow(
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

    # Convention adherence guard (league_gates): halts the run via
    # GateExit(3) on a hard-tier violation.
    if (
        getattr(args, "adherence_guard_interval", 0) > 0
        and episode % args.adherence_guard_interval == 0
    ):
        check_adherence_guard(
            training_agent,
            args,
            episode,
            checkpoint_path(state.checkpoint_dir, args, episode),
            league,
        )

    # Anchored strength probe: paired CRN greedy edge vs the frozen
    # reference (fixed deal set => probe-to-probe diffs are paired).
    if state.anchor_eval is not None and episode % state.anchor_eval["interval"] == 0:
        saved_mem = training_agent.snapshot_player_memories()
        probe = paired_edge(
            training_agent,
            state.anchor_eval["agent"],
            state.anchor_eval["agent"],
            n_deals=state.anchor_eval["deals"],
            seed=LEAGUE_ANCHOR_EVAL_SEED,
            log_every=0,
        )
        training_agent.restore_player_memories(saved_mem)
        print(
            f"⚓ Anchored eval vs {state.anchor_eval['label']}: "
            f"{probe['edge']:+.3f} ± {probe['se']:.3f} score/deal "
            f"(win {probe['win_frac']:.3f}, n={probe['n_deals']})",
            flush=True,
        )
        append_csv_row(
            state.anchored_csv,
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
        training_agent.save(checkpoint_path(state.checkpoint_dir, args, episode))
        league.save()


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
    # Oracle critic (critic_mode="oracle"): collection attaches full-information
    # oracle_state to every training-agent event; the learner uses it as the
    # GAE baseline (asymmetric actor-critic; see oracle.py). getattr keeps the
    # exploiter's SimpleNamespace args (no critic_mode field) on the limited path.
    collect_oracle = getattr(args, "critic_mode", "limited") == "oracle"
    context = MainPhaseContext(
        training_agent=training_agent,
        league=league,
        rng=random.Random(args.seed + start_episode),
        args=args,
        collect_oracle=collect_oracle,
        weight_sync={
            "version": 0,
            "base": os.path.join("runs", args.run_name, "_league_worker_weights"),
        },
        tx_counter=TransitionCounter(),
        start_episode=start_episode,
        end_episode=start_episode + n_episodes,
    )
    progress_csv, greedy_csv, anchored_csv = _setup_telemetry_csvs(
        checkpoint_dir, start_episode
    )
    entropy_controller, entropy_controller_path = _setup_entropy_controller(
        args, checkpoint_dir, context
    )
    pool = _spawn_worker_pool(args, league, context)
    stream = (
        parallel_stream(context, pool, args.num_workers)
        if pool is not None
        else sequential_stream(context)
    )
    state = _PhaseState(
        context=context,
        checkpoint_dir=checkpoint_dir,
        training_ratings=training_ratings,
        anchor_eval=anchor_eval,
        entropy_controller=entropy_controller,
        entropy_controller_path=entropy_controller_path,
        # getattr: the exploiter's SimpleNamespace args has no
        # leaster_watchdog field, so best-response training always runs
        # without the kick.
        watchdog=(
            LeasterWatchdog() if getattr(args, "leaster_watchdog", False) else None
        ),
        pool=pool,
        progress_csv=progress_csv,
        greedy_csv=greedy_csv,
        anchored_csv=anchored_csv,
        picker_scores=deque(maxlen=3000),
        pick_window=deque(maxlen=3000),
        leaster_window=deque(maxlen=3000),
        teacher_window=fresh_teacher_window(),
        start_time=time.time(),
    )

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
            _ingest_episode(
                state,
                mode,
                position,
                events,
                scores,
                training_data_single,
                summary,
                seat_to_id,
            )
            if context.tx_counter.count >= args.update_interval:
                _ppo_update(state, episode)
            _run_interval_probes(state, episode)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return last_episode


def run_exploiter_generation(args, generation: int, main_ckpt: str) -> dict:
    """Subprocess the exploiter module vs the frozen main; returns the gate result."""
    exploiter_run = f"{args.run_name}_exploiter_gen{generation}"
    cmd = [
        sys.executable,
        "-m",
        "sheepshead.training.exploiter",
        "--main-ckpt",
        main_ckpt,
        "--run-name",
        exploiter_run,
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
    with open(os.path.join("runs", exploiter_run, "gate_result.json")) as f:
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


def _bootstrap_league(args) -> League:
    """Open (or bootstrap) the league per the CLI: legacy migration, then
    checkpoint seeding, then cold-start from pure self-play."""
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
    return league


def _build_training_agent(args) -> tuple[PPOAgent, int]:
    """Construct the main agent from --resume and apply the CLI run options
    (each printing its banner). Returns (agent, resumed start episode)."""
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
        warn_if_oracle_overwrite(training_agent, args.oracle_init, args.resume)
        oracle_state_dict = torch.load(
            args.oracle_init, map_location="cpu", weights_only=True
        )
        training_agent.oracle_critic.load_state_dict(oracle_state_dict, strict=True)
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
    return training_agent, start_episode


def _setup_anchor_eval(args) -> dict | None:
    """Config for the periodic anchored strength probe vs a frozen
    reference checkpoint, or None when disabled/missing."""
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
    return anchor_eval


def _append_exploitability(csv_path: str, generation: int, episode: int, gate: dict):
    """Append one generation's exploiter-gate edge to exploitability.csv —
    the run's empirical-exploitability headline series."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
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
        writer.writerow(
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


def main():
    args = build_arg_parser().parse_args()

    if getattr(args, "worker_routed_encoder", None) and getattr(
        args, "worker_device", None
    ):
        raise SystemExit(
            "--worker-routed-encoder and --worker-device are mutually "
            "exclusive: routing exists to keep the process on CPU and ship "
            "only committee-scale encodes to the device (§16.6)"
        )

    set_all_seeds(args.seed)

    run_dir = os.path.join("runs", args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    league = _bootstrap_league(args)
    training_agent, start_episode = _build_training_agent(args)
    anchor_eval = _setup_anchor_eval(args)

    training_ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
    exploitability_csv = os.path.join(checkpoint_dir, "exploitability.csv")

    episode = start_episode
    episodes_per_gen = args.main_episodes
    # Generation index and phase boundary are derived from the ABSOLUTE episode
    # count: gen g ends at g * episodes_per_gen. The first generation to run is the one
    # whose boundary lies past the resumed episode, so a mid-run restart picks up
    # the same cadence/numbering and only trains the remainder to the next
    # boundary (rather than resetting the counter to the resume point).
    first_gen = episode // episodes_per_gen + 1
    # Always-on teacher generations (CE_Teacher_Design §3): NO phases — a
    # teacher-off window is a measured reversion window (§13.4). The expert
    # is the training network itself (closed-loop, §15a), so there is no
    # per-generation expert pin; the boundary cert (run_boundary_cert) is
    # its certification — fixed bars, never relative-to-previous, so
    # boundary-to-boundary drift cannot ratchet.
    #
    # The cert h2h anchor is resolved ONCE at launch (absolute anchoring, §3).
    cert_anchor = getattr(args, "cert_anchor_ckpt", None) or args.resume
    for generation in range(first_gen, first_gen + args.generations):
        boundary = generation * episodes_per_gen
        print(
            f"\n{'=' * 70}\n🏁 GENERATION {generation}: main phase "
            f"({episode:,} -> {boundary:,})"
            + (
                "  [teacher expert: live (training network)]"
                if getattr(args, "teacher", False)
                else ""
            )
            + f"\n{'=' * 70}"
        )
        # Per-generation view of the CLI namespace: the parsed args stay
        # immutable; the resolved cert anchor rides an explicit copy instead
        # of being written back into the shared namespace.
        generation_args = copy.copy(args)
        generation_args.cert_anchor_resolved = cert_anchor
        episode = run_main_phase(
            training_agent,
            league,
            training_ratings,
            generation_args,
            episode,
            boundary - episode,
            checkpoint_dir,
            anchor_eval=anchor_eval,
        )
        main_ckpt = checkpoint_path(checkpoint_dir, args, episode)
        if not os.path.exists(main_ckpt):
            training_agent.save(main_ckpt)
        # Boundary cert (CE_Teacher_Design §3): the live expert is never a
        # certified checkpoint, so the absolute-anchor cert at each boundary
        # is the teacher's whole certification; a fail halts for operator
        # review.
        if getattr(args, "teacher", False):
            cert = run_boundary_cert(
                training_agent, generation_args, generation, checkpoint_dir
            )
            if cert["passed"]:
                print("🧊 Boundary cert PASSED — live expert continues")
            else:
                league.save()
                print(
                    f"🚨 BOUNDARY CERT FAILED at gen {generation}: "
                    + "; ".join(cert["failures"])
                    + " — run halted for operator review "
                    f"(checkpoint: {main_ckpt})",
                    flush=True,
                )
                raise GateExit(4)

        gate = run_exploiter_generation(args, generation, main_ckpt)
        _append_exploitability(exploitability_csv, generation, episode, gate)
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
            boundary_snapshots = [
                m
                for m in league.by_role(ROLE_PAST_MAIN)
                if m.meta.training_episodes == episode
            ]
            if boundary_snapshots:
                league.promote_to_hof(boundary_snapshots[-1].member_id)
                print(
                    f"🏛️  Gen {generation} main survived its exploiter gate; "
                    f"{boundary_snapshots[-1].member_id} promoted to HOF anchor"
                )

    training_agent.save(os.path.join(run_dir, f"final_{args.arch}.pt"))
    print(f"\n✅ League run complete at episode {episode:,}")
    print(league.summary())


if __name__ == "__main__":
    main()
