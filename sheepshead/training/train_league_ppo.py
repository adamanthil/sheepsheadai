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

Terminal reward only; no search/shaping/controllers (ISMCTS is a deploy-time
amplifier + audit tool per the June 2026 value-add probe). The bidding-head KL
anchor is available (--anchor-coeff) for warm-start safety but defaults OFF:
without a distillation yank there is nothing it is known to guard against, and
it caps bidding improvement.

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

import argparse
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
from types import SimpleNamespace

import numpy as np
import torch

from sheepshead.agent import architectures
from sheepshead.training.config import LeagueConfig, PFSPHyperparams
from sheepshead.training.leaster_watchdog import LeasterWatchdog
from sheepshead.training.league import ROLE_PAST_MAIN, SELF_PLAY, League
from sheepshead.training.pfsp_runtime import (
    interpolated_weight,
    make_game_summary,
    play_population_game,
)
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead import ACTIONS
from sheepshead.training.training_utils import (
    append_csv_row,
    get_partner_selection_mode,
    greedy_health_probe,
    paired_edge,
    set_all_seeds,
    truncate_csv_rows_past_episode,
)

PFSP_HYPERPARAMS = PFSPHyperparams()  # entropy/LR decay schedules + greedy-health gates

# Fixed deal-set seed for the anchored strength probe: every probe replays the
# SAME deals, so consecutive probe values are paired and the trend line is
# policy movement, not deal luck.
LEAGUE_ANCHOR_EVAL_SEED = 20260701


class _Seat:
    """Adapter giving a LeagueMember (or the training agent itself, for
    SELF_PLAY seats) the .agent / .metadata.agent_id surface that
    play_population_game expects of population opponents. The league keeps no
    strategic profiles, so this is the whole opponent surface it needs."""

    def __init__(self, agent: PPOAgent, agent_id: str):
        self.agent = agent
        self.metadata = SimpleNamespace(agent_id=agent_id)


@dataclass
class _Job:
    episode: int
    partner_mode: int
    training_position: int
    opponent_ids: list  # member_id strings; SELF_PLAY for self seats
    weight_version: int
    collect_oracle: bool = False  # attach oracle_state to events (critic_mode=oracle)


# ----------------------------------------------------------------------------
# Worker pool (league flavor of the pfsp_runtime worker protocol: same
# versioned-weights scheme, opponents loaded from the league members dir,
# SELF_PLAY seats played by the worker's own current-weights copy).
# ----------------------------------------------------------------------------
_LWORKER: dict = {}


def _league_worker_init(init_args: dict) -> None:
    import torch as _torch

    _torch.set_num_threads(1)
    agent = PPOAgent(len(ACTIONS), arch=init_args.get("arch", "full"))
    seed = init_args["base_seed"] ^ (os.getpid() & 0xFFFFFFFF)
    random.seed(seed)
    _LWORKER.clear()
    _LWORKER.update(
        {
            "agent": agent,
            "members_dir": init_args["members_dir"],
            "weight_path_base": init_args["weight_path_base"],
            "version": 0,
            "cache": {},
        }
    )


def _league_worker_get_member(member_id: str) -> _Seat:
    cache = _LWORKER["cache"]
    seat = cache.get(member_id)
    if seat is None:
        # Arch-aware: members carry their architecture in checkpoint metadata
        # (legacy members without the key are the full architecture).
        agent = load_agent(os.path.join(_LWORKER["members_dir"], f"{member_id}.pt"))
        seat = _Seat(agent, member_id)
        cache[member_id] = seat
    return seat


def _league_worker_play(job: _Job) -> dict:
    import torch as _torch

    g = _LWORKER
    if job.weight_version > g["version"]:
        ckpt = _torch.load(
            f"{g['weight_path_base']}_v{job.weight_version}.pt", map_location="cpu"
        )
        g["agent"].load_network_states(ckpt, source=f"weights v{job.weight_version}")
        g["version"] = job.weight_version

    opponents = [
        _Seat(g["agent"], SELF_PLAY)
        if mid == SELF_PLAY
        else _league_worker_get_member(mid)
        for mid in job.opponent_ids
    ]
    game, episode_events, final_scores, training_data_single, pos_to_seat = (
        play_population_game(
            training_agent=g["agent"],
            opponents=opponents,
            partner_mode=job.partner_mode,
            training_agent_position=job.training_position,
            reward_mode="terminal",
            collect_oracle=job.collect_oracle,
        )
    )
    return {
        "episode": job.episode,
        "partner_mode": job.partner_mode,
        "training_position": job.training_position,
        "episode_events": episode_events,
        "final_scores": final_scores,
        "training_data_single": training_data_single,
        "game_summary": make_game_summary(game),
        "seat_to_member_id": {
            pos: seat.metadata.agent_id for pos, seat in pos_to_seat.items()
        },
    }


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
@dataclass
class _TxCounter:
    """Mutable box for transitions_since_update.

    Shared between run_main_phase's consuming loop (which increments it and
    resets it to 0 after each PPO update) and parallel_stream's batch-window
    sizing (which reads the live count to decide how many episodes to
    dispatch before the next expected update). A plain int can't be shared
    this way once parallel_stream is a module-level function rather than a
    closure over run_main_phase's locals.
    """

    count: int = 0


@dataclass
class MainPhaseContext:
    """Explicit bundle of the state run_main_phase's nested helpers
    (setup_episode, apply_schedules, sequential_stream, publish_weights,
    parallel_stream) used to close over, now that they are module-level
    functions."""

    training_agent: PPOAgent
    league: League
    rng: random.Random
    args: object
    collect_oracle: bool
    weight_sync: dict
    tx_counter: _TxCounter
    start_episode: int
    end_episode: int


def setup_episode(episode: int, ctx: MainPhaseContext):
    mode = get_partner_selection_mode(episode)
    table = ctx.league.sample_table(mode, ctx.rng)
    position = ctx.rng.randint(1, 5)
    return mode, table, position


def apply_schedules(episode: int, ctx: MainPhaseContext):
    pct = min(100.0, 100.0 * episode / max(ctx.args.schedule_horizon, 1))
    decay = 1.0 - pct / 100.0
    ctx.training_agent.entropy_coeff_pick = (
        PFSP_HYPERPARAMS.entropy_pick_end
        + (PFSP_HYPERPARAMS.entropy_pick_start - PFSP_HYPERPARAMS.entropy_pick_end) * decay
    )
    ctx.training_agent.entropy_coeff_partner = (
        PFSP_HYPERPARAMS.entropy_partner_end
        + (PFSP_HYPERPARAMS.entropy_partner_start - PFSP_HYPERPARAMS.entropy_partner_end) * decay
    )
    ctx.training_agent.entropy_coeff_bury = (
        PFSP_HYPERPARAMS.entropy_bury_end
        + (PFSP_HYPERPARAMS.entropy_bury_start - PFSP_HYPERPARAMS.entropy_bury_end) * decay
    )
    ctx.training_agent.entropy_coeff_play = (
        PFSP_HYPERPARAMS.entropy_play_end
        + (PFSP_HYPERPARAMS.entropy_play_start - PFSP_HYPERPARAMS.entropy_play_end) * decay
    )
    ctx.training_agent.set_learning_rates(
        interpolated_weight(PFSP_HYPERPARAMS.lr_schedule_actor, pct),
        interpolated_weight(PFSP_HYPERPARAMS.lr_schedule_critic, pct),
    )


# -------------------- episode streams --------------------
def sequential_stream(ctx: MainPhaseContext):
    for episode in range(ctx.start_episode + 1, ctx.end_episode + 1):
        mode, table, position = setup_episode(episode, ctx)
        opponents = [
            _Seat(ctx.training_agent, SELF_PLAY)
            if entry == SELF_PLAY
            else _Seat(entry.agent, entry.member_id)
            for entry in table
        ]
        game, events, scores, training_data_single, pos_to_seat = play_population_game(
            training_agent=ctx.training_agent,
            opponents=opponents,
            partner_mode=mode,
            training_agent_position=position,
            reward_mode="terminal",
            collect_oracle=ctx.collect_oracle,
        )
        yield (
            episode,
            mode,
            position,
            events,
            scores,
            training_data_single,
            make_game_summary(game),
            {pos: s.metadata.agent_id for pos, s in pos_to_seat.items()},
        )


def publish_weights(ctx: MainPhaseContext):
    ctx.weight_sync["version"] += 1
    path = f"{ctx.weight_sync['base']}_v{ctx.weight_sync['version']}.pt"
    torch.save(
        {
            "encoder_state_dict": ctx.training_agent.encoder.state_dict(),
            "actor_state_dict": ctx.training_agent.actor.state_dict(),
            "critic_state_dict": ctx.training_agent.critic.state_dict(),
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)
    stale = f"{ctx.weight_sync['base']}_v{ctx.weight_sync['version'] - 2}.pt"
    if os.path.exists(stale):
        try:
            os.remove(stale)
        except OSError:
            pass


def parallel_stream(ctx: MainPhaseContext, pool, num_workers):
    publish_weights(ctx)
    avg_tx_per_game = 26.0
    episode = ctx.start_episode + 1
    while episode <= ctx.end_episode:
        remaining_tx = max(1, ctx.args.update_interval - ctx.tx_counter.count)
        window = max(num_workers, min(256, int(remaining_tx / avg_tx_per_game) + 1))
        end = min(ctx.end_episode, episode + window - 1)
        jobs = []
        for ep in range(episode, end + 1):
            mode, table, position = setup_episode(ep, ctx)
            jobs.append(
                _Job(
                    episode=ep,
                    partner_mode=mode,
                    training_position=position,
                    opponent_ids=[
                        SELF_PLAY if e == SELF_PLAY else e.member_id for e in table
                    ],
                    weight_version=ctx.weight_sync["version"],
                    collect_oracle=ctx.collect_oracle,
                )
            )
        for r in pool.imap(_league_worker_play, jobs):
            yield (
                r["episode"],
                r["partner_mode"],
                r["training_position"],
                r["episode_events"],
                r["final_scores"],
                r["training_data_single"],
                r["game_summary"],
                r["seat_to_member_id"],
            )
        episode = end + 1


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
    watchdog = (
        LeasterWatchdog() if getattr(args, "leaster_watchdog", False) else None
    )
    t0 = time.time()

    progress_csv = os.path.join(checkpoint_dir, "league_training_progress.csv")
    greedy_csv = os.path.join(checkpoint_dir, "greedy_health.csv")
    anchored_csv = os.path.join(checkpoint_dir, "anchored_eval.csv")
    # Crash-resume dedupe: drop telemetry a crashed run wrote past the
    # resume episode, or the replayed episodes would duplicate rows.
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

    pool = None
    if args.num_workers > 1:
        mp_ctx = get_context("spawn")
        pool = mp_ctx.Pool(
            processes=args.num_workers,
            initializer=_league_worker_init,
            initargs=(
                {
                    "arch": getattr(args, "arch", "full"),
                    "members_dir": str(league.members_dir),
                    "weight_path_base": weight_sync["base"],
                    "base_seed": args.seed,
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
            training_agent.store_episode_events(events)
            tx_counter.count += sum(
                1 for ev in events if ev["kind"] == "action"
            )
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
                if watchdog is not None:
                    watchdog.tick(training_agent, leaster_window)
                stats = training_agent.update(epochs=4, batch_size=256)
                tx_counter.count = 0
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
                    print(
                        f"Ep {episode:,} | picker_avg {picker_avg:+.2f} | "
                        f"pick {100 * np.mean(pick_window):.0f}% | "
                        f"leaster {100 * np.mean(leaster_window):.1f}% | "
                        f"x-share {league.exploiter_share():.2f} | "
                        f"advσ all/pick/play "
                        f"{adv_std_all:.3f}/{adv_std_pick:.3f}/{adv_std_play:.3f} | "
                        f"{eps_s:.1f} eps/s{anchor_str}{oracle_str}",
                        flush=True,
                    )
                    write_header = not os.path.exists(progress_csv)
                    with open(progress_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(
                                [
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
                                ]
                            )
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
                            ]
                        )

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
                    f"play-spread {probe['play_logit_spread_med']:.2f}",
                    flush=True,
                )
                if probe["pick_rate"] < PFSP_HYPERPARAMS.greedy_gate_min_pick:
                    print(
                        f"🚨 GREEDY GATE VIOLATION: PICK rate < "
                        f"{PFSP_HYPERPARAMS.greedy_gate_min_pick:.0f}%"
                    )
                if probe["alone_rate"] > PFSP_HYPERPARAMS.greedy_gate_max_alone:
                    print(
                        f"🚨 GREEDY GATE VIOLATION: ALONE rate > "
                        f"{PFSP_HYPERPARAMS.greedy_gate_max_alone:.0f}%"
                    )
                if probe["t0_trump_lead_rate"] > PFSP_HYPERPARAMS.greedy_gate_max_trump_lead:
                    print(
                        f"🚨 GREEDY GATE VIOLATION: trump-lead > "
                        f"{PFSP_HYPERPARAMS.greedy_gate_max_trump_lead:.0f}%"
                    )
                if probe["play_logit_spread_med"] < PFSP_HYPERPARAMS.greedy_gate_min_play_spread:
                    print(
                        "🚨 GREEDY GATE VIOLATION: play-head logit spread < "
                        f"{PFSP_HYPERPARAMS.greedy_gate_min_play_spread} "
                        "(play head collapsing toward uniform)"
                    )
                write_header = not os.path.exists(greedy_csv)
                with open(greedy_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(
                            [
                                "episode",
                                "pick_rate",
                                "alone_rate",
                                "leaster_rate",
                                "t0_trump_lead_rate",
                                "t0_def_leads",
                                "play_logit_spread_med",
                                "play_nodes",
                                "games",
                            ]
                        )
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
                        ]
                    )

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
                training_agent.save(
                    os.path.join(
                        checkpoint_dir,
                        f"pfsp_{getattr(args, 'arch', 'full')}_checkpoint_{episode}.pt",
                    )
                )
                league.save()
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return last_episode


# ----------------------------------------------------------------------------
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


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="League training (main/exploiter generations)"
    )
    ap.add_argument(
        "--resume", required=True, help="main agent checkpoint to start from"
    )
    ap.add_argument("--league-dir", required=True)
    ap.add_argument(
        "--migrate-from",
        default=None,
        help="legacy population dir (used once if league empty)",
    )
    ap.add_argument(
        "--seed-checkpoints",
        default=None,
        help="glob or dir of PPO checkpoints to seed an empty league as "
        "past_mains (e.g. the selfplay bootstrap snapshots that seeded the "
        "original pfsp run: 'runs/reference_selfplay_ppo/checkpoints/*.pt')",
    )
    ap.add_argument("--run-name", default="league_run")
    ap.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of exploiter generations to run from the resume point. "
        "Boundaries are keyed to absolute episode (gen g ends at g*main-episodes), "
        "so the starting generation index is derived from the resumed episode.",
    )
    ap.add_argument("--main-episodes", type=int, default=1_000_000)
    ap.add_argument("--exploiter-episodes", type=int, default=50_000)
    ap.add_argument("--gate-deals", type=int, default=3000)
    ap.add_argument(
        "--screen-deals",
        type=int,
        default=200,
        help="paired deals per exploiter checkpoint for best-of-checkpoints "
        "selection before the full gate (0 = gate the final save only)",
    )
    ap.add_argument("--update-interval", type=int, default=2048)
    ap.add_argument("--save-interval", type=int, default=50_000)
    ap.add_argument("--snapshot-interval", type=int, default=50_000)
    ap.add_argument("--greedy-eval-interval", type=int, default=50_000)
    ap.add_argument("--greedy-eval-games", type=int, default=200)
    ap.add_argument("--schedule-horizon", type=int, default=20_000_000)
    ap.add_argument(
        "--critic-mode",
        choices=["limited", "oracle"],
        default="limited",
        help="'oracle' trains a privileged full-information critic as the GAE "
        "baseline (asymmetric actor-critic; see oracle.py). The actor, the "
        "limited critic, and all aux heads train identically in both modes.",
    )
    ap.add_argument("--anchor-coeff", type=float, default=0.0)
    ap.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="Override GAE lambda (default: agent's 0.95). Phase B of "
        "Learning_System_Redesign_202607 lowers this toward 0.8 once the "
        "stratified-EV gate shows trustworthy mid-game values.",
    )
    ap.add_argument(
        "--table-self-play",
        type=float,
        default=None,
        help="Learning_System_Redesign_202607 table composition: probability "
        "a table is ALL frozen current-self; remainder drawn uniformly from "
        "the recency window + HOF floor (default: historical per-seat "
        "PFSP/self/exploiter sampling)",
    )
    ap.add_argument("--anchor-ref", default=None)
    ap.add_argument(
        "--anchor-eval-ckpt",
        default="final_pfsp_swish_ppo.pt",
        help="frozen reference for the periodic anchored strength probe, the "
        "run's absolute-strength trend line ('' disables)",
    )
    ap.add_argument("--anchor-eval-interval", type=int, default=100_000)
    ap.add_argument("--anchor-eval-deals", type=int, default=300)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--arch",
        default="full",
        choices=architectures.available_architectures(),
        help="Network architecture variant for the training agent, its "
        "snapshots, and the exploiter phase (see the architectures package)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--leaster-watchdog",
        action="store_true",
        help="pick-entropy kick against the always-PASS/leaster collapse "
        "(seen re-entered from a trained policy in anchor-free stage-1 "
        "generations; see leaster_watchdog.py). Main phases only — the "
        "exploiter phase never runs it. Enable uniformly across the arms "
        "of any comparison.",
    )
    return ap


def main():
    args = build_arg_parser().parse_args()

    set_all_seeds(args.seed)

    run_dir = os.path.join("runs", args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    league_config = LeagueConfig()
    if args.table_self_play is not None:
        league_config.table_self_play_prob = args.table_self_play
        print(
            f"🎲 Table-level composition ON: {args.table_self_play:.0%} pure-self "
            "tables, remainder uniform window (PFSP/EMA/exploiter seating off)"
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
    )
    training_agent.load(args.resume, load_optimizers=True)
    if args.gae_lambda is not None:
        training_agent.gae_lambda = float(args.gae_lambda)
        print(f"λ  GAE lambda override: {training_agent.gae_lambda}")
    if args.arch != "full":
        print(f"🧬 Architecture: {args.arch}")
    if args.critic_mode == "oracle":
        print("🔮 Oracle critic ON: privileged full-information GAE baseline")
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
    for generation in range(first_gen, first_gen + args.generations):
        boundary = generation * main_ep
        print(
            f"\n{'=' * 70}\n🏁 GENERATION {generation}: main phase "
            f"({episode:,} -> {boundary:,})\n{'=' * 70}"
        )
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
        main_ckpt = os.path.join(
            checkpoint_dir,
            f"pfsp_{getattr(args, 'arch', 'full')}_checkpoint_{episode}.pt",
        )
        if not os.path.exists(main_ckpt):
            training_agent.save(main_ckpt)

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
