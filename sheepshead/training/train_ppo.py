#!/usr/bin/env python3
"""The PPO trainer of the release-candidate training program
(Training_Program_Redesign_202609 §4, §6): one loop, three phases.

  bootstrap   Phase 0. Shaped self-play from scratch: intermediate trick
              rewards + leaster bonus (reward_shaping.py), limited critic,
              an EMPTY population (every seat is the training agent), the
              bootstrap cadence (BootstrapHyperparams), leaster watchdog.
  league      Phase 2. Terminal-only reward, privileged (oracle) critic as
              the GAE baseline, per-seat PFSP population with a self share,
              seat-rotated deal-paired collection, the retention cadence
              (LeagueHyperparams), the target-entropy controller from
              generation 2 on. One invocation trains to --until (an absolute
              episode count, so generation boundaries and crash restarts
              share one clock) and promotes its boundary snapshot to a HOF
              anchor.
  bidding     The bidding-only PG phase between policy-iteration rounds
              (§4.4): the league phase with encoder, actor adapter and play
              head FROZEN (PPOAgent.set_trainable_heads) so the pick /
              partner / call heads re-optimize on terminal reward under the
              improved play without touching what search installed.

Every phase produces the same artifacts under runs/<run-name>/:
checkpoints/checkpoint_<episode>.pt, checkpoints/training_progress.csv,
checkpoints/greedy_health.csv, the population under --league-dir, and
final.pt at the end. The orchestrator (run_training_program.py) chains
phases; this module never decides when a phase is over.

Literature the loop rests on: PPO (Schulman et al. 2017) with GAE
(Schulman et al. 2016); asymmetric / centralized critics (Pinto et al.
2017; Yu et al. 2021) — the oracle; prioritized fictitious self-play over
snapshots (Vinyals et al. 2019) with an OpenAI-Five-style self share
(Berner et al. 2019); SAC automatic temperature (Haarnoja et al. 2018) in
its discrete form for the entropy controller.

Usage (bootstrap from scratch, then one league generation):
  uv run python -m sheepshead.training.train_ppo --phase bootstrap \\
      --arch perceiver-recall --run-name rc/bootstrap --until 400000
  uv run python -m sheepshead.training.train_ppo --phase league \\
      --resume runs/rc/bootstrap/final.pt --run-name rc/league \\
      --seed-checkpoints 'runs/rc/seeds/*.pt' --until 1000000 \\
      --no-entropy-controller --oracle-init runs/rc/oracle/oracle_init.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import torch

from sheepshead import ACTIONS
from sheepshead.agent import architectures
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.training.config import (
    BootstrapHyperparams,
    LeagueConfig,
    LeagueHyperparams,
)
from sheepshead.training.entropy_controller import (
    EntropyControllerConfig,
    EntropyTargetController,
)
from sheepshead.training.league import ROLE_PAST_MAIN, SELF_PLAY, League
from sheepshead.training.league_streams import (
    MainPhaseContext,
    TransitionCounter,
    parallel_stream,
    sequential_stream,
)
from sheepshead.training.league_worker import league_worker_init, publish_weights
from sheepshead.training.leaster_watchdog import LeasterWatchdog
from sheepshead.training.training_utils import (
    ensure_csv_columns,
    greedy_health_probe,
    set_all_seeds,
    truncate_csv_rows_past_episode,
)

PHASES = ("bootstrap", "league", "bidding")

# training_progress.csv schema (append-only: add at the end, never rename;
# ensure_csv_columns migrates pre-existing files on resume).
PROGRESS_CSV_HEADER = [
    "episode",
    "picker_avg",
    "pick_rate",
    "leaster_rate",
    "mu_jd",
    "mu_ca",
    "adv_std_all",
    "adv_std_pick",
    "adv_std_play",
    "ev_oracle",
    "ev_limited",
    "opt_steps",
    # theta_old per-node H/ln(n_legal) means per head (forced moves excluded)
    # and the soft-band fractions (share of eligible nodes with H_norm > 0.3,
    # ppo.SOFTBAND_HNORM): what the entropy controller reads, and the
    # boundary-band collapse canary the mean hides.
    "ent_norm_pick",
    "ent_norm_partner",
    "ent_norm_bury",
    "ent_norm_play",
    "softband_pick",
    "softband_partner",
    "softband_bury",
    "softband_play",
    "approx_kl",
    "lr_actor",
    "eps_per_s",
]

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


def checkpoint_path(checkpoint_dir: str, episode: int) -> str:
    return os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt")


def episode_of(path: str) -> int:
    """Episode count encoded in a checkpoint filename (0 for final.pt-style
    names, which carry no clock)."""
    base = os.path.basename(path)
    if "checkpoint_" in base:
        try:
            return int(base.split("checkpoint_")[-1].split(".")[0])
        except ValueError:
            return 0
    return 0


# ----------------------------------------------------------------------------
# Phase presets
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseSpec:
    """What distinguishes the three phases; everything else is shared."""

    name: str
    reward_mode: str  # "shaped" | "terminal"
    critic_mode: str  # "limited" | "oracle"
    trainable_heads: str  # "all" | "bidding"
    seat_rotation: bool
    grad_accum: bool
    gamma: float


PHASE_SPECS = {
    "bootstrap": PhaseSpec(
        name="bootstrap",
        reward_mode="shaped",
        critic_mode="limited",
        trainable_heads="all",
        seat_rotation=False,  # every seat already collects in pure self-play
        grad_accum=False,
        gamma=0.99,  # the validated shaped-reward regime
    ),
    "league": PhaseSpec(
        name="league",
        reward_mode="terminal",
        critic_mode="oracle",
        trainable_heads="all",
        seat_rotation=True,
        grad_accum=True,
        gamma=1.0,  # undiscounted terminal returns (Redesign §7.1)
    ),
    "bidding": PhaseSpec(
        name="bidding",
        reward_mode="terminal",
        critic_mode="oracle",
        trainable_heads="bidding",
        seat_rotation=True,
        grad_accum=True,
        gamma=1.0,
    ),
}


def hyperparams_for(phase: str):
    return BootstrapHyperparams() if phase == "bootstrap" else LeagueHyperparams()


# ----------------------------------------------------------------------------
# Per-update schedules
# ----------------------------------------------------------------------------
def apply_schedules(episode: int, context: MainPhaseContext) -> None:
    """Set the agent's learning rates and (schedule-owned) entropy
    coefficients for this update. The bootstrap decays its coefficients
    linearly over its own length (BootstrapHyperparams); the league phases
    hold the fixed LeagueHyperparams values unless the entropy controller
    owns them (it overwrites these right after, see _ppo_update)."""
    args = context.args
    hp = context.hyperparams
    agent = context.training_agent
    if isinstance(hp, BootstrapHyperparams):
        frac = min(1.0, episode / max(int(args.until), 1))
        agent.entropy_coeff_pick = (
            hp.entropy_pick_start + (hp.entropy_pick_end - hp.entropy_pick_start) * frac
        )
        agent.entropy_coeff_partner = (
            hp.entropy_partner_start
            + (hp.entropy_partner_end - hp.entropy_partner_start) * frac
        )
        agent.entropy_coeff_bury = (
            hp.entropy_bury_start + (hp.entropy_bury_end - hp.entropy_bury_start) * frac
        )
        agent.entropy_coeff_play = (
            hp.entropy_play_start + (hp.entropy_play_end - hp.entropy_play_start) * frac
        )
    else:
        agent.entropy_coeff_pick = hp.entropy_pick
        agent.entropy_coeff_partner = hp.entropy_partner
        agent.entropy_coeff_bury = hp.entropy_bury
        agent.entropy_coeff_play = hp.entropy_play
    agent.set_learning_rates(hp.lr_actor, hp.lr_critic)


def store_events_by_seat(agent: PPOAgent, events: list) -> int:
    """Store one episode's events as one coherent stream PER COLLECTING SEAT.

    play_population_game returns all collecting players' events in a single
    temporally-interleaved list; the recurrent update needs one memory per
    perspective (per-seat streams — the braided-storage fix of
    Learning_System_Redesign §6.1). Returns the number of action rows."""
    by_player: dict[int, list] = {}
    for ev in events:
        by_player.setdefault(ev["player_id"], []).append(ev)
    n_actions = 0
    for pid in sorted(by_player):
        agent.store_episode_events(by_player[pid])
        n_actions += sum(1 for ev in by_player[pid] if ev["kind"] == "action")
    return n_actions


def inherited_ratings(league: League, training_ratings: dict) -> dict:
    """Per-mode ratings for a new snapshot, seeded from the training agent's
    current rating rather than the mu=25 prior (a fresh prior outranks every
    rated member and turns skill pruning into newest-wins — League_Run_Review
    F1). Sigma is floored at half the prior so the copy is still re-rated."""
    min_sigma = league.rating_model.rating().sigma / 2.0
    return {
        mode: league.rating_model.rating(mu=r.mu, sigma=max(r.sigma, min_sigma))
        for mode, r in training_ratings.items()
    }


# ----------------------------------------------------------------------------
# Phase state and helpers
# ----------------------------------------------------------------------------
@dataclass
class _PhaseState:
    context: MainPhaseContext
    checkpoint_dir: str
    training_ratings: dict
    entropy_controller: EntropyTargetController | None
    entropy_controller_path: str
    watchdog: LeasterWatchdog | None
    pool: object | None
    progress_csv: str
    greedy_csv: str
    picker_scores: deque
    pick_window: deque
    leaster_window: deque
    start_time: float


def _setup_telemetry_csvs(checkpoint_dir: str, start_episode: int):
    progress_csv = os.path.join(checkpoint_dir, "training_progress.csv")
    greedy_csv = os.path.join(checkpoint_dir, "greedy_health.csv")
    if ensure_csv_columns(progress_csv, PROGRESS_CSV_HEADER):
        print("📊 Migrated training_progress.csv to wider schema")
    if ensure_csv_columns(greedy_csv, GREEDY_CSV_HEADER):
        print("📊 Migrated greedy_health.csv to wider schema")
    for csv_file in (progress_csv, greedy_csv):
        n_trimmed = truncate_csv_rows_past_episode(csv_file, start_episode)
        if n_trimmed:
            print(
                f"🧹 Trimmed {n_trimmed} stale rows past episode {start_episode:,} "
                f"from {os.path.basename(csv_file)}"
            )
    return progress_csv, greedy_csv


def _setup_entropy_controller(args, checkpoint_dir: str, context: MainPhaseContext):
    """The target-entropy controller (SAC automatic temperature, discrete
    form; entropy_controller.py) owns the coefficients when enabled:
    bumpless attach — alpha from the phase's fixed values, targets from the
    first measured H_norm. The sidecar next to the checkpoints carries its
    state across restarts, and the orchestrator edits the same file for
    the single play-target step (Training_Program_Redesign §5.1)."""
    path = os.path.join(checkpoint_dir, "entropy_controller.json")
    if not getattr(args, "entropy_controller", False):
        return None, path
    if os.path.exists(path):
        controller = EntropyTargetController.load(path)
        print(
            f"🎯 Entropy controller resumed: targets {controller.targets}  "
            f"alphas {controller.alphas}"
        )
    else:
        controller = EntropyTargetController(
            config=EntropyControllerConfig(
                floors={"play": float(getattr(args, "entropy_play_floor", 0.28))}
            )
        )
        print("🎯 Entropy controller fresh (bumpless targets pending)")
    apply_schedules(context.start_episode, context)
    controller.attach(context.training_agent)
    return controller, path


def _spawn_worker_pool(args, league: League, context: MainPhaseContext):
    """The versioned-weights worker pool (league_worker protocol), or None
    for the in-process sequential path (num_workers <= 1)."""
    inference_flags = (
        getattr(args, "worker_compile", None),
        getattr(args, "worker_device", None),
    )
    if args.num_workers <= 1:
        if any(inference_flags):
            print(
                "⚠️  --worker-compile/--worker-device ignored: they configure "
                "the worker pool, and --num-workers <= 1 runs in-process"
            )
        return None
    if any(inference_flags):
        print(
            "⚡ worker inference: "
            f"device={getattr(args, 'worker_device', None) or 'process default'}, "
            f"compile={getattr(args, 'worker_compile', None) or 'off'} "
            "(throughput only; not bit-comparable with an eager run)"
        )
    return get_context("spawn").Pool(
        processes=args.num_workers,
        initializer=league_worker_init,
        initargs=(
            {
                "arch": context.training_agent.arch_name,
                "members_dir": str(league.members_dir),
                "weight_path_base": context.weight_sync["base"],
                "base_seed": args.seed,
                "critic_mode": "limited",  # workers never train the oracle
                "oracle_aux_heads": False,
                "reward_mode": context.reward_mode,
                "worker_device": getattr(args, "worker_device", None),
                "worker_compile": getattr(args, "worker_compile", None),
                "worker_compile_granularity": getattr(
                    args, "worker_compile_granularity", 32
                ),
            },
        ),
    )


def _ingest_episode(
    state: _PhaseState,
    mode: int,
    position: int,
    events: list,
    scores: list[float],
    training_data_single: dict,
    summary: dict,
    seat_to_id: dict,
) -> None:
    league = state.context.league
    state.context.tx_counter.count += store_events_by_seat(
        state.context.training_agent, events
    )
    if training_data_single["was_picker"]:
        state.picker_scores.append(training_data_single["score"])
    state.pick_window.append(1 if training_data_single["was_picker"] else 0)
    state.leaster_window.append(1 if summary["is_leaster"] else 0)
    members_by_pos = {
        pos: member
        for pos, member_id in seat_to_id.items()
        if member_id != SELF_PLAY and (member := league.get(member_id)) is not None
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
    training_agent = state.context.training_agent
    eps_per_s = (episode - state.context.start_episode) / max(
        time.time() - state.start_time, 1e-9
    )
    picker_avg = float(np.mean(state.picker_scores)) if state.picker_scores else 0.0
    advantage_stats = stats.get("advantage_stats", {})
    head_std = advantage_stats.get("head_std", {})
    adv_std_all = advantage_stats.get("std", 0.0)
    adv_std_play = head_std.get("play", 0.0)
    adv_std_pick = head_std.get("pick", 0.0)
    oracle_stats = stats.get("oracle") or {}
    oracle_str = (
        f"  ev O/L {oracle_stats['ev_oracle']:.2f}/{oracle_stats['ev_limited']:.2f}"
        if oracle_stats
        else ""
    )
    head_entropy = stats.get("head_entropy_norm") or {}
    head_softband = stats.get("head_softband") or {}

    def fmt(v):
        return f"{v:.2f}" if v is not None else "-"

    hnorm_str = (
        " | Hn "
        + "/".join(
            fmt(head_entropy.get(h)) for h in ("pick", "partner", "bury", "play")
        )
        if head_entropy
        else ""
    )
    print(
        f"Ep {episode:,} | picker_avg {picker_avg:+.2f} | "
        f"pick {100 * np.mean(state.pick_window):.0f}% | "
        f"leaster {100 * np.mean(state.leaster_window):.1f}% | "
        f"advσ all/pick/play {adv_std_all:.3f}/{adv_std_pick:.3f}/{adv_std_play:.3f} | "
        f"{eps_per_s:.1f} eps/s{oracle_str}{hnorm_str}",
        flush=True,
    )
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
                f"{state.training_ratings[0].mu:.2f}",
                f"{state.training_ratings[1].mu:.2f}",
                f"{adv_std_all:.4f}",
                f"{adv_std_pick:.4f}",
                f"{adv_std_play:.4f}",
                f"{oracle_stats['ev_oracle']:.4f}" if oracle_stats else "",
                f"{oracle_stats['ev_limited']:.4f}" if oracle_stats else "",
                stats.get("optimizer_steps_total", ""),
                *[
                    f"{head_entropy[h]:.4f}" if head_entropy.get(h) is not None else ""
                    for h in ("pick", "partner", "bury", "play")
                ],
                *[
                    f"{head_softband[h]:.4f}"
                    if head_softband.get(h) is not None
                    else ""
                    for h in ("pick", "partner", "bury", "play")
                ],
                f"{stats.get('approx_kl', 0.0):.6f}",
                f"{training_agent.actor_optimizer.param_groups[0]['lr']:.2e}",
                f"{eps_per_s:.2f}",
            ]
        )


def _ppo_update(state: _PhaseState, episode: int) -> None:
    """One PPO update: schedules, controller, watchdog kick, the gradient
    update, weight republish, progress row."""
    training_agent = state.context.training_agent
    hp = state.context.hyperparams
    spec = PHASE_SPECS[state.context.args.phase]
    apply_schedules(episode, state.context)
    if state.entropy_controller is not None:
        # The controller owns the coefficients; the watchdog kick below
        # still multiplies on top (it stays the upward override).
        state.entropy_controller.apply(training_agent)
    if state.watchdog is not None:
        state.watchdog.tick(training_agent, state.leaster_window)
    stats = training_agent.update(
        epochs=hp.ppo_epochs,
        batch_size=hp.minibatch_episodes,
        grad_accum=spec.grad_accum,
        oracle_extra_epochs=getattr(hp, "oracle_extra_epochs", 0),
    )
    state.context.tx_counter.count = 0
    if state.entropy_controller is not None and stats:
        pending = [
            h
            for h in ("pick", "partner", "bury", "play")
            if state.entropy_controller.targets[h] is None
        ]
        state.entropy_controller.observe(stats.get("head_entropy_norm") or {})
        if pending and not any(
            state.entropy_controller.targets[h] is None for h in pending
        ):
            print(
                "🎯 Entropy targets initialized (bumpless): "
                + "  ".join(
                    f"{h} {state.entropy_controller.targets[h]:.3f}"
                    for h in ("pick", "partner", "bury", "play")
                )
            )
        state.entropy_controller.save(state.entropy_controller_path)
    if state.pool is not None:
        publish_weights(state.context)
    if stats:
        _emit_progress(state, episode, stats)


def _run_interval_probes(state: _PhaseState, episode: int) -> None:
    args = state.context.args
    training_agent = state.context.training_agent
    league = state.context.league
    hp = state.context.hyperparams

    if args.snapshot_interval > 0 and episode % args.snapshot_interval == 0:
        snapshot = copy.deepcopy(training_agent)
        snapshot.strip_oracle()  # members are inference-only
        league.add_member(
            snapshot,
            ROLE_PAST_MAIN,
            training_episodes=episode,
            initial_ratings=inherited_ratings(league, state.training_ratings),
        )
        print(f"👥 League snapshot at ep {episode:,}; {league.summary()}")

    if args.greedy_eval_interval > 0 and episode % args.greedy_eval_interval == 0:
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
        if isinstance(hp, LeagueHyperparams):
            if probe["pick_rate"] < hp.greedy_gate_min_pick:
                print("🚨 GREEDY GATE VIOLATION: PICK rate below gate", flush=True)
            if probe["alone_rate"] > hp.greedy_gate_max_alone:
                print("🚨 GREEDY GATE VIOLATION: ALONE rate above gate", flush=True)
            if probe["t0_trump_lead_rate"] > hp.greedy_gate_max_trump_lead:
                print("🚨 GREEDY GATE VIOLATION: trump-lead above gate", flush=True)
            if probe["play_logit_spread_med"] < hp.greedy_gate_min_play_spread:
                print(
                    "🚨 GREEDY GATE VIOLATION: play-head logit spread below gate "
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

    if args.save_interval > 0 and episode % args.save_interval == 0:
        training_agent.save(checkpoint_path(state.checkpoint_dir, episode))
        league.save()


# ----------------------------------------------------------------------------
# The phase loop
# ----------------------------------------------------------------------------
def run_phase(
    training_agent: PPOAgent,
    league: League,
    training_ratings: dict,
    args,
    start_episode: int,
    end_episode: int,
    checkpoint_dir: str,
) -> int:
    """Train ``training_agent`` from ``start_episode`` to ``end_episode``
    under the phase named by ``args.phase``; returns the last episode
    played. Mutates the league's ratings/EMAs and ``training_ratings``.

    ``args`` needs: phase, seed, run_name, num_workers, update_interval
    (transitions per update), save_interval, snapshot_interval,
    greedy_eval_interval, greedy_eval_games, leaster_watchdog,
    entropy_controller (+ entropy_play_floor), until, and optionally the
    worker inference flags."""
    spec = PHASE_SPECS[args.phase]
    hp = hyperparams_for(args.phase)
    context = MainPhaseContext(
        training_agent=training_agent,
        league=league,
        rng=random.Random(args.seed + start_episode),
        args=args,
        collect_oracle=spec.critic_mode == "oracle",
        weight_sync={
            "version": 0,
            "base": os.path.join("runs", args.run_name, "_league_worker_weights"),
        },
        tx_counter=TransitionCounter(),
        start_episode=start_episode,
        end_episode=end_episode,
        hyperparams=hp,
        reward_mode=spec.reward_mode,
        seat_rotation=spec.seat_rotation,
    )
    progress_csv, greedy_csv = _setup_telemetry_csvs(checkpoint_dir, start_episode)
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
        entropy_controller=entropy_controller,
        entropy_controller_path=entropy_controller_path,
        watchdog=(
            LeasterWatchdog() if getattr(args, "leaster_watchdog", False) else None
        ),
        pool=pool,
        progress_csv=progress_csv,
        greedy_csv=greedy_csv,
        picker_scores=deque(maxlen=3000),
        pick_window=deque(maxlen=3000),
        leaster_window=deque(maxlen=3000),
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
    if training_agent.events:
        # No flush update: a partial buffer would train under different
        # hyperparameters than every other update (Architecture_Ablation §4.6).
        n = len(training_agent.events)
        training_agent.reset_storage()
        print(f"   Discarding {n} leftover buffered events (no flush update)")
    return last_episode


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def seed_league_from_checkpoints(league: League, spec: str) -> None:
    """Seed an empty league with checkpoints as past_mains. ``spec`` is a
    glob (``.../*.pt``) or a directory. A single warm-start checkpoint is
    usually passed as several copies: sample_table draws members without
    replacement, so one member can fill only one seat."""
    paths = (
        sorted(glob.glob(spec))
        if any(c in spec for c in "*?[")
        else sorted(glob.glob(os.path.join(spec, "*.pt")))
    )
    if not paths:
        raise SystemExit(f"--seed-checkpoints matched no .pt files: {spec}")
    for p in paths:
        league.add_member(
            load_agent(p), ROLE_PAST_MAIN, training_episodes=episode_of(p)
        )
    print(f"🌱 Seeded league with {len(paths)} checkpoints as past_mains")


def warn_if_oracle_overwrite(agent: PPOAgent, oracle_init: str, resume: str) -> None:
    """Loud banner before --oracle-init clobbers a trained oracle: the flag
    exists for resuming PRE-oracle checkpoints (gen 1 of the league phase);
    applying it to a checkpoint that already restored oracle weights
    downgrades a trained critic to the pretrain."""
    if getattr(agent, "oracle_loaded_from_checkpoint", False):
        print(
            f"⚠️  --oracle-init {oracle_init} is OVERWRITING the trained oracle "
            f"critic restored from {resume}. This flag is meant for pre-oracle "
            "checkpoints; drop it unless the downgrade is intentional."
        )


def build_training_agent(args) -> tuple[PPOAgent, int]:
    """Construct (or resume) the training agent for ``args.phase``; returns
    (agent, start episode)."""
    spec = PHASE_SPECS[args.phase]
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        arch = ckpt.get("arch", "full")
        agent = PPOAgent(
            len(ACTIONS),
            critic_mode=spec.critic_mode,
            arch=arch,
            oracle_aux_heads=bool(getattr(args, "oracle_aux_heads", True)),
        )
        agent.load(args.resume, load_optimizers=True, checkpoint=ckpt)
        start_episode = episode_of(args.resume)
        print(f"📍 Resumed {args.resume} (episode {start_episode:,}, arch {arch})")
    else:
        if args.phase != "bootstrap":
            raise SystemExit(f"--phase {args.phase} needs --resume")
        hp = BootstrapHyperparams()
        agent = PPOAgent(
            len(ACTIONS),
            lr_actor=hp.lr_actor,
            lr_critic=hp.lr_critic,
            critic_mode=spec.critic_mode,
            arch=args.arch,
        )
        start_episode = 0
        n = sum(
            p.numel()
            for net in (agent.encoder, agent.actor, agent.critic)
            for p in net.parameters()
        )
        print(f"🆕 Fresh {args.arch} agent ({n:,} parameters)")
    agent.gamma = spec.gamma
    agent.set_trainable_heads(spec.trainable_heads)
    if getattr(args, "oracle_init", None):
        warn_if_oracle_overwrite(agent, args.oracle_init, args.resume)
        state_dict = torch.load(args.oracle_init, map_location="cpu", weights_only=True)
        agent.oracle_critic.load_state_dict(state_dict, strict=True)
        print(f"🔮⚡ Oracle warm-started from {args.oracle_init}")
    print(
        f"🏁 Phase {args.phase}: reward {spec.reward_mode}, critic {spec.critic_mode}, "
        f"heads {spec.trainable_heads}, gamma {spec.gamma}, "
        f"seat rotation {'on' if spec.seat_rotation else 'off'}"
    )
    return agent, start_episode


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument(
        "--until", type=int, required=True, help="train to this ABSOLUTE episode"
    )
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument(
        "--arch",
        default="perceiver-recall",
        choices=architectures.available_architectures(),
        help="architecture of a fresh bootstrap agent (resumed phases read it "
        "from the checkpoint)",
    )
    p.add_argument("--league-dir", default=None, help="default runs/<run-name>/league")
    p.add_argument(
        "--seed-checkpoints",
        default=None,
        help="glob or dir of checkpoints seeding an EMPTY league as past_mains",
    )
    p.add_argument(
        "--oracle-init",
        default=None,
        help="pretrained oracle state_dict (pretrain_oracle.py) loaded after --resume",
    )
    p.add_argument(
        "--oracle-aux-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build the oracle critic with the team membership / team points "
        "aux heads (Learning_System_Redesign §4.3)",
    )
    p.add_argument(
        "--update-interval",
        type=int,
        default=None,
        help="transitions per PPO update (default: the phase's hyperparams)",
    )
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="episodes between population snapshots (default: 50000 for the "
        "league phases, never for the bootstrap; 0 = never)",
    )
    p.add_argument("--greedy-eval-interval", type=int, default=50_000)
    p.add_argument("--greedy-eval-games", type=int, default=200)
    p.add_argument(
        "--entropy-controller",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="target-entropy controller owns the coefficients (default: on "
        "for the league phases, off for the bootstrap; the orchestrator "
        "passes --no-entropy-controller for league generation 1)",
    )
    p.add_argument("--entropy-play-floor", type=float, default=0.28)
    p.add_argument(
        "--leaster-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pick-entropy kick against the all-PASS/leaster collapse",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--worker-device",
        default=None,
        help="device for worker inference (e.g. mps); throughput only",
    )
    p.add_argument(
        "--worker-compile",
        nargs="?",
        const="default",
        default=None,
        help="torch.compile the encoder in workers; throughput only",
    )
    p.add_argument("--worker-compile-granularity", type=int, default=32)
    return p


def resolve_args(args) -> None:
    """Fill phase-dependent defaults in place."""
    hp = hyperparams_for(args.phase)
    if args.update_interval is None:
        args.update_interval = hp.update_interval
    if args.snapshot_interval is None:
        args.snapshot_interval = 0 if args.phase == "bootstrap" else 50_000
    if args.entropy_controller is None:
        args.entropy_controller = args.phase != "bootstrap"
    if args.league_dir is None:
        args.league_dir = os.path.join("runs", args.run_name, "league")


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    resolve_args(args)
    set_all_seeds(args.seed)
    run_dir = os.path.join("runs", args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    league = League(args.league_dir, LeagueConfig())
    if len(league) == 0 and args.seed_checkpoints:
        seed_league_from_checkpoints(league, args.seed_checkpoints)
    if len(league) == 0:
        print("♟️  Empty population: every opponent seat is the training agent")
    else:
        print(league.summary())
    agent, start_episode = build_training_agent(args)
    if start_episode >= args.until:
        print(f"nothing to do: resumed at {start_episode:,} >= --until {args.until:,}")
        return 0
    ratings = {mode: league.rating_model.rating() for mode in (0, 1)}
    print(f"🎮 Training {start_episode:,} -> {args.until:,} ({args.phase})")
    episode = run_phase(
        agent, league, ratings, args, start_episode, args.until, checkpoint_dir
    )
    final_ckpt = checkpoint_path(checkpoint_dir, episode)
    if not os.path.exists(final_ckpt):
        agent.save(final_ckpt)
    if args.phase != "bootstrap":
        # The boundary snapshot is the generation's HOF anchor
        # (Training_Program_Redesign §4.3): every boundary joins the
        # anti-forgetting floor, quota by rating.
        snapshot = copy.deepcopy(agent)
        snapshot.strip_oracle()
        member_id = league.add_member(
            snapshot,
            ROLE_PAST_MAIN,
            training_episodes=episode,
            initial_ratings=inherited_ratings(league, ratings),
        )
        league.promote_to_hof(member_id)
        league.save()
        print(f"🏛️  Boundary snapshot {member_id} promoted to HOF anchor")
    agent.save(os.path.join(run_dir, "final.pt"))
    print(f"✅ Phase {args.phase} complete at episode {episode:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
