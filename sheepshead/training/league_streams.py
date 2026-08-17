"""Episode streams for run_main_phase: the sequential (in-process) and
parallel (worker-pool) generators that produce one training episode's
results at a time, plus the shared context/state types they close over.

Split out of train_league_ppo.py as pure code motion (Stage 1 of the
league-trainer maintainability refactor). apply_schedules and
fresh_entropy_targets stay in train_league_ppo.py (entropy/schedule
concerns, not stream mechanics).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sheepshead.agent.ppo import PPOAgent
from sheepshead.training.league import SELF_PLAY, League
from sheepshead.training.league_teacher import _teacher_kwargs
from sheepshead.training.league_worker import (
    OpponentAdapter,
    _Job,
    _league_worker_play,
    publish_weights,
)
from sheepshead.training.pfsp_runtime import make_game_summary, play_population_game
from sheepshead.training.training_utils import get_partner_selection_mode

# parallel_stream's dispatch-window sizing formula (jobs to submit per
# pool.imap window, sized off the expected transitions/game).
AVG_TX_PER_GAME = 26.0


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


def rotation_plan(episode: int, start_episode: int, rot_state: dict | None, ctx):
    """Seat-rotation grouping shared by sequential_stream and
    parallel_stream: groups of 5 consecutive episodes share one sampled
    (mode, table, deal); the hero plays every seat of the same deal against
    the same opponents (the train-time duplicate instrument). The deal seed
    is drawn once per group so the cards are identical across the 5
    rotations. When --seat-rotation is off, falls through to a fresh
    setup_episode draw every call (no grouping, no game_seed).

    Returns (mode, table, position, game_seed, rot_state) — rot_state is
    the (possibly newly created) group state to pass into the next call in
    the group; callers persist it across calls as needed (sequential_stream
    a local var, parallel_stream on ctx._rot_state so it survives a window
    split).
    """
    rotate = bool(getattr(ctx.args, "seat_rotation", False))
    game_seed = None
    if rotate:
        phase = (episode - start_episode - 1) % 5
        if phase == 0 or not rot_state:
            mode, table, _ = setup_episode(episode, ctx)
            rot_state = {
                "mode": mode,
                "table": table,
                "seed": random.randrange(2**31),
            }
        mode, table = rot_state["mode"], rot_state["table"]
        position = phase + 1
        game_seed = rot_state["seed"]
    else:
        mode, table, position = setup_episode(episode, ctx)
    return mode, table, position, game_seed, rot_state


def sequential_stream(ctx: MainPhaseContext):
    # Seat rotation (deal-paired collection): groups of 5 consecutive
    # episodes share one sampled (mode, table, deal); the hero plays every
    # seat of the same deal against the same opponents — the train-time
    # duplicate instrument. The deal seed is drawn once per group so the
    # cards are identical across the 5 rotations.
    rot_state = {}
    teacher_kwargs = _teacher_kwargs(ctx)
    for episode in range(ctx.start_episode + 1, ctx.end_episode + 1):
        mode, table, position, game_seed, rot_state = rotation_plan(
            episode, ctx.start_episode, rot_state, ctx
        )
        opponents = [
            OpponentAdapter(ctx.training_agent, SELF_PLAY)
            if entry == SELF_PLAY
            else OpponentAdapter(entry.agent, entry.member_id)
            for entry in table
        ]
        game, events, scores, training_data_single, pos_to_seat = play_population_game(
            training_agent=ctx.training_agent,
            opponents=opponents,
            partner_mode=mode,
            training_agent_position=position,
            reward_mode="terminal",
            collect_oracle=ctx.collect_oracle,
            game_seed=game_seed,
            **teacher_kwargs,
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


def parallel_stream(ctx: MainPhaseContext, pool, num_workers):
    publish_weights(ctx)
    episode = ctx.start_episode + 1
    while episode <= ctx.end_episode:
        remaining_tx = max(1, ctx.args.update_interval - ctx.tx_counter.count)
        window = max(num_workers, min(256, int(remaining_tx / AVG_TX_PER_GAME) + 1))
        end = min(ctx.end_episode, episode + window - 1)
        jobs = []
        rot_state = getattr(ctx, "_rot_state", None)
        for ep in range(episode, end + 1):
            mode, table, position, game_seed, rot_state = rotation_plan(
                ep, ctx.start_episode, rot_state, ctx
            )
            ctx._rot_state = rot_state
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
                    game_seed=game_seed,
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
