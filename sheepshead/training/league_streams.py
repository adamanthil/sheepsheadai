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
from sheepshead.training.config import PFSPHyperparams
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
class TransitionCounter:
    """Mutable box for transitions_since_update.

    Shared between run_main_phase's consuming loop (which increments it and
    resets it to 0 after each PPO update) and parallel_stream's batch-window
    sizing (which reads the live count to decide how many episodes to
    dispatch before the next expected update). A plain int can't be shared
    this way once parallel_stream is a module-level function rather than a
    closure over run_main_phase's locals.
    """

    count: int = 0


# Re-exported for compatibility: tests/callers import the historical name.
_TxCounter = TransitionCounter


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
    tx_counter: TransitionCounter
    start_episode: int
    end_episode: int
    # Seat-rotation group state carried across parallel_stream's dispatch
    # windows (a window boundary can cut through a 5-episode group).
    rot_state: dict | None = None
    # Schedule/gate hyperparameters. None = the trainer's module-level
    # PFSP_HYPERPARAMS singleton; tests inject a custom instance here
    # instead of monkeypatching the module.
    hyperparams: PFSPHyperparams | None = None


def setup_episode(episode: int, context: MainPhaseContext):
    mode = get_partner_selection_mode(episode)
    table = context.league.sample_table(mode, context.rng)
    position = context.rng.randint(1, 5)
    return mode, table, position


def rotation_plan(episode: int, start_episode: int, rot_state: dict | None, context):
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
    a local var, parallel_stream on context.rot_state so it survives a window
    split).
    """
    rotate = bool(getattr(context.args, "seat_rotation", False))
    game_seed = None
    if rotate:
        phase = (episode - start_episode - 1) % 5
        if phase == 0 or not rot_state:
            mode, table, _ = setup_episode(episode, context)
            rot_state = {
                "mode": mode,
                "table": table,
                "seed": random.randrange(2**31),
            }
        mode, table = rot_state["mode"], rot_state["table"]
        position = phase + 1
        game_seed = rot_state["seed"]
    else:
        mode, table, position = setup_episode(episode, context)
    return mode, table, position, game_seed, rot_state


def sequential_stream(context: MainPhaseContext):
    # Seat rotation (deal-paired collection): groups of 5 consecutive
    # episodes share one sampled (mode, table, deal); the hero plays every
    # seat of the same deal against the same opponents — the train-time
    # duplicate instrument. The deal seed is drawn once per group so the
    # cards are identical across the 5 rotations.
    rot_state = {}
    teacher_kwargs = _teacher_kwargs(context)
    for episode in range(context.start_episode + 1, context.end_episode + 1):
        mode, table, position, game_seed, rot_state = rotation_plan(
            episode, context.start_episode, rot_state, context
        )
        opponents = [
            OpponentAdapter(context.training_agent, SELF_PLAY)
            if entry == SELF_PLAY
            else OpponentAdapter(entry.agent, entry.member_id)
            for entry in table
        ]
        game, events, scores, training_data_single, pos_to_seat = play_population_game(
            training_agent=context.training_agent,
            opponents=opponents,
            partner_mode=mode,
            training_agent_position=position,
            reward_mode="terminal",
            collect_oracle=context.collect_oracle,
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
            {pos: seat.metadata.agent_id for pos, seat in pos_to_seat.items()},
        )


def parallel_stream(context: MainPhaseContext, pool, num_workers):
    publish_weights(context)
    next_episode = context.start_episode + 1
    while next_episode <= context.end_episode:
        remaining_transitions = max(
            1, context.args.update_interval - context.tx_counter.count
        )
        window_size = max(
            num_workers, min(256, int(remaining_transitions / AVG_TX_PER_GAME) + 1)
        )
        window_end = min(context.end_episode, next_episode + window_size - 1)
        jobs = []
        for episode in range(next_episode, window_end + 1):
            mode, table, position, game_seed, context.rot_state = rotation_plan(
                episode, context.start_episode, context.rot_state, context
            )
            jobs.append(
                _Job(
                    episode=episode,
                    partner_mode=mode,
                    training_position=position,
                    opponent_ids=[
                        SELF_PLAY if entry == SELF_PLAY else entry.member_id
                        for entry in table
                    ],
                    weight_version=context.weight_sync["version"],
                    collect_oracle=context.collect_oracle,
                    game_seed=game_seed,
                )
            )
        for result in pool.imap(_league_worker_play, jobs):
            yield (
                result["episode"],
                result["partner_mode"],
                result["training_position"],
                result["episode_events"],
                result["final_scores"],
                result["training_data_single"],
                result["game_summary"],
                result["seat_to_member_id"],
            )
        next_episode = window_end + 1
