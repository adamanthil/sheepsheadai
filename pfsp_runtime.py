#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) training runtime for Sheepshead.

Shared population-based training machinery used by both PFSP entry points:
``train_pfsp_ppo.py`` (shaped baseline) and ``train_pfsp_exit.py`` (ISMCTS
hybrid). The strategy is selected by ``PFSPHyperparams.reward_mode`` /
``.search`` (see config.py): shaped uses reward shaping + entropy bumps;
terminal uses terminal-only return + ISMCTS search-distillation. The
hyperparameters live in config.py; pure helper functions in training_utils.py.
"""

import copy
import csv
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from openskill.models import PlackettLuce

from config import DEFAULT_HYPERPARAMS, PFSPHyperparams, SearchConfig
from ismcts import ISMCTSConfig, ISMCTSTeacher
from pfsp import (
    AgentMetadata,
    PFSPPopulation,
    PopulationAgent,
    apply_trick_profile_sample,
    compute_action_profile_events,
    compute_trick_profile_samples,
    create_initial_population_from_checkpoints,
    profile_pop_agent_action,
    profile_trick_completion,
)
from ppo import PPOAgent
from sheepshead import (
    ACTION_IDS,
    ACTIONS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    Game,
    get_partner_mode_name,
)
from training_utils import (
    analyze_strategic_decisions,
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    estimate_hand_strength_category,
    get_partner_selection_mode,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    save_training_plot,
    update_intermediate_rewards_for_action,
)

# ----------------------------------------------------------------------------
# Parallel self-play (Lever 1): worker-process game generation.
#
# The learner owns the authoritative training agent + population + optimizer and
# does the single gradient update; a pool of worker processes generates games
# with frozen, versioned weights and returns plain-data GameResults. Opponents are
# sampled in the learner (authoritative ratings/diversity); workers resolve them by
# agent_id from a local lazy cache of the on-disk population. Opponent strategic
# profiling is captured (not mutated) in the worker and replayed by the learner.
# These objects/functions are module-level so they pickle under the spawn start
# method. See ISMCTS_Overview_And_Roadmap.md §4 and Throughput_Profiling_Notes.md.
# ----------------------------------------------------------------------------


@dataclass
class JobSpec:
    """One game-generation job sent to a worker. ``opponent_specs`` are the sampled
    (agent_id, partner_mode) pairs; ``weight_version`` tags which weights to play
    with (the worker reloads from disk when this exceeds its local version)."""

    episode: int
    partner_mode: int
    training_position: int
    shaping_weights: dict
    opponent_specs: list
    weight_version: int


@dataclass
class GameResult:
    """Plain, picklable result of a worker-generated game. ``training_data_single``
    carries the search diagnostics and (under capture) the per-agent profile events;
    ``game_summary`` is the normalized public record (see make_game_summary)."""

    episode: int
    partner_mode: int
    training_position: int
    episode_events: list
    final_scores: list
    training_data_single: dict
    game_summary: dict
    seat_to_agent_id: dict = field(default_factory=dict)


# Per-worker global state (populated by _worker_init under the spawn start method).
_WORKER: dict = {}


def _worker_init(init_args: dict) -> None:
    """Pool initializer: build this worker's training agent, ISMCTS teacher, lazy
    population cache, and determinization RNG once. Single-thread the BLAS to avoid
    oversubscription across workers."""
    import torch as _torch

    _torch.set_num_threads(1)

    agent = PPOAgent(len(ACTIONS), activation=init_args["activation"])
    use_search = init_args["use_search"]
    teacher = None
    if use_search:
        agent.searched_pg_weight = init_args["searched_pg_weight"]
        teacher = ISMCTSTeacher(agent, init_args["ismcts_config"])

    seed = init_args["base_seed"] ^ (os.getpid() & 0xFFFFFFFF)
    random.seed(seed)  # in-worker opponent-seat shuffles in play_population_game

    _WORKER.clear()
    _WORKER.update(
        {
            "agent": agent,
            "teacher": teacher,
            "use_search": use_search,
            "reward_mode": init_args["reward_mode"],
            "search_config": init_args["search_config"],
            "activation": init_args["activation"],
            "population_dir": init_args["population_dir"],
            "weight_path_base": init_args["weight_path_base"],
            "version": 0,
            "cache": {},
            "det_rng": random.Random(seed ^ 0x9E3779B9),
        }
    )


def _worker_get_opponent(agent_id: str, partner_mode: int) -> PopulationAgent:
    """Resolve an opponent from the worker's lazy cache, loading its weights from the
    on-disk population on first use."""
    cache = _WORKER["cache"]
    opp = cache.get(agent_id)
    if opp is not None:
        return opp
    subdir = "jd_agents" if partner_mode == PARTNER_BY_JD else "called_ace_agents"
    path = os.path.join(_WORKER["population_dir"], subdir, f"{agent_id}.pt")
    agent = PPOAgent(len(ACTIONS), activation=_WORKER["activation"])
    agent.load(path, load_optimizers=False)
    meta = AgentMetadata(
        agent_id=agent_id,
        creation_time=0.0,
        parent_id=None,
        training_episodes=0,
        partner_mode=partner_mode,
        activation=_WORKER["activation"],
    )
    opp = PopulationAgent(agent, meta)
    cache[agent_id] = opp
    return opp


def _worker_play_game(job: JobSpec) -> GameResult:
    """Play one game in the worker with frozen (versioned) weights, capturing opponent
    profile events for the learner to replay."""
    import torch as _torch

    g = _WORKER
    if job.weight_version > g["version"]:
        weight_path = f"{g['weight_path_base']}_v{job.weight_version}.pt"
        ckpt = _torch.load(weight_path, map_location="cpu")
        agent = g["agent"]
        agent.encoder.load_state_dict(ckpt["encoder_state_dict"])
        agent.actor.load_state_dict(ckpt["actor_state_dict"])
        agent.critic.load_state_dict(ckpt["critic_state_dict"], strict=False)
        agent._player_memories = {}
        g["version"] = job.weight_version

    opponents = [_worker_get_opponent(aid, pm) for (aid, pm) in job.opponent_specs]

    game, episode_events, final_scores, training_data_single, pos_to_pop_agent = (
        play_population_game(
            training_agent=g["agent"],
            opponents=opponents,
            partner_mode=job.partner_mode,
            training_agent_position=job.training_position,
            shaping_weights=job.shaping_weights,
            reward_mode=g["reward_mode"],
            teacher=g["teacher"],
            determinization_rng=g["det_rng"],
            search_config=g["search_config"],
            capture_profile_events=True,
        )
    )

    seat_to_agent_id = {
        seat: opp.metadata.agent_id for seat, opp in pos_to_pop_agent.items()
    }
    return GameResult(
        episode=job.episode,
        partner_mode=job.partner_mode,
        training_position=job.training_position,
        episode_events=episode_events,
        final_scores=final_scores,
        training_data_single=training_data_single,
        game_summary=make_game_summary(game),
        seat_to_agent_id=seat_to_agent_id,
    )


def _is_private_decision(valid_actions) -> bool:
    """True when the decision is a private bury/under (excluded from the public
    record fed to the ISMCTS teacher's forced replay)."""
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid_actions
    )


def _search_head(valid_actions) -> str:
    """Classify a decision into a search head (mirrors ISMCTSTeacher._infer_head)."""
    names = [ACTIONS[a - 1] for a in valid_actions]
    if any(n in ("PICK", "PASS") for n in names):
        return "pick"
    if any(n == "ALONE" or n == "JD PARTNER" or n.startswith("CALL ") for n in names):
        return "partner"
    if any(n.startswith("BURY ") or n.startswith("UNDER ") for n in names):
        return "bury"
    return "play"


def interpolated_weight(schedule: dict, progress_pct: float) -> float:
    """Linear interpolation of schedule weights by percent progress.

    schedule: mapping of percent (0-100) to weight.
    progress_pct: current percent progress in [0, 100].
    """
    if not schedule:
        return 1.0
    # Normalize and sort points by percent
    points = sorted((float(k), float(v)) for k, v in schedule.items())
    # Clamp to endpoints
    if progress_pct <= points[0][0]:
        return points[0][1]
    if progress_pct >= points[-1][0]:
        return points[-1][1]
    # Find segment and interpolate
    for (k0, v0), (k1, v1) in zip(points, points[1:]):
        if k0 <= progress_pct <= k1:
            if k1 == k0:
                return v1
            t = (progress_pct - k0) / (k1 - k0)
            return v0 + t * (v1 - v0)
    # Fallback (should not hit due to clamps)
    return points[-1][1]


def play_population_game(
    training_agent: PPOAgent,
    opponents: list,
    partner_mode: int,
    training_agent_position: int = 1,
    shaping_weights: dict | None = None,
    reward_mode: str = "shaped",
    teacher: "ISMCTSTeacher | None" = None,
    determinization_rng: "random.Random | None" = None,
    search_config: "SearchConfig | None" = None,
    capture_profile_events: bool = False,
) -> tuple:
    """Play a single game with the training agent and population opponents.

    ``reward_mode`` selects the return: ``"shaped"`` applies the intermediate
    reward shaping + per-trick rewards and ``process_episode_rewards``;
    ``"terminal"`` skips all shaping and uses ``process_terminal_rewards``
    (final_score-only), optionally attaching ISMCTS soft-teacher targets to a
    per-head fraction of the training agent's decisions (search is teacher-only;
    the agent still acts on-policy).

    ``capture_profile_events``: parallel self-play workers cannot mutate the
    learner's authoritative population, so when set the opponent strategic-profile
    updates are *captured* (keyed by agent_id) into
    ``training_agent_data["profile_events"]`` instead of mutating the opponent
    objects in-place; the learner replays them. When False (sequential path) the
    opponent profiles are mutated in-place exactly as before.

    Returns:
        tuple: (game, episode_events, final_scores, training_agent_data, opponents_by_position)
    """
    game = Game(partner_selection_mode=partner_mode)
    weights = shaping_weights or {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 1.0}
    shaped = reward_mode == "shaped"
    search_enabled = (
        reward_mode == "terminal"
        and teacher is not None
        and determinization_rng is not None
        and search_config is not None
        and search_config.enabled
    )
    # Public (seat, action_id) record for the teacher's forced replay (search only).
    forced_public: list[tuple[int, int]] = []
    # Per-game ISMCTS search diagnostics (terminal/ExIt mode), aggregated by head:
    # how many decisions were searched, how many cleared the ESS gate (accepted),
    # and the summed ESS / pi' entropy for averaging. Attached to training_agent_data
    # so the driver can window + log them (ESS-abort fraction, target sharpness).
    search_diagnostics = {
        head: {"count": 0, "accepted": 0, "ess_sum": 0.0, "entropy_sum": 0.0}
        for head in ("pick", "partner", "bury", "play")
    }
    # Captured opponent strategic-profile events (parallel/worker mode only),
    # keyed by agent_id, for the learner to replay onto the authoritative
    # population. action_events: list of update_strategic_profile_from_game dicts;
    # trick_samples: list of (role, trick_win, is_leader, lead_win) tuples.
    captured_action_events: dict[str, list[dict]] = {}
    captured_trick_samples: dict[str, list[tuple]] = {}

    # Reset recurrent states for all agents
    training_agent.reset_recurrent_state()
    for opponent in opponents:
        opponent.agent.reset_recurrent_state()

    # Create position-to-agent mapping; all five seats must be populated by training + 4 opponents
    agents = [None] * 5
    agents[training_agent_position - 1] = training_agent

    # Randomize which opponent sits in which non-training seat to reduce seat-assignment bias
    opponent_seat_positions = [
        pos for pos in range(1, 6) if pos != training_agent_position
    ]
    random.shuffle(opponent_seat_positions)
    for opponent, seat_pos in zip(opponents[:4], opponent_seat_positions):
        agents[seat_pos - 1] = opponent.agent

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

    # Map positions to population opponents for profile updates
    pos_to_pop_agent = {}
    opp_positions = opponent_seat_positions.copy()
    for opp, seat_pos in zip(opponents[: len(opp_positions)], opp_positions):
        pos_to_pop_agent[seat_pos] = opp

    # Hand strength categories captured once at start
    hand_strength_by_pos = {
        p.position: estimate_hand_strength_category(p.hand) for p in game.players
    }

    while not game.is_done():
        for player in game.players:
            current_agent = agents[player.position - 1]
            valid_actions = player.get_valid_action_ids()

            while valid_actions:
                state = player.get_state_dict()
                is_private = _is_private_decision(valid_actions)

                # Get action from appropriate agent
                if current_agent == training_agent:
                    action, log_prob, value = current_agent.act(
                        state, valid_actions, player.position
                    )

                    # Store transition for training agent
                    transition = {
                        "kind": "action",
                        "player": player,
                        "state": state,
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "valid_actions": valid_actions.copy(),
                        "intermediate_reward": 0.0,
                        "secret_partner_label": 1.0
                        if player.is_secret_partner
                        else 0.0,
                        "points_label": compute_known_points_rel(player),
                        "seen_trump_mask_label": compute_seen_trump_mask(player),
                        "unseen_trump_higher_than_hand_label": compute_any_unseen_trump_higher_than_hand(
                            player
                        ),
                        "search_target": None,
                        "has_search_target": False,
                    }
                    episode_transitions.append(transition)

                    if shaped:
                        # Shared intermediate reward shaping and trick tracking
                        update_intermediate_rewards_for_action(
                            game,
                            player,
                            action,
                            transition,
                            current_trick_transitions,
                            pick_weight=weights["pick"],
                            partner_weight=weights["partner"],
                            bury_weight=weights["bury"],
                            play_weight=weights["play"],
                        )
                    elif search_enabled:
                        # ISMCTS soft-teacher target on a per-head fraction of
                        # decisions (teacher-only; agent acted on-policy above;
                        # search() is memory-neutral — snapshots/restores). Leaster
                        # PLAY decisions ARE searched: with the per-trick reward +
                        # leaster bonus gone, the pass->leaster branch the bidding
                        # EV rides on is only well-valued if the agent plays
                        # leasters well, which needs a teacher signal there.
                        # sample_determinization handles the no-picker leaster
                        # state (Game._sample_leaster_deal).
                        head = _search_head(valid_actions)
                        head_fraction = search_config.head_search_fractions.get(
                            head, 0.0
                        )
                        if (
                            head_fraction > 0.0
                            and determinization_rng.random() < head_fraction
                        ):
                            current_trick = game.current_trick
                            # Trick-indexed rollout depth: roll (near) to terminal
                            # in the early tricks where the critic is blind, then
                            # bootstrap d_short plies later (validated by the t_full
                            # probe: a search at trick t bootstraps at ~t+d_short,
                            # so t_full=1 + d_short=2 lands every bootstrap at
                            # trick >= 4 where R^2 >= 0.73). Leasters ALWAYS roll to
                            # terminal: the critic never calibrates on leaster
                            # outcomes (R^2 <= 0.21 even at trick 5), so a bootstrap
                            # there is noise.
                            if game.is_leaster or current_trick <= search_config.t_full:
                                rollout_depth = 6 - current_trick
                            else:
                                rollout_depth = search_config.d_short
                            res = teacher.search(
                                game,
                                player.position,
                                list(forced_public),
                                determinization_rng,
                                d_rollout=rollout_depth,
                            )
                            target_accepted = res["ok"] and float(res["pi"].sum()) > 0.0
                            if target_accepted:
                                transition["search_target"] = res["pi"].tolist()
                                transition["has_search_target"] = True
                            # Diagnostics: ESS-abort fraction and pi' sharpness.
                            head_diag = search_diagnostics[head]
                            head_diag["count"] += 1
                            head_diag["ess_sum"] += float(res["ess"])
                            if target_accepted:
                                head_diag["accepted"] += 1
                                pi = res["pi"]
                                nonzero = pi[pi > 0]
                                head_diag["entropy_sum"] += float(
                                    -(nonzero * np.log(nonzero)).sum()
                                )

                else:
                    # Opponent action (stochastic for diversity)
                    action, _, _ = current_agent.act(
                        state, valid_actions, player.position, deterministic=False
                    )

                # Record this seat's public action for the teacher's forced replay.
                if search_enabled and not is_private:
                    forced_public.append((player.position, action))

                # --- Strategic profile updates for opponents (pre-action; uses pre-action hand + trick state) ---
                pop_agent = pos_to_pop_agent.get(player.position)
                if pop_agent:
                    if capture_profile_events:
                        events = compute_action_profile_events(
                            game, player, action, hand_strength_by_pos
                        )
                        if events:
                            captured_action_events.setdefault(
                                pop_agent.metadata.agent_id, []
                            ).extend(events)
                    else:
                        profile_pop_agent_action(
                            game, player, action, pop_agent, hand_strength_by_pos
                        )

                player.act(action)

                # Handle trick completion; PFSP-specific observation propagation
                trick_completed = handle_trick_completion(
                    game, current_trick_transitions
                )
                if trick_completed and not game.is_done():
                    # Emit observations for the completed trick using dedicated accessor
                    for seat in game.players:
                        seat_agent = agents[seat.position - 1]
                        if seat_agent == training_agent:
                            # Update training agent's recurrent hidden state and also store for unroll
                            training_agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
                            episode_transitions.append(
                                {
                                    "kind": "observation",
                                    "player": seat,
                                    "state": seat.get_last_trick_state_dict(),
                                }
                            )
                        else:
                            seat_agent.observe(
                                seat.get_last_trick_state_dict(), seat.position
                            )
                    # Update trick-level EWMAs for population agents
                    pos_to_pop_agent_local = {}
                    for i, opp in enumerate(opponents[: len(opp_positions)]):
                        pos_to_pop_agent_local[opp_positions[i]] = opp
                    if capture_profile_events:
                        for (
                            seat_pos,
                            role,
                            trick_win,
                            is_leader,
                            lead_win,
                        ) in compute_trick_profile_samples(game):
                            sample_agent = pos_to_pop_agent_local.get(seat_pos)
                            if sample_agent:
                                captured_trick_samples.setdefault(
                                    sample_agent.metadata.agent_id, []
                                ).append((role, trick_win, is_leader, lead_win))
                    else:
                        profile_trick_completion(game, pos_to_pop_agent_local)

                valid_actions = player.get_valid_action_ids()

    final_scores = [player.get_score() for player in game.players]

    # Return training agent specific data
    training_agent_score = final_scores[training_agent_position - 1]
    was_picker = game.picker == training_agent_position

    training_agent_data = {
        "score": training_agent_score,
        "was_picker": was_picker,
        "position": training_agent_position,
        "search_diagnostics": search_diagnostics,
    }
    if capture_profile_events:
        training_agent_data["profile_events"] = {
            "action": captured_action_events,
            "trick": captured_trick_samples,
        }

    # Compute rewards for training agent actions. Shaped: intermediate + final
    # (+ leaster bonus). Terminal: final_score-only on the last action, no shaping
    # and no leaster bonus (get_score scores leasters correctly).
    reward_fn = process_episode_rewards if shaped else process_terminal_rewards
    reward_map = {}
    for reward_data in reward_fn(
        [t for t in episode_transitions if t["kind"] == "action"],
        final_scores,
        game.is_leaster,
    ):
        reward_map[id(reward_data["transition"])] = reward_data["reward"]

    # Build final episode event stream for storage
    episode_events = []
    for ev in episode_transitions:
        if ev["kind"] == "observation":
            episode_events.append(
                {
                    "kind": "observation",
                    "state": ev["state"],
                    "player_id": ev["player"].position,
                }
            )
        else:
            seat_pos = ev["player"].position
            episode_events.append(
                {
                    "kind": "action",
                    "state": ev["state"],
                    "action": ev["action"],
                    "log_prob": ev["log_prob"],
                    "value": ev["value"],
                    "valid_actions": ev["valid_actions"],
                    "reward": reward_map[id(ev)],
                    "player_id": seat_pos,
                    "win_label": 1.0 if final_scores[seat_pos - 1] > 0 else 0.0,
                    "final_return_label": float(final_scores[seat_pos - 1]),
                    "secret_partner_label": ev.get("secret_partner_label", 0.0),
                    "points_label": ev.get("points_label", None),
                    "seen_trump_mask_label": ev.get("seen_trump_mask_label", None),
                    "unseen_trump_higher_than_hand_label": ev.get(
                        "unseen_trump_higher_than_hand_label", None
                    ),
                    "search_target": ev.get("search_target"),
                    "has_search_target": ev.get("has_search_target", False),
                }
            )

    return (
        game,
        episode_events,
        final_scores,
        training_agent_data,
        dict(pos_to_pop_agent),
    )


def make_game_summary(game) -> dict:
    """Normalize the post-game public fields the training driver needs into a plain,
    picklable dict, so the per-episode bookkeeping path is identical whether the game
    was played in-process (sequential) or in a worker (parallel).

    seat_roles maps every seat (1-5) to picker/partner/defender/leaster using the same
    logic as the original role-perf loop.
    """
    is_leaster = bool(game.is_leaster)
    is_partner_seat = getattr(game, "is_partner_seat", None)
    seat_roles: dict[int, str] = {}
    for pos in range(1, 6):
        if is_leaster:
            seat_roles[pos] = "leaster"
        elif game.picker == pos:
            seat_roles[pos] = "picker"
        elif (is_partner_seat(pos) if callable(is_partner_seat) else False) or getattr(
            game.players[pos - 1], "is_partner", False
        ):
            seat_roles[pos] = "partner"
        else:
            seat_roles[pos] = "defender"

    if game.picker and not is_leaster:
        final_picker_points = game.get_final_picker_points()
        final_defender_points = game.get_final_defender_points()
    else:
        final_picker_points = None
        final_defender_points = None

    return {
        "picker": game.picker,
        "partner": game.partner,
        "is_leaster": is_leaster,
        "alone_called": bool(game.alone_called),
        "is_called_under": bool(game.is_called_under),
        "called_card": game.called_card,
        "seat_roles": seat_roles,
        "final_picker_points": final_picker_points,
        "final_defender_points": final_defender_points,
    }


def run_pfsp_training(
    num_episodes: int = 500000,
    update_interval: int = 2048,
    save_interval: int = 5000,
    strategic_eval_interval: int = 10000,
    population_add_interval: int = 5000,
    cross_eval_interval: int = 20000,
    resume_model: str = None,
    activation: str = "swish",
    initial_checkpoints: list = None,
    schedule_horizon_episodes: int | None = None,
    hyperparams: PFSPHyperparams = DEFAULT_HYPERPARAMS,
    run_name: str = "pfsp_run",
    population_dir: str | None = None,
):
    """
    PFSP training with population-based opponents.
    """
    schedule_horizon = (
        num_episodes if schedule_horizon_episodes is None else schedule_horizon_episodes
    )
    if schedule_horizon <= 0:
        raise ValueError("schedule_horizon_episodes must be > 0")

    def get_schedule_progress_pct(episode: int) -> float:
        clamped_episode = min(episode, schedule_horizon)
        return min(100.0, max(0.0, (clamped_episode / schedule_horizon) * 100.0))

    print("🚀 Starting PFSP (Population-Based) Training...")
    print("=" * 80)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Population add interval: {population_add_interval}")
    print(f"  Cross-evaluation interval: {cross_eval_interval}")
    print(f"  Schedule horizon episodes: {schedule_horizon:,}")
    print(f"  Activation function: {activation.upper()}")
    print("  Opponent strategy: POPULATION-BASED (PFSP)")
    print("  Population management: OpenSkill ratings + diversity")
    print("=" * 80)

    # Create training agent with initial LRs from schedule start (0% progress)
    initial_actor_lr = interpolated_weight(hyperparams.lr_schedule_actor, 0.0)
    initial_critic_lr = interpolated_weight(hyperparams.lr_schedule_critic, 0.0)
    training_agent = PPOAgent(
        len(ACTIONS),
        lr_actor=initial_actor_lr,
        lr_critic=initial_critic_lr,
        activation=activation,
    )

    # ISMCTS soft-teacher (terminal/ExIt mode only). Training-time only; the
    # agent acts on-policy and population opponents never search.
    # determinization_rng drives determinization sampling.
    use_search = (
        hyperparams.reward_mode == "terminal" and hyperparams.search is not None
    )
    # Single ISMCTSConfig shared by the in-process teacher and the parallel workers.
    ismcts_config = ISMCTSConfig()
    ismcts_teacher = (
        ISMCTSTeacher(training_agent, ismcts_config) if use_search else None
    )
    determinization_rng = random.Random(20260529) if use_search else None
    if use_search:
        # PG-mask vs additive-form A/B for searched transitions (plan §4).
        training_agent.searched_pg_weight = hyperparams.search.searched_pg_weight

    # Parallel game-generation worker count (Lever 1). None => auto: parallelize the
    # expensive ISMCTS ExIt generation (terminal mode), keep the cheap shaped PPO
    # baseline sequential. <=1 runs the original in-process sequential loop.
    num_workers_cfg = getattr(hyperparams, "num_workers", None)
    if num_workers_cfg is None:
        num_workers = (
            min((os.cpu_count() or 1) - 1, 8)
            if hyperparams.reward_mode == "terminal"
            else 1
        )
        num_workers = max(1, num_workers)
    else:
        num_workers = max(1, int(num_workers_cfg))
    print(
        f"  Game generation: {'PARALLEL (' + str(num_workers) + ' workers)' if num_workers > 1 else 'SEQUENTIAL'}"
    )

    # OpenSkill rating for the training agent
    rating_model = PlackettLuce()
    training_rating = rating_model.rating()

    # All generated artifacts (checkpoints, final model, plots, CSVs, and the
    # population) live under runs/<run_name>/ so nothing collides with committed
    # or frozen files at the repo root. population_dir can be overridden (e.g. to
    # point at a seeded pool); defaults under the run dir.
    output_dir = os.path.join("runs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    if population_dir is None:
        population_dir = os.path.join(output_dir, "population")

    # Create population
    population = PFSPPopulation(
        max_population_jd=75,
        max_population_called_ace=75,
        population_dir=population_dir,
    )

    # Initialize population from checkpoints if provided
    if initial_checkpoints:
        create_initial_population_from_checkpoints(
            population,
            initial_checkpoints,
            activation=activation,
            max_agents_per_mode=10,
        )

    # Load or create initial training agent
    start_episode = 0
    if resume_model:
        try:
            training_agent.load(resume_model, load_optimizers=True)
            print(f"✅ Loaded training agent from {resume_model}")
            if "checkpoint_" in resume_model:
                start_episode = int(resume_model.split("_")[-1].split(".")[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training agent")

    # Initialize tracking variables
    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    leaster_window = deque(maxlen=3000)
    alone_call_window = deque(maxlen=3000)
    training_alone_window = deque(maxlen=3000)
    called_ace_window = deque(maxlen=3000)
    called_under_window = deque(maxlen=3000)
    called_10_window = deque(maxlen=3000)
    team_point_differences = deque(maxlen=3000)
    picker_window = deque(maxlen=3000)

    training_data = {
        "episodes": [],
        "picker_avg": [],
        "called_pick_rate": [],
        "jd_pick_rate": [],
        "learning_rate": [],
        "time_elapsed": [],
        "pick_hand_correlation": [],
        "picker_trump_rate": [],
        "defender_trump_rate": [],
        "bury_quality_rate": [],
        "team_point_diff": [],
        "alone_rate": [],
        "leaster_rate": [],
        "strategic_episodes": [],
        "population_stats": [],
    }

    # Checkpoints / CSVs / plots all live under the run dir.
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # CSV log files for ongoing progress/metrics
    progress_csv = os.path.join(checkpoint_dir, "pfsp_training_progress.csv")
    strategic_csv = os.path.join(checkpoint_dir, "pfsp_strategic_metrics.csv")

    # If CSVs exist, preload them so training_data is continuous across resumes
    start_time_offset = 0.0
    if os.path.exists(progress_csv):
        with open(progress_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data["episodes"].append(int(row["episode"]))
                training_data["picker_avg"].append(float(row["picker_avg"]))
                training_data["called_pick_rate"].append(float(row["called_pick_rate"]))
                training_data["jd_pick_rate"].append(float(row["jd_pick_rate"]))
                training_data["learning_rate"].append(float(row["learning_rate"]))
                training_data["time_elapsed"].append(float(row["time_elapsed"]))
                training_data["team_point_diff"].append(float(row["team_point_diff"]))
                # Optional historical series (may be missing in early CSVs)
                if "alone_rate" in row and row["alone_rate"] != "":
                    training_data["alone_rate"].append(float(row["alone_rate"]))
                if "leaster_rate" in row and row["leaster_rate"] != "":
                    training_data["leaster_rate"].append(float(row["leaster_rate"]))
        if training_data["time_elapsed"]:
            start_time_offset = training_data["time_elapsed"][-1]

    if os.path.exists(strategic_csv):
        with open(strategic_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data["strategic_episodes"].append(int(row["episode"]))
                training_data["pick_hand_correlation"].append(
                    float(row["pick_hand_correlation"])
                )
                training_data["picker_trump_rate"].append(
                    float(row["picker_trump_rate"])
                )
                training_data["defender_trump_rate"].append(
                    float(row["defender_trump_rate"])
                )
                training_data["bury_quality_rate"].append(
                    float(row["bury_quality_rate"])
                )

    start_time = time.time()
    game_count = 0
    # Track cumulative μ renormalisation shifts for logging
    cumulative_renorm = 0.0
    transitions_since_update = 0
    last_checkpoint_time = start_time
    # ISMCTS search diagnostics accumulated over the games in an update window
    # (terminal/ExIt mode); logged + reset at each update.
    search_diagnostics_window = {
        head: {"count": 0, "accepted": 0, "ess_sum": 0.0, "entropy_sum": 0.0}
        for head in ("pick", "partner", "bury", "play")
    }

    bury_entropy_bump_until = 0
    partner_entropy_bump_until = 0
    pick_entropy_bump_until = 0

    # Epsilon-floor exploration controllers (shaped mode only; the adaptive
    # blocks below are gated by reward_mode == "shaped"). Init is harmless in
    # terminal mode — it sets epsilon to base (0.0), a no-op for the teacher.
    # Partner CALL mixture epsilon controller state
    current_partner_call_eps = hyperparams.partner_call_eps_base
    training_agent.set_partner_call_epsilon(current_partner_call_eps)

    # PASS-floor epsilon controller state
    current_pass_floor_eps = hyperparams.pass_floor_eps_base
    training_agent.set_pass_floor_epsilon(current_pass_floor_eps)

    # PICK-floor epsilon controller state
    current_pick_floor_eps = hyperparams.pick_floor_eps_base
    training_agent.set_pick_floor_epsilon(current_pick_floor_eps)

    print(f"\n🎮 Beginning PFSP training... (target: {num_episodes:,} episodes)")
    print(population.get_population_summary())
    print("-" * 80)

    # Opponent sampling schedule state: occasional "anchor blocks" where most seats are anchors
    anchor_block_remaining = 0

    def _setup_episode(episode: int):
        """Per-episode setup shared by the sequential and parallel drivers: μ-renorm,
        partner mode, opponent scheduling + sampling, training seat, shaping weights.
        Mutates the renorm/anchor schedule state, preserving the sequential RNG order."""
        nonlocal cumulative_renorm, anchor_block_remaining
        # --------------------------------------------------------------
        # Prevent numerical overflow in OpenSkill ratings by renormalising μ
        # --------------------------------------------------------------
        MAX_ABS_MU = 350.0
        all_ratings = (
            [training_rating]
            + [ag.rating for ag in population.jd_population]
            + [ag.rating for ag in population.called_ace_population]
        )
        extreme_mu = max(abs(r.mu) for r in all_ratings) if all_ratings else 0
        if extreme_mu > MAX_ABS_MU:
            shift = extreme_mu - MAX_ABS_MU
            for r in all_ratings:
                r.mu -= np.sign(r.mu) * shift
            cumulative_renorm -= shift

        # Throttle log noise – print cumulative shift once every 2 000 episodes
        if episode % 2000 == 1:
            print(
                f"⚖️  Cumulative rating μ renorm Δ over last 2k eps: {cumulative_renorm:+.1f}  (|μ|max={extreme_mu:.1f})"
            )
            cumulative_renorm = 0.0

        partner_mode = get_partner_selection_mode(episode)

        # ---------------- Opponent scheduling ----------------
        # Default: PFSP mixture most of the time. Occasionally run an "anchor block"
        # where most seats are anchors to reduce forgetting without consuming throughput constantly.
        anchor_slots = 0
        pressure_slots = 0
        support_slots = 0

        if anchor_block_remaining > 0:
            anchor_slots = hyperparams.anchor_slots_in_block
            anchor_block_remaining -= 1
        else:
            if random.random() < hyperparams.anchor_block_start_prob:
                anchor_block_remaining = (
                    random.randint(
                        hyperparams.anchor_block_len_min,
                        hyperparams.anchor_block_len_max,
                    )
                    - 1
                )
                anchor_slots = hyperparams.anchor_slots_in_block
            else:
                # Low-prob inclusions during normal PFSP episodes
                if random.random() < hyperparams.pressure_slot_prob:
                    pressure_slots = 1
                if random.random() < hyperparams.support_slot_prob:
                    support_slots = 1

        opponents = population.sample_opponents(
            partner_mode=partner_mode,
            n_opponents=4,
            variable_weight=0.7,
            hard_weight=0.3,
            hard_power=2.0,
            diversity_weight=0.2,
            uniform_mix=0.1,
            anchor_slots=anchor_slots,
            pressure_slots=pressure_slots,
            support_slots=support_slots,
        )

        # Require 4 opponents in population; exit script if not available
        if len(opponents) < 4:
            print(
                f"❌ Insufficient PFSP opponents for {get_partner_mode_name(partner_mode)} (need 4, got {len(opponents)}). Provide --initial-checkpoints or pre-populate pfsp_population."
            )
            sys.exit(1)

        # Randomly select training agent position (1-5)
        training_position = random.randint(1, 5)

        # Compute per-episode shaping weights from schedules (shaped mode only;
        # ignored by play_population_game when reward_mode="terminal").
        progress_pct = get_schedule_progress_pct(episode)
        shaping_weights = {
            "pick": interpolated_weight(
                hyperparams.shaping_schedule_pick, progress_pct
            ),
            "partner": interpolated_weight(
                hyperparams.shaping_schedule_partner, progress_pct
            ),
            "bury": interpolated_weight(
                hyperparams.shaping_schedule_bury, progress_pct
            ),
            "play": interpolated_weight(
                hyperparams.shaping_schedule_play, progress_pct
            ),
        }
        return partner_mode, opponents, training_position, shaping_weights

    def _sequential_episode_stream():
        """Generate one game at a time in-process (num_workers <= 1). Behavior-
        identical to the original inline loop."""
        for episode in range(start_episode + 1, num_episodes + 1):
            partner_mode, opponents, training_position, shaping_weights = (
                _setup_episode(episode)
            )
            (
                game,
                episode_events,
                final_scores,
                training_data_single,
                position_to_agent,
            ) = play_population_game(
                training_agent=training_agent,
                opponents=opponents,
                partner_mode=partner_mode,
                training_agent_position=training_position,
                shaping_weights=shaping_weights,
                reward_mode=hyperparams.reward_mode,
                teacher=ismcts_teacher,
                determinization_rng=determinization_rng,
                search_config=hyperparams.search,
            )
            yield (
                episode,
                partner_mode,
                training_position,
                episode_events,
                final_scores,
                training_data_single,
                make_game_summary(game),
                position_to_agent,
            )

    # Weight-sync coordination for parallel workers: the update block publishes the
    # training agent's fresh weights (versioned) here; the parallel stream tags each
    # dispatched job with the current version and workers reload on a bump. Defined
    # even in sequential mode (a harmless no-op there).
    weight_sync = {"version": 0, "base": os.path.join(output_dir, "_worker_weights")}

    def _publish_weights():
        """Write the training agent's current weights to a fresh, immutable per-version
        file for workers and bump the version. Per-version files avoid any
        read-during-replace race (a worker always loads the exact tagged version).
        Called by the update block after a successful update (parallel only)."""
        weight_sync["version"] += 1
        version = weight_sync["version"]
        path = f"{weight_sync['base']}_v{version}.pt"
        tmp_path = path + ".tmp"
        torch.save(
            {
                "encoder_state_dict": training_agent.encoder.state_dict(),
                "actor_state_dict": training_agent.actor.state_dict(),
                "critic_state_dict": training_agent.critic.state_dict(),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
        # Keep current + previous version; drop older to bound disk use.
        stale = f"{weight_sync['base']}_v{version - 2}.pt"
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass

    def _parallel_episode_stream(setup_episode, weight_sync, publish_weights):
        """Windowed synchronous self-play across a worker pool (Lever 1).

        Each window of episodes is sampled in the learner (authoritative population),
        dispatched to workers with the current frozen weight version, and yielded in
        episode order. Windows are sized *under* one update interval so the update
        always fires at a window boundary on games generated with a single weight
        version (strictly on-policy; the only lag is in opponent sampling, which is
        sampled up to a window ahead). After each update the learner publishes fresh
        weights and the next window picks up the new version.
        """
        import multiprocessing as mp

        init_args = {
            "activation": activation,
            "use_search": use_search,
            "searched_pg_weight": (
                hyperparams.search.searched_pg_weight if use_search else 0.0
            ),
            "reward_mode": hyperparams.reward_mode,
            "search_config": hyperparams.search,
            "ismcts_config": ismcts_config,
            "population_dir": population_dir,
            "weight_path_base": weight_sync["base"],
            "base_seed": 20260529,
        }
        # Publish v1 so the first jobs load the current training-agent weights.
        publish_weights()

        # Conservative transitions-per-game estimate used to size windows to the
        # remaining budget before the next update. Overestimating keeps windows from
        # overshooting the update threshold, which bounds any off-policy straddle to a
        # small, batch-lag-tolerable amount (the user-accepted approximation): every
        # game in a window uses one weight version; the update fires near the window
        # boundary and the next window picks up fresh weights.
        avg_tx_per_game = 12.0
        max_window = 8 * num_workers

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(init_args,),
        )
        try:
            episode = start_episode + 1
            while episode <= num_episodes:
                # Size this window to ~the transitions remaining before the next update
                # (reads the learner's live counter), floored for worker utilization.
                remaining_tx = max(1, update_interval - transitions_since_update)
                window_games = max(
                    num_workers,
                    min(max_window, int(remaining_tx / avg_tx_per_game) + 1),
                )
                end = min(num_episodes, episode + window_games - 1)
                version = weight_sync["version"]
                jobs = []
                episode_opponents = {}
                for ep in range(episode, end + 1):
                    partner_mode, opponents, training_position, _shaping = (
                        setup_episode(ep)
                    )
                    episode_opponents[ep] = {o.metadata.agent_id: o for o in opponents}
                    jobs.append(
                        JobSpec(
                            episode=ep,
                            partner_mode=partner_mode,
                            training_position=training_position,
                            shaping_weights=_shaping,
                            opponent_specs=[
                                (o.metadata.agent_id, o.metadata.partner_mode)
                                for o in opponents
                            ],
                            weight_version=version,
                        )
                    )
                for result in pool.imap(_worker_play_game, jobs):
                    id_to_auth = episode_opponents.pop(result.episode, {})
                    # Replay captured opponent profile events onto the authoritative
                    # population (batch-lagged: applied at result time, not in-game).
                    profile_events = result.training_data_single.pop(
                        "profile_events", {}
                    )
                    for aid, events in profile_events.get("action", {}).items():
                        auth = id_to_auth.get(aid)
                        if auth is not None:
                            for ev in events:
                                auth.update_strategic_profile_from_game(ev)
                    for aid, samples in profile_events.get("trick", {}).items():
                        auth = id_to_auth.get(aid)
                        if auth is not None:
                            for role, trick_win, is_leader, lead_win in samples:
                                apply_trick_profile_sample(
                                    auth, role, trick_win, is_leader, lead_win
                                )
                    position_to_agent = {
                        seat: id_to_auth.get(aid)
                        for seat, aid in result.seat_to_agent_id.items()
                    }
                    yield (
                        result.episode,
                        result.partner_mode,
                        result.training_position,
                        result.episode_events,
                        result.final_scores,
                        result.training_data_single,
                        result.game_summary,
                        position_to_agent,
                    )
                episode = end + 1
        finally:
            pool.close()
            pool.join()

    if num_workers > 1:
        episode_stream = _parallel_episode_stream(
            setup_episode=_setup_episode,
            weight_sync=weight_sync,
            publish_weights=_publish_weights,
        )
    else:
        episode_stream = _sequential_episode_stream()

    for (
        episode,
        partner_mode,
        training_position,
        episode_events,
        final_scores,
        training_data_single,
        summary,
        position_to_agent,
    ) in episode_stream:
        training_agent.store_episode_events(episode_events)
        transitions_since_update += sum(
            1 for ev in episode_events if ev["kind"] == "action"
        )

        # Accumulate ISMCTS search diagnostics for this update window.
        game_search_diagnostics = training_data_single.get("search_diagnostics")
        if game_search_diagnostics:
            for head, head_diag in game_search_diagnostics.items():
                window_stats = search_diagnostics_window[head]
                window_stats["count"] += head_diag["count"]
                window_stats["accepted"] += head_diag["accepted"]
                window_stats["ess_sum"] += head_diag["ess_sum"]
                window_stats["entropy_sum"] += head_diag["entropy_sum"]

        # Update statistics
        picker_score = (
            training_data_single["score"] if training_data_single["was_picker"] else 0
        )
        if training_data_single["was_picker"]:
            picker_scores.append(picker_score)

        # Track how often the training agent is the picker (unconditional)
        picker_window.append(1 if training_data_single["was_picker"] else 0)

        # ---------- OpenSkill rating update (centralized in PFSPPopulation) ----------
        opponent_positions = [pos for pos in range(1, 6) if pos != training_position]

        training_rating = population.update_ratings_with_training(
            training_rating=training_rating,
            final_scores=final_scores,
            training_position=training_position,
            opponents_by_position=position_to_agent,
            picker_seat=summary["picker"],
            partner_seat=summary["partner"],
            is_leaster=summary["is_leaster"],
        )

        # Track team point differences
        if summary["picker"] and not summary["is_leaster"]:
            picker_team_points = summary["final_picker_points"]
            defender_team_points = summary["final_defender_points"]
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0
        team_point_differences.append(team_point_diff)

        # Final performance profiling per role for diversity/clustering
        for pos in opponent_positions:
            opp_agent = position_to_agent.get(pos)
            if opp_agent:
                opp_agent.update_strategic_profile_from_game(
                    {
                        "final_score": final_scores[pos - 1],
                        "role": summary["seat_roles"].get(pos, "defender"),
                    }
                )

        # Track other statistics (similar to train_ppo.py)
        is_leaster_ep = 1 if summary["is_leaster"] else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if summary["is_called_under"] else 0)
            called_10_window.append(
                1
                if (summary["called_card"] and summary["called_card"].startswith("10"))
                else 0
            )
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        # Track ALONE calls only for games with a picker (exclude leaster)
        if not summary["is_leaster"]:
            # Global population ALONE indicator (any picker)
            alone_call_window.append(1 if summary["alone_called"] else 0)
            # Training-agent-only ALONE indicator (only when training agent is picker)
            if training_data_single["was_picker"]:
                training_alone_window.append(1 if summary["alone_called"] else 0)

        # Count pick/pass decisions from transitions
        episode_picks = sum(
            1
            for ev in episode_events
            if ev["kind"] == "action" and ev["action"] == ACTION_IDS["PICK"]
        )
        episode_passes = sum(
            1
            for ev in episode_events
            if ev["kind"] == "action" and ev["action"] == ACTION_IDS["PASS"]
        )

        pick_decisions[partner_mode].append(episode_picks)
        pass_decisions[partner_mode].append(episode_passes)

        game_count += 1

        # Update model periodically
        if transitions_since_update >= update_interval:
            print(
                f"🔄 Updating model after {game_count} games... (Episode {episode:,})"
            )

            # Entropy decay
            entropy_play_start, entropy_play_end = (
                hyperparams.entropy_play_start,
                hyperparams.entropy_play_end,
            )
            entropy_pick_start, entropy_pick_end = (
                hyperparams.entropy_pick_start,
                hyperparams.entropy_pick_end,
            )
            entropy_partner_start, entropy_partner_end = (
                hyperparams.entropy_partner_start,
                hyperparams.entropy_partner_end,
            )
            entropy_bury_start, entropy_bury_end = (
                hyperparams.entropy_bury_start,
                hyperparams.entropy_bury_end,
            )
            decay_fraction = get_schedule_progress_pct(episode) / 100.0
            training_agent.entropy_coeff_play = (
                entropy_play_start
                + (entropy_play_end - entropy_play_start) * decay_fraction
            )
            training_agent.entropy_coeff_pick = (
                entropy_pick_start
                + (entropy_pick_end - entropy_pick_start) * decay_fraction
            )
            training_agent.entropy_coeff_partner = (
                entropy_partner_start
                + (entropy_partner_end - entropy_partner_start) * decay_fraction
            )
            training_agent.entropy_coeff_bury = (
                entropy_bury_start
                + (entropy_bury_end - entropy_bury_start) * decay_fraction
            )

            # Apply temporary bumps to entropies (shaped mode only; the ExIt
            # hybrid keeps only the baseline entropy decay above).
            if hyperparams.reward_mode == "shaped":
                if episode <= bury_entropy_bump_until:
                    training_agent.entropy_coeff_bury += hyperparams.bury_entropy_bump
                if episode <= partner_entropy_bump_until:
                    training_agent.entropy_coeff_partner += (
                        hyperparams.partner_entropy_bump
                    )
                if episode <= pick_entropy_bump_until:
                    training_agent.entropy_coeff_pick += hyperparams.pick_entropy_bump

            # Learning rate decay (apply scheduled LRs based on training progress)
            progress_pct = min(100.0, max(0.0, (episode / num_episodes) * 100.0))
            scheduled_actor_lr = interpolated_weight(
                hyperparams.lr_schedule_actor, progress_pct
            )
            scheduled_critic_lr = interpolated_weight(
                hyperparams.lr_schedule_critic, progress_pct
            )
            training_agent.set_learning_rates(
                actor_lr=scheduled_actor_lr, critic_lr=scheduled_critic_lr
            )

            # Update
            update_stats = training_agent.update(epochs=4, batch_size=256)

            if update_stats:
                adv_stats = update_stats["advantage_stats"]
                val_stats = update_stats["value_target_stats"]
                num_transitions = update_stats["num_transitions"]
                approx_kl = update_stats.get("approx_kl", None)
                early_stop = update_stats.get("early_stop", False)
                head_entropy = update_stats.get("head_entropy", {})
                pick_pass_adv = update_stats.get("pick_pass_adv", {})
                critic_losses = update_stats.get("critic_losses", {})

                print(f"   Transitions: {num_transitions}")
                print(
                    f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]"
                )
                print(
                    f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]"
                )
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                # Instrumentation logs: head entropy and PICK/PASS advantages
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )
                if pick_pass_adv:
                    print(
                        f"   Adv(PICK): {pick_pass_adv.get('pick_mean', 0.0):+.3f} (n={pick_pass_adv.get('pick_count', 0)}), "
                        f"Adv(PASS): {pick_pass_adv.get('pass_mean', 0.0):+.3f} (n={pick_pass_adv.get('pass_count', 0)})"
                    )
                if "timing" in update_stats:
                    t = update_stats["timing"]
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )
                if critic_losses:
                    print(
                        "   Scaled critic losses - value: %.4f, win: %.4f, return: %.4f, points: %.4f, secret partner: %.4f, seen trump mask: %.4f, unseen higher than hand: %.4f"
                        % (
                            critic_losses.get("value", 0.0),
                            critic_losses.get("win", 0.0),
                            critic_losses.get("return", 0.0),
                            critic_losses.get("points", 0.0),
                            critic_losses.get("secret_partner", 0.0),
                            critic_losses.get("seen_trump_mask", 0.0),
                            critic_losses.get("unseen_trump_higher_than_hand", 0.0),
                        )
                    )

                # Stage C distillation diagnostics (only populated in terminal/ExIt
                # mode; dormant under shaped PPO, so the block self-gates).
                distill = update_stats.get("distill", {})
                if distill.get("pg_masked_fraction", 0.0) > 0.0:
                    print(
                        "   Distill - loss: %.4f, teacher_kl: %.4f, pi' entropy: %.4f, masked frac: %.3f"
                        % (
                            distill.get("loss", 0.0),
                            distill.get("teacher_kl", 0.0),
                            distill.get("pi_target_entropy", 0.0),
                            distill.get("pg_masked_fraction", 0.0),
                        )
                    )
                # ISMCTS search diagnostics over this update window, per head:
                # ESS-abort fraction (searches that missed the ESS floor), mean
                # root ESS, and mean pi' entropy of the accepted targets.
                if sum(s["count"] for s in search_diagnostics_window.values()) > 0:
                    parts = []
                    for head in ("pick", "partner", "bury", "play"):
                        stats = search_diagnostics_window[head]
                        if stats["count"] == 0:
                            continue
                        abort_pct = 100.0 * (1.0 - stats["accepted"] / stats["count"])
                        mean_ess = stats["ess_sum"] / stats["count"]
                        mean_entropy = (
                            (stats["entropy_sum"] / stats["accepted"])
                            if stats["accepted"]
                            else float("nan")
                        )
                        parts.append(
                            f"{head}: n={stats['count']} abort={abort_pct:.0f}% "
                            f"ess={mean_ess:.1f} ent={mean_entropy:.2f}"
                        )
                    print("   Search - " + " | ".join(parts))

            game_count = 0
            transitions_since_update = 0
            for stats in search_diagnostics_window.values():
                stats.update(count=0, accepted=0, ess_sum=0.0, entropy_sum=0.0)

            # Publish fresh weights for parallel workers (no-op in sequential mode).
            if num_workers > 1:
                _publish_weights()

        # Add agent to population periodically
        if episode % population_add_interval == 0:
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                # Snapshot training agent for population opponents
                agent_snapshot = copy.deepcopy(training_agent)
                # Disable epsilon-floor mixing for population agents (no-op in
                # terminal mode, where epsilon is already 0).
                agent_snapshot.set_partner_call_epsilon(
                    hyperparams.partner_call_eps_base
                )
                agent_snapshot.set_pass_floor_epsilon(hyperparams.pass_floor_eps_base)
                agent_snapshot.set_pick_floor_epsilon(hyperparams.pick_floor_eps_base)

                agent_id = population.add_agent(
                    agent=agent_snapshot,
                    partner_mode=mode,
                    training_episodes=episode,
                    parent_id=None,
                    activation=activation,
                    # Seed snapshot rating with current training agent's rating (same μ, large σ)
                    initial_rating=population.rating_model.rating(
                        mu=training_rating.mu,
                        sigma=max(float(training_rating.sigma), 12.5),
                    ),
                )
                print(
                    f"👥 Added training agent snapshot to {get_partner_mode_name(mode)} population (ID: {agent_id})"
                )

            # Log updated diversity stats after population change
            jd_div = population.get_diversity_stats(PARTNER_BY_JD)
            ca_div = population.get_diversity_stats(PARTNER_BY_CALLED_ACE)
            print(
                "   Diversity (JD): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}"
                % (
                    jd_div["avg_pairwise_diversity"],
                    jd_div["diversity_spread"],
                    jd_div["strategic_clusters"],
                    jd_div["alone_rate_range"][0],
                    jd_div["alone_rate_range"][1],
                    jd_div["pick_rate_diversity"]["weak"],
                    jd_div["pick_rate_diversity"]["medium"],
                    jd_div["pick_rate_diversity"]["strong"],
                    jd_div["coverage"]["early_leads"],
                    jd_div["coverage"]["void_events"],
                )
            )
            print(
                "   Diversity (CA): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}"
                % (
                    ca_div["avg_pairwise_diversity"],
                    ca_div["diversity_spread"],
                    ca_div["strategic_clusters"],
                    ca_div["alone_rate_range"][0],
                    ca_div["alone_rate_range"][1],
                    ca_div["pick_rate_diversity"]["weak"],
                    ca_div["pick_rate_diversity"]["medium"],
                    ca_div["pick_rate_diversity"]["strong"],
                    ca_div["coverage"]["early_leads"],
                    ca_div["coverage"]["void_events"],
                )
            )

        # Cross-evaluation periodically
        if episode % cross_eval_interval == 0:
            print(f"🏆 Running cross-evaluation tournaments... (Episode {episode:,})")
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                eval_stats = population.run_cross_evaluation(
                    mode, num_games=40, max_agents=75
                )
                print(
                    f"   {get_partner_mode_name(mode)}: {eval_stats['games_played']} games, "
                    f"avg skill: {eval_stats['avg_skill_after']:.1f}"
                )
            # Rebuild clusters post-tournament to refresh anchors/sampling policies
            # Clustering is computed lazily when sampling; this call just forces recomputation for visibility
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                _ = population._cluster_population(mode)
            print(population.get_population_summary())
            print("-" * 80)

        # Strategic evaluation
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(
                training_agent, num_samples=200
            )

            training_data["strategic_episodes"].append(episode)
            training_data["pick_hand_correlation"].append(
                strategic_metrics["pick_hand_correlation"]
            )
            training_data["picker_trump_rate"].append(
                strategic_metrics["picker_trump_rate"]
            )
            training_data["defender_trump_rate"].append(
                strategic_metrics["defender_trump_rate"]
            )
            training_data["bury_quality_rate"].append(
                strategic_metrics["bury_quality_rate"]
            )

            # Append strategic metrics row to CSV (episode-keyed)
            write_header = (not os.path.exists(strategic_csv)) or (
                os.path.getsize(strategic_csv) == 0
            )
            with open(strategic_csv, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "pick_hand_correlation",
                        "picker_trump_rate",
                        "defender_trump_rate",
                        "bury_quality_rate",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "episode": episode,
                        "pick_hand_correlation": strategic_metrics[
                            "pick_hand_correlation"
                        ],
                        "picker_trump_rate": strategic_metrics["picker_trump_rate"],
                        "defender_trump_rate": strategic_metrics["defender_trump_rate"],
                        "bury_quality_rate": strategic_metrics["bury_quality_rate"],
                    }
                )

            print(
                f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}"
            )
            print(
                f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%"
            )
            print(
                f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%"
            )
            print(
                f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%"
            )

            # --- Check bury quality rate and schedule bury-head entropy bump if too low ---
            current_bury_quality_rate = strategic_metrics["bury_quality_rate"]
            if (
                current_bury_quality_rate < hyperparams.low_bury_quality_threshold
                and episode > bury_entropy_bump_until
            ):
                bury_entropy_bump_until = (
                    episode + hyperparams.bury_entropy_bump_duration
                )
                print(
                    f"   ⚠️  Low bury-quality rate detected ({current_bury_quality_rate:.1f}%). Increasing bury entropy by {hyperparams.bury_entropy_bump:.3f} for the next {hyperparams.bury_entropy_bump_duration:,} episodes."
                )
            if (
                current_bury_quality_rate >= hyperparams.low_bury_quality_threshold
                and episode > bury_entropy_bump_until
                and bury_entropy_bump_until != 0
            ):
                # Reset any expired bump marker to reduce log noise later
                bury_entropy_bump_until = 0

        # Progress reporting
        if episode % 1000 == 0:
            current_avg_picker_score = np.mean(picker_scores) if picker_scores else 0

            # Calculate pick rates
            total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
            total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
            total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
            total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
            current_called_pick_rate = (
                (100 * total_called_picks / (total_called_picks + total_called_passes))
                if (total_called_picks + total_called_passes) > 0
                else 0
            )
            current_jd_pick_rate = (
                (100 * total_jd_picks / (total_jd_picks + total_jd_passes))
                if (total_jd_picks + total_jd_passes) > 0
                else 0
            )
            current_team_diff = (
                np.mean(team_point_differences) if team_point_differences else 0
            )

            # Rolling window rates
            current_leaster_rate = (
                (sum(leaster_window) / len(leaster_window)) * 100
                if leaster_window
                else 0
            )
            current_alone_rate = (
                (sum(alone_call_window) / len(alone_call_window)) * 100
                if alone_call_window
                else 0
            )
            current_training_alone_rate = (
                (sum(training_alone_window) / len(training_alone_window)) * 100
                if training_alone_window
                else 0
            )
            ca_denominator = sum(called_ace_window) or 1
            current_called_under_rate = (
                sum(called_under_window) / ca_denominator
            ) * 100
            current_called_10s_rate = (sum(called_10_window) / ca_denominator) * 100
            current_picker_frequency = (
                (sum(picker_window) / len(picker_window)) * 100 if picker_window else 0
            )
            elapsed = start_time_offset + (time.time() - start_time)

            # Collect data for plotting
            training_data["episodes"].append(episode)
            training_data["picker_avg"].append(current_avg_picker_score)
            training_data["called_pick_rate"].append(current_called_pick_rate)
            training_data["jd_pick_rate"].append(current_jd_pick_rate)
            training_data["learning_rate"].append(
                training_agent.actor_optimizer.param_groups[0]["lr"]
            )
            training_data["time_elapsed"].append(elapsed)
            training_data["team_point_diff"].append(current_team_diff)
            training_data["alone_rate"].append(current_alone_rate)
            training_data["leaster_rate"].append(current_leaster_rate)

            # Append progress row to CSV (create with header if new)
            write_header = (not os.path.exists(progress_csv)) or (
                os.path.getsize(progress_csv) == 0
            )
            with open(progress_csv, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "episode",
                        "picker_avg",
                        "called_pick_rate",
                        "jd_pick_rate",
                        "learning_rate",
                        "time_elapsed",
                        "team_point_diff",
                        "alone_rate",
                        "leaster_rate",
                    ],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(
                    {
                        "episode": episode,
                        "picker_avg": current_avg_picker_score,
                        "called_pick_rate": current_called_pick_rate,
                        "jd_pick_rate": current_jd_pick_rate,
                        "learning_rate": training_agent.actor_optimizer.param_groups[0][
                            "lr"
                        ],
                        "time_elapsed": elapsed,
                        "team_point_diff": current_team_diff,
                        "alone_rate": current_alone_rate,
                        "leaster_rate": current_leaster_rate,
                    }
                )

            # Population statistics
            jd_stats = population.get_population_stats(PARTNER_BY_JD)
            ca_stats = population.get_population_stats(PARTNER_BY_CALLED_ACE)
            training_data["population_stats"].append(
                {
                    "episode": episode,
                    "jd_size": jd_stats["size"],
                    "jd_avg_skill": jd_stats["avg_skill"],
                    "ca_size": ca_stats["size"],
                    "ca_avg_skill": ca_stats["avg_skill"],
                }
            )

            current_run_elapsed = time.time() - start_time
            games_per_min = (
                (episode - start_episode) / (current_run_elapsed / 60)
                if current_run_elapsed > 0
                else 0
            )

            print(
                f"📊 Episode {episode:,}/{num_episodes:,} ({episode / num_episodes * 100:.1f}%)"
            )
            print("   " + "-" * 50)
            print(f"   Picker avg: {current_avg_picker_score:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
            print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
            print(f"   Picker Frequency: {current_picker_frequency:.2f}%")
            print("   " + "-" * 25)
            print(f"   Alone Call Rate (population): {current_alone_rate:.2f}%")
            print(f"   Alone Call Rate (training): {current_training_alone_rate:.2f}%")
            print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
            print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
            print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
            print("   " + "-" * 25)
            print(
                f"   Population JD: {jd_stats['size']} agents (avg skill: {jd_stats['avg_skill']:.1f})"
            )
            print(
                f"   Population CA: {ca_stats['size']} agents (avg skill: {ca_stats['avg_skill']:.1f})"
            )
            print("   " + "-" * 50)
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed / 60:.1f} min")
            print("   " + "-" * 50)

            # --- Adaptive partner-head entropy bump scheduling ---
            if (
                current_training_alone_rate > hyperparams.high_alone_rate_threshold
                or current_training_alone_rate < hyperparams.low_alone_rate_threshold
            ) and episode > partner_entropy_bump_until:
                partner_entropy_bump_until = (
                    episode + hyperparams.partner_entropy_bump_duration
                )
                print(
                    f"   ⚠️  ALONE rate out of band ({current_training_alone_rate:.2f}%). "
                    f"Increasing partner entropy by {hyperparams.partner_entropy_bump:.3f} for the next "
                    f"{hyperparams.partner_entropy_bump_duration:,} episodes."
                )
            if (
                (
                    hyperparams.low_alone_rate_threshold
                    <= current_training_alone_rate
                    <= hyperparams.high_alone_rate_threshold
                )
                and episode > partner_entropy_bump_until
                and partner_entropy_bump_until != 0
            ):
                partner_entropy_bump_until = 0

            # --- Partner CALL mixture epsilon controller (shaped mode only) ---
            # Gradually increase ε when ALONE rate is high and picker avg is poor.
            # Tiered caps: <= mid_picker_avg -> mid cap; <= high_picker_avg -> high cap. Otherwise, decay toward base.
            if hyperparams.reward_mode == "shaped":
                desired_partner_eps_max = hyperparams.partner_call_eps_base
                # Gate epsilon scheduling on training-agent-only ALONE rate
                if current_training_alone_rate > hyperparams.high_alone_rate_ceiling:
                    if (
                        current_avg_picker_score
                        <= hyperparams.partner_call_eps_high_picker_avg_threshold
                    ):
                        desired_partner_eps_max = hyperparams.partner_call_eps_max_high
                    elif (
                        current_avg_picker_score
                        <= hyperparams.partner_call_eps_mid_picker_avg_threshold
                    ):
                        desired_partner_eps_max = hyperparams.partner_call_eps_max_mid
                    else:
                        desired_partner_eps_max = hyperparams.partner_call_eps_base
                else:
                    desired_partner_eps_max = hyperparams.partner_call_eps_base

                desired_partner_eps = current_partner_call_eps
                if desired_partner_eps_max > current_partner_call_eps:
                    desired_partner_eps = min(
                        current_partner_call_eps + hyperparams.partner_call_eps_step_up,
                        desired_partner_eps_max,
                    )
                elif desired_partner_eps_max < current_partner_call_eps:
                    desired_partner_eps = max(
                        current_partner_call_eps
                        - hyperparams.partner_call_eps_step_down,
                        desired_partner_eps_max,
                    )

                if abs(desired_partner_eps - current_partner_call_eps) > 1e-6:
                    current_partner_call_eps = desired_partner_eps
                    training_agent.set_partner_call_epsilon(current_partner_call_eps)
                    print(
                        f"   ⚠️  Partner CALL epsilon ε adjusted to: {current_partner_call_eps:.3f}"
                    )

            # --- Adaptive pick-head entropy bump scheduling ---
            overall_picks = total_called_picks + total_jd_picks
            overall_decisions = overall_picks + total_called_passes + total_jd_passes
            overall_pick_rate = (
                (100 * overall_picks / overall_decisions)
                if overall_decisions > 0
                else 0.0
            )

            if (
                overall_pick_rate > hyperparams.high_pick_rate_threshold
                or overall_pick_rate < hyperparams.low_pick_rate_threshold
            ) and episode > pick_entropy_bump_until:
                pick_entropy_bump_until = (
                    episode + hyperparams.pick_entropy_bump_duration
                )
                print(
                    f"   ⚠️  Pick rate out of band ({overall_pick_rate:.2f}%). "
                    f"Increasing pick entropy by {hyperparams.pick_entropy_bump:.3f} for the next "
                    f"{hyperparams.pick_entropy_bump_duration:,} episodes."
                )
            if (
                (
                    hyperparams.low_pick_rate_threshold
                    <= overall_pick_rate
                    <= hyperparams.high_pick_rate_threshold
                )
                and episode > pick_entropy_bump_until
                and pick_entropy_bump_until != 0
            ):
                pick_entropy_bump_until = 0

            # --- PASS-floor epsilon controller (shaped mode only) ---
            # Activate when pick rate high and picker avg negative.
            if hyperparams.reward_mode == "shaped":
                desired_pass_eps_max = hyperparams.pass_floor_eps_base
                if (overall_pick_rate > hyperparams.high_pick_rate_ceiling) and (
                    current_avg_picker_score
                    < hyperparams.pass_floor_eps_picker_avg_threshold
                ):
                    desired_pass_eps_max = hyperparams.pass_floor_eps_target

                desired_pass_eps = current_pass_floor_eps
                if desired_pass_eps_max > current_pass_floor_eps:
                    desired_pass_eps = min(
                        current_pass_floor_eps + hyperparams.pass_floor_eps_step_up,
                        desired_pass_eps_max,
                    )
                elif desired_pass_eps_max < current_pass_floor_eps:
                    desired_pass_eps = max(
                        current_pass_floor_eps - hyperparams.pass_floor_eps_step_down,
                        desired_pass_eps_max,
                    )

                if abs(desired_pass_eps - current_pass_floor_eps) > 1e-6:
                    current_pass_floor_eps = desired_pass_eps
                    training_agent.set_pass_floor_epsilon(current_pass_floor_eps)
                    print(
                        f"   ⚠️  PASS floor epsilon ε_pass adjusted to: {current_pass_floor_eps:.3f}"
                    )

            # --- PICK-floor epsilon controller (shaped mode only) ---
            # Activate when overall pick rate is very low.
            if hyperparams.reward_mode == "shaped":
                desired_pick_eps_max = hyperparams.pick_floor_eps_base
                if overall_pick_rate < hyperparams.low_pick_rate_floor:
                    desired_pick_eps_max = hyperparams.pick_floor_eps_target

                desired_pick_eps = current_pick_floor_eps
                if desired_pick_eps_max > current_pick_floor_eps:
                    desired_pick_eps = min(
                        current_pick_floor_eps + hyperparams.pick_floor_eps_step_up,
                        desired_pick_eps_max,
                    )
                elif desired_pick_eps_max < current_pick_floor_eps:
                    desired_pick_eps = max(
                        current_pick_floor_eps - hyperparams.pick_floor_eps_step_down,
                        desired_pick_eps_max,
                    )

                if abs(desired_pick_eps - current_pick_floor_eps) > 1e-6:
                    current_pick_floor_eps = desired_pick_eps
                    training_agent.set_pick_floor_epsilon(current_pick_floor_eps)
                    print(
                        f"   ⚠️  PICK floor epsilon ε_pick adjusted to: {current_pick_floor_eps:.3f}"
                    )

        # Save checkpoints
        if episode % save_interval == 0:
            checkpoint_path = (
                f"{checkpoint_dir}/pfsp_{activation}_checkpoint_{episode}.pt"
            )
            training_agent.save(checkpoint_path)

            # Save population state
            population.save_population_state()

            # Save training plot
            if len(training_data["episodes"]) > 10:
                plot_path = f"{checkpoint_dir}/pfsp_training_progress_{episode}.png"
                save_training_plot(training_data, plot_path)  # Reuse plotting function

            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"💾 Checkpoint saved at episode {episode:,}")
            print(
                f"   Time for last {save_interval:,} episodes: {time_since_last / 60:.1f} min"
            )
            remaining_episodes = num_episodes - episode
            if remaining_episodes > 0:
                estimated_time = (
                    remaining_episodes * (time_since_last / save_interval) / 60
                )
                print(f"   Estimated time remaining: {estimated_time:.1f} min")

    # Final update and save
    if training_agent.events:
        print("🔄 Final model update...")
        final_update_stats = training_agent.update()

        if final_update_stats:
            adv_stats = final_update_stats["advantage_stats"]
            val_stats = final_update_stats["value_target_stats"]
            num_transitions = final_update_stats["num_transitions"]

            print(f"   Final Transitions: {num_transitions}")
            print(
                f"   Final Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]"
            )
            print(
                f"   Final Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]"
            )

    training_agent.save(os.path.join(output_dir, f"final_{activation}.pt"))
    population.save_population_state()

    # Save final training plot
    if len(training_data["episodes"]) > 0:
        save_training_plot(
            training_data, os.path.join(output_dir, f"final_{activation}_training.png")
        )

    total_time = time.time() - start_time
    print("\n🎉 PFSP Training completed!")
    print(
        f"   Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.1f} hours)"
    )
    print(
        f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}"
    )
    print(
        f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}"
    )

    # Final population summary
    print("\n" + population.get_population_summary())
