#!/usr/bin/env python3
"""Shared game-generation primitives for population/league training.

This module holds the single-game playing primitive (``play_population_game``)
and the small pure helpers (head classification, schedule interpolation, public
post-game summary) that the league trainer (``train_league_ppo.py``), the
exploiter (``exploiter.py``) and the deploy/audit ISMCTS probes build on. The
old PFSP museum (dual-population management, strategic-profile clustering, the
``run_pfsp_training`` loop and its worker pool) was dissolved in the June 2026
consolidation; ``train_league_ppo.py`` owns the training loop and its own
worker pool now. Hyperparameters live in config.py; pure helpers in
training_utils.py.
"""

import random
from typing import TYPE_CHECKING

import numpy as np

from ppo import PPOAgent
from sheepshead import (
    ACTIONS,
    Game,
)
from training_utils import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)

if TYPE_CHECKING:
    from config import SearchConfig
    from ismcts import ISMCTSTeacher


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
) -> tuple:
    """Play a single game with the training agent and population opponents.

    ``reward_mode`` selects the return: ``"shaped"`` applies the intermediate
    reward shaping + per-trick rewards and ``process_episode_rewards``;
    ``"terminal"`` skips all shaping and uses ``process_terminal_rewards``
    (final_score-only), optionally attaching ISMCTS soft-teacher targets to a
    per-head fraction of the training agent's decisions (search is teacher-only;
    the agent still acts on-policy). The league/exploiter trainers call this with
    ``reward_mode="terminal"`` and no teacher; the search arguments are the
    deploy/audit hook (the ISMCTS engine lives in ismcts.py).

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

    # Population grounding for the teacher: model each non-training seat in the
    # search with the agent ACTUALLY controlling it this game, so teacher Q is
    # the EV against the real field (a self-modeled rollout field can't punish
    # information-revealing play; see notebooks/Population_Grounded_Teacher_Plan.md).
    search_seat_policies = {
        pos: agents[pos - 1]
        for pos in range(1, 6)
        if agents[pos - 1] is not None and agents[pos - 1] is not training_agent
    }

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

    # Map positions to population opponents (returned for the caller's bookkeeping)
    pos_to_pop_agent = {}
    opp_positions = opponent_seat_positions.copy()
    for opp, seat_pos in zip(opponents[: len(opp_positions)], opp_positions):
        pos_to_pop_agent[seat_pos] = opp

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
                                seat_policies=search_seat_policies,
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
