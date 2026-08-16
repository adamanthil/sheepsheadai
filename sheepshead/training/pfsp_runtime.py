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
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from sheepshead import (
    Game,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import infer_head, is_private_action
from sheepshead.training.training_utils import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)

if TYPE_CHECKING:
    from sheepshead.ismcts import ISMCTSTeacher
    from sheepshead.training.config import SearchConfig


def _is_private_decision(valid_actions) -> bool:
    """True when the decision is a private bury/under (excluded from the public
    record fed to the ISMCTS teacher's forced replay)."""
    return any(is_private_action(a) for a in valid_actions)


def play_cell(game, player) -> str:
    """Node-class label for a PLAY decision: ``t{trick}-{role}-{lead|follow}``.

    The E9 stratification (Search_Teacher_Design §3): the three axes every
    prior study conditioned on, all computable from public state at decision
    time — which is what lets the gated teacher classify nodes online.
    """
    pos = player.position
    if pos == game.picker:
        role = "picker"
    elif pos == game.partner or player.is_secret_partner:
        role = "partner"
    else:
        role = "defender"
    kind = (
        "lead" if all(c == "" for c in game.history[game.current_trick]) else "follow"
    )
    return f"t{game.current_trick}-{role}-{kind}"


def _search_head(valid_actions) -> str:
    """Classify a decision into a search head (delegates to ismcts.infer_head)."""
    return infer_head(valid_actions)


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


def _setup_seats(
    training_agent: PPOAgent,
    opponents: list,
    training_agent_position: int,
) -> tuple[list, dict]:
    """Build the position-to-agent seat mapping (shuffling which opponent sits
    where) and derive the position->pop-agent map from it."""
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

    # Map positions to population opponents (returned for the caller's bookkeeping)
    pos_to_pop_agent = {}
    opp_positions = opponent_seat_positions.copy()
    for opp, seat_pos in zip(opponents[: len(opp_positions)], opp_positions):
        pos_to_pop_agent[seat_pos] = opp

    return agents, pos_to_pop_agent


def _attach_gated_search_target(
    game,
    player,
    valid_actions,
    transition: dict,
    teacher: "ISMCTSTeacher",
    determinization_rng: "random.Random",
    search_config: "SearchConfig",
    forced_public: list[tuple[int, int]],
    search_diagnostics: dict,
    live_probs: "np.ndarray | None",
) -> None:
    """Agreement-gated soft-teacher target (Search_Teacher_Design §9, §12.1).

    Run the same cheap search ``gate_replicates`` times with independent RNG
    (a committee of stochastic experts — query-by-committee, Seung et al.
    1992). Emit a target only when >= ``gate_agreement`` replicates agree on
    the SAME action and it differs from the LIVE policy's greedy choice.
    The emitted target is the replicate-AVERAGED pi_gumbel distribution
    (Danihelka et al. 2022; averaging = root parallelization, Chaslot et al.
    2008), so near-equivalent cards share mass contextually instead of the
    label pretending 7C beats 8C. Abstention is the designed common case:
    E9 certification showed single cheap searches at near-tie nodes flip to
    worse actions ~11% of the time, while 2-of-3-agreed non-policy labels
    measured +0.0112 mean uplift with 0/22 harm.

    Stationary expert vs live student (§12.1, DAgger — Ross et al. 2011):
    the teacher wraps a FROZEN snapshot of the generation-start policy
    (priors, rollouts, critic leaves), so the expert cannot chase a
    drifting student out of its certified regime — attempt 7 showed the
    live-expert loop re-labels its own drift (emission-gap rebound, greedy
    t0 trump-lead climbing linearly past the certified band). The
    referent/anchors, by contrast, must come from the LIVE policy
    (``live_probs``, stashed by the caller's act() — a second forward pass
    would advance the recurrent memory): emission compares the committee
    against the student's CURRENT argmax, so the gate self-retires exactly
    as the student adopts the expert's choices, and the margin-loss clip
    anchors measure the student's label-time state.

    Self-play worlds (no ``seat_policies``): E8 found no ecology effect, and
    the E9 calibration this gate rests on searched self-play continuations —
    population grounding here would decalibrate the gate.

    Eligibility: PLAY head in BOTH partner-selection modes (called-ace and
    jack-of-diamonds — ``play_cell`` role detection is mode-aware via
    ``is_secret_partner``), no leaster / alone games, >= 2 legal actions,
    node class in ``gate_cells``, then ``gate_node_prob`` subsampling (the
    budget knob). Calibration caveat: the E9 map and gate calibration were
    measured on called-ace deals; JD-mode labels extrapolate that
    calibration — defensible because the gate mechanism is mode-agnostic
    and abstains wherever the committee splits, but a JD-mode spot-check
    of emission quality is a recorded follow-up (Search_Teacher_Design §9).
    """
    if game.is_leaster or game.alone_called or len(valid_actions) < 2:
        return
    if _search_head(valid_actions) != "play":
        return
    if play_cell(game, player) not in search_config.gate_cells:
        return
    if determinization_rng.random() >= search_config.gate_node_prob:
        return
    if live_probs is None:
        return  # no live referent/anchors possible -> label would be a no-op

    diag = search_diagnostics["play"]
    diag["count"] += 1
    # LIVE policy's greedy choice + label-time distribution: the emission
    # referent and the margin-loss clip anchors.
    live_argmax = int(max(valid_actions, key=lambda a: float(live_probs[a - 1])))
    picks: list[int] = []
    gumbels: list[np.ndarray] = []
    for _ in range(search_config.gate_replicates):
        rng = random.Random(determinization_rng.getrandbits(64))
        res = teacher.search(
            game,
            player.position,
            list(forced_public),
            rng,
            d_rollout=search_config.gate_d_rollout,
        )
        if not res["ok"] or res.get("pi_gumbel") is None:
            continue
        gum = res["pi_gumbel"]
        picks.append(int(max(res["valid"], key=lambda a: float(gum[a - 1]))))
        gumbels.append(np.asarray(gum, dtype=np.float64))
        if (
            picks
            and Counter(picks).most_common(1)[0][1] >= search_config.gate_agreement
        ):
            # Committee early-stop: the majority is decided, so further
            # replicates cannot change the gate outcome — identical decisions
            # at ~25% less search (measured agreement ~0.85). The emitted
            # target averages the replicates actually run.
            break
    if not picks:
        return
    top_action, top_count = Counter(picks).most_common(1)[0]
    diag["ess_sum"] += float(top_count) / len(picks)  # committee agreement rate
    if top_count < search_config.gate_agreement or top_action == live_argmax:
        return  # abstain: no majority, or committee backs the live policy
    if getattr(search_config, "gate_target", "agreed_onehot") == "avg_gumbel":
        # Study-only (see config): near-uniform at near-ties -> entropy bomb.
        target = np.mean(gumbels, axis=0)
        total = float(target.sum())
        if total <= 0.0:
            return
        target /= total
    else:
        # Smoothed one-hot on the agreed action — the calibrated semantics.
        eps = float(search_config.gate_target_smooth)
        target = np.zeros_like(gumbels[0])
        others = [a for a in valid_actions if a != top_action]
        target[top_action - 1] = 1.0 - eps if others else 1.0
        for a in others:
            target[a - 1] = eps / len(others)
    transition["search_target"] = target.tolist()
    transition["has_search_target"] = True
    # LIVE policy's label-time argmax: the margin loss's ranking referent
    # (the claim is "committee prefers a* over the action the student would
    # take HERE, NOW").
    transition["search_ref_action"] = live_argmax
    # LIVE label-time log-probs of the pair: the anchors for the pair-gap
    # trust region in the margin loss (analog of PPO's ratio clip, Schulman
    # et al. 2017; see ppo.py).
    star_logp = float(np.log(max(float(live_probs[top_action - 1]), 1e-12)))
    ref_logp = float(np.log(max(float(live_probs[live_argmax - 1]), 1e-12)))
    transition["search_star_logp"] = star_logp
    transition["search_ref_logp"] = ref_logp
    diag["accepted"] += 1
    # Label-time gap g = log pi(a_ref) - log pi(a*): how strongly the policy
    # disagreed with the committee. Sizes the trust-region delta ((median g +
    # m)/2 lets the median label complete in one update) and identifies the
    # high-g tail the clip is meant to rate-limit.
    diag["gap_sum"] = diag.get("gap_sum", 0.0) + (ref_logp - star_logp)
    nonzero = target[target > 0]
    diag["entropy_sum"] += float(-(nonzero * np.log(nonzero)).sum())


def _finalize_rewards(
    episode_transitions: list,
    final_scores: list,
    is_leaster: bool,
    shaped: bool,
) -> list:
    # Compute rewards for training agent actions. Shaped: intermediate + final
    # (+ leaster bonus). Terminal: final_score-only on the last action, no shaping
    # and no leaster bonus (get_score scores leasters correctly).
    reward_fn = process_episode_rewards if shaped else process_terminal_rewards
    # Both reward functions require "a chronological list of action
    # transitions for one player only" — the terminal return belongs on each
    # player's own last action. Group the merged multi-seat list per player:
    # feeding it whole put the single terminal reward on the globally-last
    # actor and left every other collecting seat's stream all-zero.
    reward_map = {}
    by_player: dict[int, list] = {}
    for t in episode_transitions:
        if t["kind"] == "action":
            by_player.setdefault(t["player"].position, []).append(t)
    for group in by_player.values():
        for reward_data in reward_fn(group, final_scores, is_leaster):
            reward_map[id(reward_data["transition"])] = reward_data["reward"]

    # Build final episode event stream for storage
    episode_events = []
    for ev in episode_transitions:
        if ev["kind"] == "observation":
            out = {
                "kind": "observation",
                "state": ev["state"],
                "player_id": ev["player"].position,
            }
        else:
            seat_pos = ev["player"].position
            out = {
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
                # Margin-loss ranking referent + label-time pair log-priors
                # (trust-region anchors): without them a labeled row
                # contributes ZERO distill loss (ppo.py hardens missing
                # referents/anchors to no-op), so dropping a key here
                # silently disarms the teacher while leaving the PG-mask
                # active — exactly the attempt-5a bug (notebook §10.3).
                "search_ref_action": ev.get("search_ref_action"),
                "search_star_logp": ev.get("search_star_logp"),
                "search_ref_logp": ev.get("search_ref_logp"),
            }
        if ev.get("oracle_state") is not None:
            out["oracle_state"] = ev["oracle_state"]
        episode_events.append(out)
    return episode_events


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
    collect_oracle: bool = False,
    game_seed: int | None = None,
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

    ``collect_oracle``: attach a full-information ``oracle_state`` (captured at
    decision time, while the Game holds the hidden cards) to every training-agent
    event, for the privileged critic's GAE baseline (critic_mode="oracle").
    When False the event dicts are byte-identical to the historical schema.

    Returns:
        tuple: (game, episode_events, final_scores, training_agent_data, opponents_by_position)
    """
    game = (
        Game(partner_selection_mode=partner_mode, seed=game_seed)
        if game_seed is not None
        else Game(partner_selection_mode=partner_mode)
    )
    weights = shaping_weights or {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 1.0}
    shaped = reward_mode == "shaped"
    search_enabled = (
        reward_mode == "terminal"
        and teacher is not None
        and determinization_rng is not None
        and search_config is not None
        and search_config.enabled
    )
    if search_enabled:
        # The gate reads the LIVE policy's label-time distribution (referent
        # + clip anchors) from the act() stash — see _attach_gated_search_target.
        training_agent.stash_action_probs = True
    # Public (seat, action_id) record for the teacher's forced replay (search only).
    forced_public: list[tuple[int, int]] = []
    # Per-game gate diagnostics (the gated teacher searches PLAY nodes only):
    # gate firings (count), emitted labels (accepted), summed committee
    # agreement rate (ess_sum) and emitted-target entropy for averaging.
    # Attached to training_agent_data so the driver can window + log them.
    search_diagnostics = {
        "play": {"count": 0, "accepted": 0, "ess_sum": 0.0, "entropy_sum": 0.0}
    }

    # Reset recurrent states for all agents
    training_agent.reset_recurrent_state()
    for opponent in opponents:
        opponent.agent.reset_recurrent_state()

    agents, pos_to_pop_agent = _setup_seats(
        training_agent, opponents, training_agent_position
    )

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

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
                    if collect_oracle:
                        transition["oracle_state"] = player.get_oracle_state_dict()
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
                        _attach_gated_search_target(
                            game,
                            player,
                            valid_actions,
                            transition,
                            teacher,
                            determinization_rng,
                            search_config,
                            forced_public,
                            search_diagnostics,
                            training_agent.last_action_probs,
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
                            obs_transition = {
                                "kind": "observation",
                                "player": seat,
                                "state": seat.get_last_trick_state_dict(),
                            }
                            if collect_oracle:
                                obs_transition["oracle_state"] = (
                                    seat.get_last_trick_oracle_state_dict()
                                )
                            episode_transitions.append(obs_transition)
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

    episode_events = _finalize_rewards(
        episode_transitions, final_scores, game.is_leaster, shaped
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
