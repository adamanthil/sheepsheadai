from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from server.api.schemas import (
    AnalyzeActionDetail,
    AnalyzeCalibrationSummary,
    AnalyzeGameSummary,
    AnalyzeSeatCalibration,
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
)
from server.config import get_settings
from server.runtime.seating import ANALYZE_SEAT_NAMES
from server.services.ai_loader import load_agent
from server.services.analysis_common import (
    build_observation,
    build_probability_list,
    compute_oracle_values,
    infer_phase_from_action_id,
    run_inference_step,
    set_seed,
)
from sheepshead import ACTION_LOOKUP, DECK, Game
from sheepshead.training.reward_shaping import (
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)


def _setup_simulation(
    req: AnalyzeSimulateRequest,
) -> tuple[Any, Any, Game, Dict[str, List[str]]]:
    """Seed RNGs, load the configured agent, deal the game, and capture the
    initial hands for the summary.

    Stage (a) of ``simulate_game``: deal/agent setup + seeding.
    """
    # Set seed if provided
    if req.seed is not None:
        set_seed(req.seed)

    # Load the configured agent; clients cannot influence which file is read.
    settings = get_settings()
    agent = load_agent(settings.sheepshead_model_path)

    # Reset recurrent state before simulation
    agent.reset_recurrent_state()

    # Initialize game with default double_on_the_bump=True (model doesn't consider this anyway)
    partner_mode = 1 if req.partnerMode == 1 else 0  # Convert to Game's expected format
    # Pass the seed through to the Game so the deal itself is reproducible:
    # Game uses its own local RNG instance, which ignores the global seeds that
    # set_seed() configures. Without this, the same seed reshuffles a new deal
    # on every run.
    game = Game(
        partner_selection_mode=partner_mode,
        seed=req.seed,
    )

    # Capture initial hands for summary
    initial_hands: Dict[str, List[str]] = {}
    for player in game.players:
        initial_hands[ANALYZE_SEAT_NAMES[player.position - 1]] = sorted(
            list(player.hand), key=lambda card: DECK.index(card)
        )

    return agent, settings, game, initial_hands


def _compute_discounted_returns(
    trace: List[AnalyzeActionDetail],
    episode_transitions: List[Dict[str, Any]],
    game: Game,
    agent: Any,
) -> None:
    """Replay the trainers' reward-shaping math over the simulated episode
    and attach per-step rewards and discounted returns back onto ``trace``
    (mutated in place, aligned by index).

    Stage (c) of ``simulate_game``: the reward-shaping replay + discounted-
    return computation. Both reward schedules are computed from the SAME
    reward_shaping functions the trainers use: the shaped baseline
    (``process_episode_rewards``) and the terminal-only return
    (``process_terminal_rewards``, i.e. the league trainer's
    ``reward_mode="terminal"``). The client toggles between them.
    """
    if not episode_transitions:
        return

    final_scores = [p.get_score() for p in game.players]
    head_shaping = [tr["head_shaping_reward"] for tr in episode_transitions]

    # Group indices by player (to compute rewards per player sequence)
    idxs_by_player: Dict[int, List[int]] = {}
    for i, tr in enumerate(episode_transitions):
        idxs_by_player.setdefault(tr["player"].position, []).append(i)

    # Fill rewards aligned to original order by calling per-player.
    rewards = [0.0] * len(episode_transitions)
    terminal_rewards = [0.0] * len(episode_transitions)
    dones = [False] * len(episode_transitions)
    for idxs in idxs_by_player.values():
        acts = [episode_transitions[i] for i in idxs]
        # Per-player rewards (last item is terminal)
        for offset, reward_data in enumerate(
            process_episode_rewards(
                acts,
                final_scores,
                game.is_leaster,
            )
        ):
            rewards[idxs[offset]] = float(reward_data["reward"])
        for offset, reward_data in enumerate(
            process_terminal_rewards(
                acts,
                final_scores,
                game.is_leaster,
            )
        ):
            terminal_rewards[idxs[offset]] = float(reward_data["reward"])
        if idxs:
            dones[idxs[-1]] = True

    discounted_by_index: Dict[int, float] = {}
    for _, idxs in idxs_by_player.items():
        ret = 0.0
        for idx in reversed(idxs):
            if dones[idx]:
                ret = 0.0
            ret = rewards[idx] + agent.gamma * ret
            discounted_by_index[idx] = ret

    # Attach discounted returns and per-step rewards back to trace elements (aligned by order)
    for i, action_detail in enumerate(trace):
        if i in discounted_by_index:
            action_detail.discountedReturn = float(discounted_by_index[i])
        # Always provide the raw per-step reward value for this action
        if i < len(rewards):
            action_detail.stepReward = float(rewards[i])
            action_detail.stepRewardHeadShaping = float(head_shaping[i])
            action_detail.stepRewardBase = float(rewards[i]) - float(
                action_detail.stepRewardHeadShaping
            )
            action_detail.stepRewardTerminal = float(terminal_rewards[i])


def _build_calibration_summary(
    trace: List[AnalyzeActionDetail], game: Game, players: List[str]
) -> Optional[AnalyzeCalibrationSummary]:
    """Roll the per-step aux-head predictions up into game-level calibration
    metrics against the final outcome. None when the game didn't finish or
    the model has no aux heads (no winProb anywhere in the trace)."""
    if not game.is_done():
        return None
    if not any(step.winProb is not None for step in trace):
        return None

    final_scores = [p.get_score() for p in game.players]
    final_points = list(game.points_taken)

    # Point-prediction errors, grouped by the seat being predicted ABOUT.
    point_errors_about: Dict[int, List[float]] = {s: [] for s in range(1, 6)}
    for step in trace:
        for est in step.pointEstimates or []:
            seat = int(est["seat"] if isinstance(est, dict) else est.seat)
            points = float(est["points"] if isinstance(est, dict) else est.points)
            point_errors_about[seat].append(abs(points - float(final_points[seat - 1])))

    trump_correct = 0
    trump_total = 0
    for step in trace:
        for entry in step.trumpSeenMask or []:
            prob = float(
                entry["probabilitySeen"]
                if isinstance(entry, dict)
                else entry.probabilitySeen
            )
            actual = bool(
                entry["actualSeen"] if isinstance(entry, dict) else entry.actualSeen
            )
            trump_total += 1
            if (prob > 0.5) == actual:
                trump_correct += 1

    seats: List[AnalyzeSeatCalibration] = []
    all_sq_errors: List[float] = []
    for seat in range(1, 6):
        win_probs = [
            float(step.winProb)
            for step in trace
            if step.seat == seat and step.winProb is not None
        ]
        if not win_probs:
            continue
        won = final_scores[seat - 1] > 0
        target = 1.0 if won else 0.0
        sq_errors = [(p - target) ** 2 for p in win_probs]
        all_sq_errors.extend(sq_errors)
        errors_about = point_errors_about[seat]
        seats.append(
            AnalyzeSeatCalibration(
                seat=seat,
                seatName=players[seat - 1],
                won=won,
                decisionCount=len(win_probs),
                firstWinProb=win_probs[0],
                lastWinProb=win_probs[-1],
                meanWinProb=sum(win_probs) / len(win_probs),
                brierScore=sum(sq_errors) / len(sq_errors),
                pointsMae=(
                    sum(errors_about) / len(errors_about) if errors_about else 0.0
                ),
            )
        )

    all_point_errors = [e for errs in point_errors_about.values() for e in errs]
    return AnalyzeCalibrationSummary(
        seats=seats,
        overallBrier=(
            sum(all_sq_errors) / len(all_sq_errors) if all_sq_errors else 0.0
        ),
        overallPointsMae=(
            sum(all_point_errors) / len(all_point_errors) if all_point_errors else 0.0
        ),
        trumpMaskAccuracy=(trump_correct / trump_total) if trump_total else None,
        trumpMaskCount=trump_total,
    )


def _build_final_summary(
    game: Game, initial_hands: Dict[str, List[str]], players: List[str]
) -> tuple[Optional[Dict[str, Any]], Optional[AnalyzeGameSummary]]:
    """Build the final-state payload and the game summary once the
    simulation reaches a terminal game state.

    Stage (d) of ``simulate_game`` (part 1): response assembly.
    """
    final_payload: Optional[Dict[str, Any]] = None
    game_summary: Optional[AnalyzeGameSummary] = None
    if game.is_done():
        from server.runtime.tables import build_player_state

        # Use any player to get the final state
        final_state = build_player_state(game.players[0])
        final_payload = final_state["view"].get("final")

        # Build game summary
        picker_seat = game.picker if hasattr(game, "picker") and game.picker else 0
        partner_seat = game.partner if hasattr(game, "partner") and game.partner else 0

        # If partner not revealed yet, try to find the secret partner
        if partner_seat == 0 and picker_seat > 0 and not game.alone_called:
            for player in game.players:
                if player.is_secret_partner:
                    partner_seat = player.position
                    break

        # Get bury cards (from the picker or game)
        bury_cards = []
        if picker_seat > 0:
            picker_player = game.players[picker_seat - 1]
            bury_cards = (
                list(picker_player.bury) if hasattr(picker_player, "bury") else []
            )
        elif hasattr(game, "bury"):
            bury_cards = list(game.bury)

        # Get point totals
        picker_points = 0
        defender_points = 0
        if final_payload and final_payload.get("mode") == "standard":
            picker_points = final_payload.get("picker_score", 0)
            defender_points = final_payload.get("defender_score", 0)

        # Get final scores
        scores = [p.get_score() for p in game.players] if final_payload else [0] * 5

        game_summary = AnalyzeGameSummary(
            hands=initial_hands,
            blind=game.blind,
            picker=players[picker_seat - 1] if picker_seat > 0 else None,
            partner=players[partner_seat - 1] if partner_seat > 0 else None,
            bury=bury_cards,
            pickerPoints=picker_points,
            defenderPoints=defender_points,
            scores=scores,
        )

    return final_payload, game_summary


def _assemble_response(
    req: AnalyzeSimulateRequest,
    settings: Any,
    agent: Any,
    players: List[str],
    game_summary: Optional[AnalyzeGameSummary],
    calibration: Optional[AnalyzeCalibrationSummary],
    trace: List[AnalyzeActionDetail],
    final_payload: Optional[Dict[str, Any]],
) -> AnalyzeSimulateResponse:
    """Assemble the final ``AnalyzeSimulateResponse``.

    Stage (d) of ``simulate_game`` (part 2): response assembly.
    """
    return AnalyzeSimulateResponse(
        meta={
            "partnerMode": req.partnerMode,
            "deterministic": req.deterministic,
            "seed": req.seed,
            "model": settings.sheepshead_model_label,
            "gamma": float(agent.gamma),
            "criticMode": getattr(agent, "critic_mode", "limited"),
            "hasOracle": getattr(agent, "oracle_critic", None) is not None,
        },
        summary=game_summary,
        calibration=calibration,
        trace=trace,
        final=final_payload,
    )


def simulate_game(req: AnalyzeSimulateRequest) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""

    agent, settings, game, initial_hands = _setup_simulation(req)

    # Player display names
    players = ANALYZE_SEAT_NAMES

    from server.runtime.tables import build_player_state

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trace storage
    trace: List[AnalyzeActionDetail] = []
    step_index = 0
    # For reward shaping and discounted return computation
    episode_transitions: List[Dict[str, Any]] = []
    current_trick_transitions: List[Dict[str, Any]] = []
    # Oracle critic event streams (per seat, chronological), collected the
    # same way the trainer does with collect_oracle=True.
    has_oracle = getattr(agent, "oracle_critic", None) is not None
    oracle_events: Dict[int, List[Dict[str, Any]]] = {}
    oracle_decision_pos: List[tuple[int, int]] = []  # aligned with trace

    # Simulation loop
    while not game.is_done() and step_index < req.maxSteps:
        # Find the current actor seat
        actor_seat = None
        actor_player = None
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            if valid_actions:
                actor_seat = player.position
                actor_player = player
                break

        if actor_player is None:
            # No valid actions found, game should be done
            break

        inference = run_inference_step(agent, actor_player, actor_seat, players, device)

        if has_oracle:
            # Must be captured at decision time: hands shrink as cards play.
            seq = oracle_events.setdefault(actor_seat, [])
            oracle_decision_pos.append((actor_seat, len(seq)))
            seq.append(actor_player.get_oracle_state_dict())

        # Choose action
        if req.deterministic:
            action_id = (
                torch.argmax(inference.action_probs, dim=1).item() + 1
            )  # Convert to 1-indexed
        else:
            dist = torch.distributions.Categorical(inference.action_probs)
            action_id = dist.sample().item() + 1  # Convert to 1-indexed

        # Per-valid-action probabilities, sorted descending
        probabilities = build_probability_list(
            inference.action_probs, inference.logits, inference.valid_actions
        )

        # Get player state/view
        player_state = build_player_state(actor_player)

        # Infer phase
        phase = infer_phase_from_action_id(action_id, agent.action_groups)

        # Create action detail
        action_detail = AnalyzeActionDetail(
            stepIndex=step_index,
            seat=actor_seat,
            seatName=players[actor_seat - 1],  # Convert to 0-indexed for array access
            phase=phase,
            actionId=action_id,
            action=ACTION_LOOKUP[action_id],
            valueEstimate=float(inference.value.item()),
            discountedReturn=None,
            validActionIds=inference.valid_actions,
            probabilities=probabilities,
            view=player_state["view"],
            observation=build_observation(inference.state, actor_seat, players),
            winProb=float(inference.win_prob_val)
            if inference.win_prob_val is not None
            else None,
            expectedFinalReturn=inference.expected_final_val,
            secretPartnerProb=float(inference.secret_partner_prob)
            if inference.secret_partner_prob is not None
            else None,
            pointEstimates=inference.point_estimates or None,
            pointActuals=inference.point_actuals or None,
            trumpSeenMask=inference.trump_seen_mask or None,
            unseenTrumpHigherThanHandProb=inference.unseen_trump_higher_than_hand_prob,
            unseenTrumpHigherThanHandActual=inference.unseen_trump_higher_than_hand_actual,
            memoryCosineDistance=inference.memory_cosine_distance,
            memoryNorm=inference.memory_norm,
        )

        trace.append(action_detail)

        # Build training-like transition for reward shaping
        transition = {
            "player": actor_player,
            "state": inference.state,
            "action": action_id,
            "log_prob": 0.0,
            "value": float(inference.value.item()),
            "valid_actions": set(inference.valid_actions),
            "intermediate_reward": 0.0,
        }

        # Apply shared intermediate shaping and trick tracking
        update_intermediate_rewards_for_action(
            game,
            actor_player,
            action_id,
            transition,
            current_trick_transitions,
        )

        episode_transitions.append(transition)

        # Apply action
        actor_player.act(action_id)

        # Trick completion rewards, if any
        trick_completed = handle_trick_completion(game, current_trick_transitions)
        if trick_completed:
            # Propagate an observation for the just-completed trick to all seats
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
                # Trainer protocol: no observation event after the final trick.
                if has_oracle and not game.is_done():
                    oracle_events.setdefault(seat.position, []).append(
                        seat.get_last_trick_oracle_state_dict()
                    )

        step_index += 1

    # Get final state and build summary if game is done
    final_payload, game_summary = _build_final_summary(game, initial_hands, players)

    # Compute discounted returns per action if we have any transitions
    _compute_discounted_returns(trace, episode_transitions, game, agent)

    if has_oracle:
        compute_oracle_values(
            agent, oracle_events, oracle_decision_pos, trace, device
        )

    calibration = _build_calibration_summary(trace, game, players)

    # Build response
    return _assemble_response(
        req, settings, agent, players, game_summary, calibration, trace, final_payload
    )
