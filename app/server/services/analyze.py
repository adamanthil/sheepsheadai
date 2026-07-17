from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from server.api.schemas import (
    AnalyzeActionDetail,
    AnalyzeCalibrationSummary,
    AnalyzeMemoryObserve,
    AnalyzeSeatCalibration,
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
)
from server.config import get_settings
from server.runtime.seating import ANALYZE_SEAT_NAMES
from server.services.ai_loader import load_agent
from server.runtime.tables import build_player_state
from server.services.analysis_common import (
    build_action_detail,
    compute_oracle_values,
    find_actor,
    memory_drift,
    run_inference_step,
    select_action_id,
    set_seed,
)
from sheepshead import Game
from sheepshead.training.reward_shaping import (
    handle_trick_completion,
    process_episode_rewards,
    process_terminal_rewards,
    update_intermediate_rewards_for_action,
)


def _setup_simulation(
    req: AnalyzeSimulateRequest,
) -> tuple[Any, Any, Game]:
    """Seed RNGs, load the configured agent, and deal the game.

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

    return agent, settings, game


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

    # Point-prediction errors, grouped by the seat being predicted ABOUT.
    # The points head's training target is the points known at decision time
    # (compute_known_points_rel) — a deterministic state-tracking task — so
    # errors are measured against each step's own pointActuals, not the
    # final totals.
    point_errors_about: Dict[int, List[float]] = {s: [] for s in range(1, 6)}
    for step in trace:
        actual_by_seat: Dict[int, float] = {
            act.seat: act.points for act in step.pointActuals or []
        }
        for est in step.pointEstimates or []:
            if est.seat in actual_by_seat:
                point_errors_about[est.seat].append(
                    abs(est.points - actual_by_seat[est.seat])
                )

    trump_correct = 0
    trump_total = 0
    for step in trace:
        for entry in step.trumpSeenMask or []:
            trump_total += 1
            if (entry.probabilitySeen > 0.5) == entry.actualSeen:
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


def _build_final_payload(game: Game) -> Optional[Dict[str, Any]]:
    """Build the final-state payload once the simulation reaches a terminal
    game state.

    Stage (d) of ``simulate_game`` (part 1): response assembly.
    """
    if not game.is_done():
        return None
    # Use any player to get the final state
    final_state = build_player_state(game.players[0])
    return final_state["view"].get("final")


def _assemble_response(
    req: AnalyzeSimulateRequest,
    settings: Any,
    agent: Any,
    calibration: Optional[AnalyzeCalibrationSummary],
    trace: List[AnalyzeActionDetail],
    memory_observes: List[AnalyzeMemoryObserve],
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
        calibration=calibration,
        trace=trace,
        memoryObserves=memory_observes,
        final=final_payload,
    )


def simulate_game(req: AnalyzeSimulateRequest) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""

    agent, settings, game = _setup_simulation(req)

    # Player display names
    players = ANALYZE_SEAT_NAMES

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
    # Trick-completion observes are memory updates too; measure their drift
    # so the memory chart shows the full update sequence.
    memory_observes: List[AnalyzeMemoryObserve] = []
    tricks_observed = 0

    # Simulation loop
    while not game.is_done() and step_index < req.maxSteps:
        actor_player = find_actor(game)
        if actor_player is None:
            # No valid actions found, game should be done
            break
        actor_seat = actor_player.position

        inference = run_inference_step(agent, actor_player, actor_seat, players, device)

        if has_oracle:
            # Must be captured at decision time: hands shrink as cards play.
            seq = oracle_events.setdefault(actor_seat, [])
            oracle_decision_pos.append((actor_seat, len(seq)))
            seq.append(actor_player.get_oracle_state_dict())

        action_id = select_action_id(inference.action_probs, req.deterministic)

        trace.append(
            build_action_detail(
                step_index,
                actor_seat,
                players,
                agent,
                inference,
                action_id,
                build_player_state(actor_player)["view"],
            )
        )

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
                memory_before = agent.get_recurrent_memory(
                    seat.position, device=device
                )
                agent.observe(seat.get_last_trick_state_dict(), player_id=seat.position)
                memory_after = agent.get_recurrent_memory(
                    seat.position, device=device
                )
                distance, norm = memory_drift(memory_before, memory_after)
                memory_observes.append(
                    AnalyzeMemoryObserve(
                        afterStepIndex=step_index,
                        trick=tricks_observed,
                        seat=seat.position,
                        seatName=players[seat.position - 1],
                        memoryCosineDistance=distance,
                        memoryNorm=norm,
                    )
                )
                # Trainer protocol: no observation event after the final trick.
                if has_oracle and not game.is_done():
                    oracle_events.setdefault(seat.position, []).append(
                        seat.get_last_trick_oracle_state_dict()
                    )
            tricks_observed += 1

        step_index += 1

    # Get final state if game is done
    final_payload = _build_final_payload(game)

    # Compute discounted returns per action if we have any transitions
    _compute_discounted_returns(trace, episode_transitions, game, agent)

    if has_oracle:
        compute_oracle_values(
            agent, oracle_events, oracle_decision_pos, trace, device
        )

    calibration = _build_calibration_summary(trace, game, players)

    # Build response
    return _assemble_response(
        req, settings, agent, calibration, trace, memory_observes, final_payload
    )
