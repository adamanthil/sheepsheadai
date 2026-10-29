from __future__ import annotations

from typing import Any, Dict, List, Optional

from server.api.schemas import (
    AnalyzeActionDetail,
    AnalyzeCalibrationSummary,
    AnalyzeMemoryObserve,
    AnalyzeSeatCalibration,
    AnalyzeSimulateMeta,
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
)
from server.config import get_settings
from server.services.ai_loader import inference_turn_in_thread, load_agent
from server.services.analysis_common import (
    build_action_detail,
    compute_oracle_values,
    deployment_sampler,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.agent.traced_game import (
    SEAT_NAMES,
    default_device,
    play_traced_game,
)
from sheepshead.game import Game
from sheepshead.rewards import process_episode_rewards, process_terminal_rewards


def _compute_discounted_returns(
    trace: List[AnalyzeActionDetail],
    episode_transitions: List[Dict[str, Any]],
    game: Game,
    agent: Any,
) -> None:
    """Replay the trainers' reward-shaping math over the simulated episode
    and attach per-step rewards and discounted returns back onto ``trace``
    (mutated in place, aligned by index).

    The reward-shaping replay + discounted-return computation. Both reward schedules are computed from the SAME
    sheepshead.rewards functions the trainers use: the shaped baseline
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


def _assemble_response(
    req: AnalyzeSimulateRequest,
    settings: Any,
    agent: Any,
    calibration: Optional[AnalyzeCalibrationSummary],
    trace: List[AnalyzeActionDetail],
    memory_observes: List[AnalyzeMemoryObserve],
    final_payload: Optional[Dict[str, Any]],
) -> AnalyzeSimulateResponse:
    """Assemble the final ``AnalyzeSimulateResponse``."""
    return AnalyzeSimulateResponse(
        meta=AnalyzeSimulateMeta(
            partnerMode=req.partnerMode,
            deterministic=req.deterministic,
            sampling=(
                "argmax"
                if req.deterministic
                else deployment_sampler(settings, req.seed)[1]
            ),
            seed=req.seed,
            model=settings.sheepshead_model_label,
            gamma=float(agent.gamma),
            criticMode=getattr(agent, "critic_mode", "limited"),
            hasOracle=getattr(agent, "oracle_critic", None) is not None,
        ),
        calibration=calibration,
        trace=trace,
        memoryObserves=memory_observes,
        final=final_payload,
    )


def simulate_game(
    req: AnalyzeSimulateRequest, agent: PPOAgent | None = None
) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace.

    ``agent`` defaults to the configured server model; clients of the API
    cannot influence which file is read. The game itself is played by
    sheepshead.agent.traced_game, the same path the research scanners run.
    """
    settings = get_settings()
    if agent is None:
        agent = load_agent(settings.sheepshead_model_path)
    sampler, _ = deployment_sampler(settings, req.seed)
    device = default_device()

    traced = play_traced_game(
        agent,
        seed=req.seed,
        partner_mode=1 if req.partnerMode == 1 else 0,
        deterministic=req.deterministic,
        max_steps=req.maxSteps,
        sampler=sampler,
        device=device,
        inference_turn=inference_turn_in_thread,
    )

    trace = [build_action_detail(decision, agent) for decision in traced.decisions]
    memory_observes = [
        AnalyzeMemoryObserve(
            afterStepIndex=o.after_step_index,
            trick=o.trick,
            seat=o.seat,
            seatName=SEAT_NAMES[o.seat - 1],
            memoryCosineDistance=o.memory_cosine_distance,
            memoryNorm=o.memory_norm,
        )
        for o in traced.observes
    ]

    _compute_discounted_returns(trace, traced.transitions, traced.game, agent)
    compute_oracle_values(
        agent, traced.oracle_events, traced.oracle_decision_pos, trace, device
    )
    calibration = _build_calibration_summary(trace, traced.game, list(SEAT_NAMES))

    return _assemble_response(
        req, settings, agent, calibration, trace, memory_observes, traced.final
    )
