"""Response-schema plumbing shared by the analysis endpoints.

The full-game simulation (services.analyze) and the pick-scenario
analysis (services.pick_analysis) both run decisions through
sheepshead.agent.traced_game; this module turns those records into the
API's trace entries, and resolves the deployed sampling and the oracle
critic's values.
"""

from __future__ import annotations

import secrets
from typing import Any, Dict, List, Optional

import torch

from server.api.schemas import (
    AnalyzeActionDetail,
    AnalyzePointEstimate,
    AnalyzeProbability,
    AnalyzeTrumpSeenMaskEntry,
)
from server.services.ai_loader import inference_turn_in_thread
from server.services.ai_sampling import SamplingSpec
from sheepshead.agent.sampling import DeploySampling
from sheepshead.agent.traced_game import SEAT_NAMES, Decision
from sheepshead.game import TRUMP


def deployment_sampler(
    settings, seed: Optional[int]
) -> tuple[Optional[DeploySampling], str]:
    """The /analyze stochastic policy: the deployed sampling (settings;
    server.services.ai_sampling), reseeded from the request seed so a seeded
    run replays. While the deployment is still argmax (no tuned temperatures)
    the raw policy is sampled at T = 1, the pre-tuning behaviour, so
    "stochastic" keeps meaning something. Returns (sampler or None, label)."""
    spec = SamplingSpec.from_settings(settings)
    if spec.is_argmax:
        return None, "raw policy (T=1)"
    sampler = spec.sampler()
    assert sampler is not None
    sampler.reseed(seed if seed is not None else secrets.randbits(62))
    return sampler, spec.label()


def infer_phase_from_action_id(
    action_id: int, action_groups: Dict[str, List[int]]
) -> str:
    """Infer the game phase from action ID using PPOAgent action groups."""
    # Convert to 0-indexed
    zero_indexed = action_id - 1

    for phase, indices in action_groups.items():
        if zero_indexed in indices:
            return phase

    return "unknown"


def build_probability_list(decision: Decision) -> List[AnalyzeProbability]:
    """Per-valid-action probabilities and logits, sorted descending."""
    return [
        AnalyzeProbability(
            actionId=p.action_id, action=p.action, prob=p.prob, logit=p.logit
        )
        for p in decision.inference.probabilities()
    ]


def compute_oracle_values(
    agent: Any,
    oracle_events: Dict[int, List[Dict[str, Any]]],
    oracle_decision_pos: List[tuple[int, int]],
    trace: List[AnalyzeActionDetail],
    device: torch.device,
) -> None:
    """Attach privileged critic values to the trace (mutated in place).

    Mirrors PPOAgent._fill_oracle_values: each seat's full event stream
    (decisions + trick observes, chronological) goes through the recurrent
    oracle critic with fresh zero memory, and values are read off at the
    decision positions."""
    if not oracle_decision_pos:
        return
    values_by_seat: Dict[int, List[float]] = {}
    with torch.no_grad():
        for seat, events in oracle_events.items():
            with inference_turn_in_thread():
                vals = agent.oracle_critic.forward_sequences([events], device=device)
            values_by_seat[seat] = [float(v) for v in vals[0].cpu().tolist()]
    for action_detail, (seat, idx) in zip(trace, oracle_decision_pos):
        action_detail.oracleValue = values_by_seat[seat][idx]


def _point_entries(
    values_rel: List[float], actor_seat: int
) -> List[AnalyzePointEstimate]:
    """Convert a per-seat point vector in relative order (index 1 = the actor)
    into entries labeled with absolute seats."""
    entries: List[AnalyzePointEstimate] = []
    for rel_idx, points in enumerate(values_rel, start=1):
        abs_seat = ((actor_seat + rel_idx - 2) % 5) + 1
        entries.append(
            AnalyzePointEstimate(
                seat=abs_seat,
                seatName=SEAT_NAMES[abs_seat - 1],
                points=float(points),
                relativePosition=rel_idx,
            )
        )
    return entries


def _trump_seen_entries(decision: Decision) -> List[AnalyzeTrumpSeenMaskEntry]:
    inference = decision.inference
    if inference.seen_trump_mask_probs is None:
        return []
    assert inference.seen_trump_mask_actual is not None
    return [
        AnalyzeTrumpSeenMaskEntry(
            card=TRUMP[i],
            probabilitySeen=float(inference.seen_trump_mask_probs[i]),
            actualSeen=bool(inference.seen_trump_mask_actual[i]),
        )
        for i in range(len(TRUMP))
    ]


def build_action_detail(decision: Decision, agent: Any) -> AnalyzeActionDetail:
    """Assemble the per-decision trace entry shared by the full-game
    simulation and the pick-scenario analysis. Fields that only exist after
    a finished game (rewards, returns, oracle values) stay None; the callers
    that have them fill them in afterwards."""
    inference = decision.inference
    point_estimates = _point_entries(inference.point_estimates_rel or [], decision.seat)
    point_actuals = _point_entries(inference.known_points_rel or [], decision.seat)
    trump_seen_mask = _trump_seen_entries(decision)
    return AnalyzeActionDetail(
        stepIndex=decision.step_index,
        seat=decision.seat,
        seatName=decision.seat_name,
        phase=infer_phase_from_action_id(decision.action_id, agent.action_groups),
        actionId=decision.action_id,
        action=decision.action,
        valueEstimate=decision.value_estimate,
        validActionIds=list(inference.valid_actions),
        probabilities=build_probability_list(decision),
        view=decision.view,
        winProb=inference.win_prob,
        expectedFinalReturn=inference.expected_final,
        secretPartnerProb=inference.secret_partner_prob,
        pointEstimates=point_estimates or None,
        pointActuals=point_actuals or None,
        trumpSeenMask=trump_seen_mask or None,
        unseenTrumpHigherThanHandProb=inference.unseen_trump_higher_than_hand_prob,
        unseenTrumpHigherThanHandActual=inference.unseen_trump_higher_than_hand_actual,
        memoryCosineDistance=inference.memory_cosine_distance,
        memoryNorm=inference.memory_norm,
    )
