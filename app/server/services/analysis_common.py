"""Shared inference helpers for the analysis endpoints.

Everything here is per-decision plumbing used by both the full-game
simulation (services.analyze) and the pick-scenario analysis
(services.pick_analysis): seeding, the encoder/actor/critic forward pass
with aux-head extraction, and assembling the valid-action probability
list.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from server.api.schemas import AnalyzeActionDetail, AnalyzeProbability
from sheepshead import ACTION_LOOKUP, TRUMP
from sheepshead.training.reward_shaping import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def build_probability_list(
    action_probs: torch.Tensor,
    logits: torch.Tensor,
    valid_actions: List[int],
) -> List[AnalyzeProbability]:
    """Per-valid-action probabilities and logits, sorted descending."""
    action_probs_np = action_probs.squeeze().cpu().numpy()
    logits_np = logits.squeeze().cpu().numpy()
    probabilities = [
        AnalyzeProbability(
            actionId=valid_action_id,
            action=ACTION_LOOKUP[valid_action_id],
            prob=float(action_probs_np[valid_action_id - 1]),
            logit=float(logits_np[valid_action_id - 1]),
        )
        for valid_action_id in valid_actions
    ]
    probabilities.sort(key=lambda x: x.prob, reverse=True)
    return probabilities


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
            vals = agent.oracle_critic.forward_sequences([events], device=device)
            values_by_seat[seat] = [float(v) for v in vals[0].cpu().tolist()]
    for action_detail, (seat, idx) in zip(trace, oracle_decision_pos):
        action_detail.oracleValue = values_by_seat[seat][idx]


@dataclass
class StepInference:
    """Encoder/actor/critic forward pass results and aux-head extraction for
    one acting player's turn."""

    state: Dict[str, Any]
    valid_actions: List[int]
    encoder_out: Dict[str, Any]
    action_probs: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor
    win_prob_val: Optional[float]
    expected_final_val: Optional[float]
    secret_partner_prob: Optional[float]
    trump_seen_mask: List[Dict[str, Any]]
    point_estimates: List[Dict[str, Any]]
    point_actuals: List[Dict[str, Any]]
    unseen_trump_higher_than_hand_prob: Optional[float]
    unseen_trump_higher_than_hand_actual: Optional[bool]
    memory_cosine_distance: Optional[float]
    memory_norm: float


def run_inference_step(
    agent: Any,
    actor_player: Any,
    actor_seat: int,
    players: List[str],
    device: torch.device,
) -> StepInference:
    """Run the encoder/actor/critic forward pass for the current actor and
    extract the auxiliary-head predictions (seen-trump mask, known-points
    estimate, win probability, etc)."""
    state = actor_player.get_state_dict()
    valid_actions = actor_player.get_valid_action_ids()

    # Get or init memory for this player
    memory_in = agent.get_recurrent_memory(actor_player.position, device=device)

    # Encode dict state with memory
    encoder_out = agent.encoder.encode_batch(
        [state], memory_in=memory_in.unsqueeze(0), device=device
    )

    # Store updated memory
    memory_out = encoder_out["memory_out"][0]
    agent.set_recurrent_memory(actor_player.position, memory_out)

    # Memory drift across this encode. A zero memory_in is the seat's first
    # encode: there is no previous belief state to drift from, so leave None.
    memory_norm = float(memory_out.detach().norm().item())
    if bool(torch.any(memory_in != 0)):
        memory_cosine_distance = float(
            1.0
            - torch.nn.functional.cosine_similarity(
                memory_in.detach().flatten(), memory_out.detach().flatten(), dim=0
            ).item()
        )
    else:
        memory_cosine_distance = None

    with torch.no_grad():
        # Build mask and hand ids for actor
        action_mask_t = (
            agent.get_action_mask(valid_actions, agent.action_size)
            .unsqueeze(0)
            .to(device)
        )
        hand_ids_t = torch.as_tensor(
            state["hand_ids"], dtype=torch.long, device=device
        ).view(1, -1)

        # Use existing encoder_out for logits and probabilities
        action_probs, logits = agent.actor.forward_with_logits(
            encoder_out,
            action_mask_t,
            hand_ids_t,
            agent.encoder.card,
        )

        value = agent.critic(encoder_out)

        # Auxiliary critic heads via accessor. No-aux architecture variants
        # (e.g. "no-aux", "perceiver") have no adapter or aux heads at all,
        # so every aux-derived field stays None for them.
        win_prob_val: Optional[float] = None
        expected_final_val: Optional[float] = None
        secret_partner_prob: Optional[float] = None
        point_vector: Optional[List[float]] = None
        seen_trump_mask_probs: Optional[List[float]] = None
        unseen_trump_higher_than_hand_prob: Optional[float] = None
        if agent.critic.has_aux_heads:
            win_prob_val, expected_final_val, secret_partner_prob, point_vector = (
                agent.critic.aux_predictions(encoder_out)
            )
            # _aux_features_single is the critic's overridable seam for aux
            # features: adapter(features) on the pooled critics, but a token
            # readout on perceiver critics — reading critic_adapter(features)
            # directly would feed those archs the vestigial memory vector.
            aux_feat = agent.critic._aux_features_single(encoder_out)
            seen_trump_mask_logits = agent.critic.seen_trump_mask_logits(
                aux_feat, agent.encoder.card
            ).squeeze(0)
            seen_trump_mask_probs = (
                torch.sigmoid(seen_trump_mask_logits).detach().cpu().tolist()
            )
            unseen_trump_higher_than_hand_logit = (
                agent.critic.unseen_trump_higher_than_hand_logits(aux_feat).squeeze(0)
            )
            unseen_trump_higher_than_hand_prob = float(
                torch.sigmoid(unseen_trump_higher_than_hand_logit).item()
            )

    trump_seen_mask: List[Dict[str, Any]] = []
    unseen_trump_higher_than_hand_actual: Optional[bool] = None
    if seen_trump_mask_probs is not None:
        seen_trump_mask_actual = compute_seen_trump_mask(actor_player)
        unseen_trump_higher_than_hand_actual = bool(
            compute_any_unseen_trump_higher_than_hand(actor_player)
        )
        trump_seen_mask = [
            {
                "card": TRUMP[i],
                "probabilitySeen": float(seen_trump_mask_probs[i]),
                "actualSeen": bool(seen_trump_mask_actual[i]),
            }
            for i in range(len(TRUMP))
        ]

    point_estimates: List[Dict[str, Any]] = []
    if point_vector:
        for rel_idx, rel_val in enumerate(point_vector, start=1):
            abs_seat = ((actor_seat + rel_idx - 2) % 5) + 1
            seat_label = (
                players[abs_seat - 1]
                if 0 < abs_seat <= len(players)
                else f"Seat {abs_seat}"
            )
            point_estimates.append(
                {
                    "seat": abs_seat,
                    "seatName": seat_label,
                    "points": rel_val,
                    "relativePosition": rel_idx,
                }
            )

    # Actuals exist only to be compared against the aux-head estimates, so
    # they are omitted whenever the estimates are.
    point_actuals: List[Dict[str, Any]] = []
    known_points_rel = (
        compute_known_points_rel(actor_player) if agent.critic.has_aux_heads else None
    )
    if known_points_rel:
        for rel_idx, rel_val in enumerate(known_points_rel, start=1):
            abs_seat = ((actor_seat + rel_idx - 2) % 5) + 1
            seat_label = (
                players[abs_seat - 1]
                if 0 < abs_seat <= len(players)
                else f"Seat {abs_seat}"
            )
            point_actuals.append(
                {
                    "seat": abs_seat,
                    "seatName": seat_label,
                    "points": rel_val,
                    "relativePosition": rel_idx,
                }
            )

    return StepInference(
        state=state,
        valid_actions=valid_actions,
        encoder_out=encoder_out,
        action_probs=action_probs,
        logits=logits,
        value=value,
        win_prob_val=win_prob_val,
        expected_final_val=expected_final_val,
        secret_partner_prob=secret_partner_prob,
        trump_seen_mask=trump_seen_mask,
        point_estimates=point_estimates,
        point_actuals=point_actuals,
        unseen_trump_higher_than_hand_prob=unseen_trump_higher_than_hand_prob,
        unseen_trump_higher_than_hand_actual=unseen_trump_higher_than_hand_actual,
        memory_cosine_distance=memory_cosine_distance,
        memory_norm=memory_norm,
    )
