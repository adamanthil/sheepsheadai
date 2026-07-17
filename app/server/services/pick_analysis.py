"""Pick/call/bury scenario analysis.

Builds a game where a chosen seat is guaranteed to face the pick/pass
decision with a chosen (or random) hand and blind, then follows the
policy through the pre-play phases (partner call, under placement, bury)
until play would start, recording the same per-decision detail as the
full-game simulation.

Forced predecessor passes are realized by setting ``game.last_passed``
(the mechanism Game.__init__ itself uses for ``picking_player``) instead
of stepping literal PASS actions: recurrent memory only changes via a
seat's own encodes and trick-completion observes, so the two are
behaviorally identical for the target seat.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from server.api.schemas import (
    AnalyzeActionDetail,
    AnalyzePickOutcome,
    AnalyzePickRequest,
    AnalyzePickResponse,
    AnalyzePickScenario,
)
from server.config import get_settings
from server.runtime.seating import ANALYZE_SEAT_NAMES
from server.services.ai_loader import load_agent
from server.services.analysis_common import (
    build_probability_list,
    infer_phase_from_action_id,
    run_inference_step,
    set_seed,
)
from sheepshead import ACTION_LOOKUP, DECK, Game

# Pre-play is at most 5 picks/passes + call + under + 2 buries; anything
# past this indicates a bug rather than a legal phase sequence.
_MAX_DECISIONS = 16


def _validate_cards(cards: List[str], what: str, max_count: int) -> None:
    if len(cards) > max_count:
        raise ValueError(f"{what} can contain at most {max_count} cards")
    unknown = [c for c in cards if c not in DECK]
    if unknown:
        raise ValueError(f"{what} contains unknown cards: {', '.join(unknown)}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"{what} contains duplicate cards")


def _apply_scenario(
    game: Game, seat: int, hand: Optional[List[str]], blind: Optional[List[str]]
) -> None:
    """Re-deal the game so `seat` holds `hand`, the blind is `blind`, and
    the pick decision is on `seat` (earlier seats have passed). Partial
    hand/blind selections are completed randomly from the remaining deck."""
    hand_fixed = list(hand or [])
    blind_fixed = list(blind or [])
    fixed = hand_fixed + blind_fixed
    pool = [c for c in DECK if c not in fixed]
    game.rng.shuffle(pool)

    def draw(n: int) -> List[str]:
        taken, pool[:] = pool[:n], pool[n:]
        return taken

    hand_cards = hand_fixed + draw(6 - len(hand_fixed))
    blind_cards = blind_fixed + draw(2 - len(blind_fixed))

    game.blind = blind_cards
    for player in game.players:
        cards = hand_cards if player.position == seat else draw(6)
        player.initial_hand = cards
        player.hand = cards[:]

    # Seats before the target have passed; the pick decision is on `seat`.
    game.last_passed = seat - 1


def analyze_pick(req: AnalyzePickRequest) -> AnalyzePickResponse:
    """Run the pre-play scenario and return every decision until play
    would begin (or the hand goes to leaster)."""
    if not 1 <= req.seat <= 5:
        raise ValueError("seat must be between 1 and 5")
    if req.hand:
        _validate_cards(req.hand, "hand", 6)
    if req.blind:
        _validate_cards(req.blind, "blind", 2)
    if req.hand and req.blind:
        overlap = set(req.hand) & set(req.blind)
        if overlap:
            raise ValueError(
                f"hand and blind overlap: {', '.join(sorted(overlap))}"
            )

    if req.seed is not None:
        set_seed(req.seed)

    settings = get_settings()
    agent = load_agent(settings.sheepshead_model_path)
    agent.reset_recurrent_state()

    partner_mode = 1 if req.partnerMode == 1 else 0
    game = Game(partner_selection_mode=partner_mode, seed=req.seed)
    _apply_scenario(game, req.seat, req.hand, req.blind)

    players = ANALYZE_SEAT_NAMES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario = AnalyzePickScenario(
        seat=req.seat,
        seatName=players[req.seat - 1],
        hand=sorted(
            game.players[req.seat - 1].hand, key=lambda card: DECK.index(card)
        ),
        blind=list(game.blind),
    )

    from server.runtime.tables import build_player_state

    decisions: List[AnalyzeActionDetail] = []
    while (
        not game.play_started
        and not game.is_leaster
        and not game.is_done()
        and len(decisions) < _MAX_DECISIONS
    ):
        actor_player = next(
            (p for p in game.players if p.get_valid_action_ids()), None
        )
        if actor_player is None:
            break
        actor_seat = actor_player.position

        inference = run_inference_step(agent, actor_player, actor_seat, players, device)

        if req.deterministic:
            action_id = torch.argmax(inference.action_probs, dim=1).item() + 1
        else:
            dist = torch.distributions.Categorical(inference.action_probs)
            action_id = dist.sample().item() + 1

        decisions.append(
            AnalyzeActionDetail(
                stepIndex=len(decisions),
                seat=actor_seat,
                seatName=players[actor_seat - 1],
                phase=infer_phase_from_action_id(action_id, agent.action_groups),
                actionId=action_id,
                action=ACTION_LOOKUP[action_id],
                valueEstimate=float(inference.value.item()),
                validActionIds=inference.valid_actions,
                probabilities=build_probability_list(
                    inference.action_probs, inference.logits, inference.valid_actions
                ),
                view=build_player_state(actor_player)["view"],
                winProb=inference.win_prob_val,
                expectedFinalReturn=inference.expected_final_val,
                secretPartnerProb=inference.secret_partner_prob,
                pointEstimates=inference.point_estimates or None,
                pointActuals=inference.point_actuals or None,
                trumpSeenMask=inference.trump_seen_mask or None,
                unseenTrumpHigherThanHandProb=inference.unseen_trump_higher_than_hand_prob,
                unseenTrumpHigherThanHandActual=inference.unseen_trump_higher_than_hand_actual,
                memoryCosineDistance=inference.memory_cosine_distance,
                memoryNorm=inference.memory_norm,
            )
        )

        actor_player.act(action_id)

    picker_seat = game.picker or None
    bury = list(game.bury) if picker_seat else []
    outcome = AnalyzePickOutcome(
        pickerSeat=picker_seat,
        pickerName=players[picker_seat - 1] if picker_seat else None,
        isLeaster=bool(game.is_leaster),
        aloneCalled=bool(game.alone_called),
        calledCard=game.called_card,
        calledUnder=bool(game.is_called_under),
        underCard=game.under_card,
        bury=bury,
    )

    return AnalyzePickResponse(
        meta={
            "partnerMode": req.partnerMode,
            "seat": req.seat,
            "seed": req.seed,
            "deterministic": req.deterministic,
            "model": settings.sheepshead_model_label,
        },
        scenario=scenario,
        decisions=decisions,
        outcome=outcome,
    )
