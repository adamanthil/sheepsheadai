"""One hand's AI review as the hand page is served it.

The stored review (persistence.review_reads) says what the model made of
each decision. This joins it to the recorded hand for what the page shows
around that: the seats, the deal, and at each decision the bookkeeping a
player keeps in their head, which is worked out from the record rather
than stored (the high trump, and whether the partner had been shown).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

from server.services.hand_review import HandRecord
from server.services.high_trump import high_trump, played_before
from server.services.persistence import history as history_db
from server.services.persistence import review_reads
from server.services.persistence.hand_records import load_hands
from server.services.review_bands import is_notable
from sheepshead.game import PARTNER_BY_CALLED_ACE

_SUMMARY_FIGURES = (
    "decisions",
    "agreed",
    "even",
    "slight",
    "costly",
    "better",
    "cost",
    "gained",
    "headline_index",
)


def partner_card(record: HandRecord) -> Optional[str]:
    """The card whose play shows the table who the picker's partner is: the
    called ace, or the jack of diamonds where that names the partner. None
    where there is no partner to show: a leaster, or a picker gone alone."""
    if record.picker is None or record.alone:
        return None
    if record.partner_mode == PARTNER_BY_CALLED_ACE:
        return record.called_card
    return "JD"


def _decision(stored: Dict[str, Any], record: HandRecord, seat: int) -> dict:
    """A stored decision, with what its seat could see of the hand then."""
    table = [(s, card) for s, card in stored["table_cards"]]
    played = played_before(record.tricks, stored["trick_index"], table)
    # A picker's own bury is gone for good once the cards are down.
    own_bury = record.bury if record.picker == seat else []
    trump = high_trump(
        stored["hand"], played, own_bury if stored["kind"] == "play" else []
    )
    return {
        "index": stored["index"],
        "kind": stored["kind"],
        "trick": stored["trick_index"],
        "hand": stored["hand"],
        "table": table,
        "action": stored["action"],
        "ai_action": stored["model_action"],
        "policy": stored["policy"],
        "standing": stored["standing"],
        "expected_score": stored["expected_score"],
        "oracle_score": stored["oracle_score"],
        "chosen_score": stored["chosen_score"],
        "ai_score": stored["model_score"],
        "difference": stored["difference"],
        "high_trump": {
            "card": trump.card,
            "held": trump.held,
            "unseen": trump.unseen,
        },
        "partner_known": partner_card(record) in played,
    }


def _summary(row: asyncpg.Record) -> dict:
    summary = {key: row[key] for key in _SUMMARY_FIGURES}
    summary["notable"] = is_notable(
        row["cost"], row["gained"], row["costly"], row["better"]
    )
    return summary


async def hand_review_view(
    pool: asyncpg.Pool, player_id: UUID, game_id: UUID
) -> Optional[dict]:
    """``player_id``'s hand ``game_id`` as HandReviewResponse; None if it is
    not one of their finished hands. A hand whose review is not done (still
    queued, skipped, failed) comes back with no summary and no decisions."""
    hand = await history_db.hand_row(pool, player_id, game_id)
    if hand is None:
        return None
    stored = await review_reads.hand_review(pool, player_id, game_id)
    seats = await review_reads.hand_seats(pool, game_id)
    seat = next(s["position"] for s in seats if s["player_id"] == player_id)
    newer, older = await history_db.neighbours(pool, player_id, game_id)
    records = await load_hands(pool, [game_id])
    record = records[0].record if records else None

    row = stored["row"] if stored is not None else None
    reviewed = row is not None and row["status"] == "reviewed"
    decisions: List[dict] = []
    if stored is not None and reviewed and record is not None:
        decisions = [_decision(d, record, seat) for d in stored["decisions"]]

    return {
        "hand": hand,
        "seat": seat,
        "called_card": record.called_card if record is not None else None,
        "called_under": record is not None and record.under_card is not None,
        "partner_card": partner_card(record) if record is not None else None,
        "dealt_hand": record.hands[seat] if record is not None else [],
        "blind": record.blind if record is not None else [],
        "partner_mode": (
            record.partner_mode if record is not None else PARTNER_BY_CALLED_ACE
        ),
        "seats": [
            {
                "seat": s["position"],
                "is_ai": s["player_id"] is None,
                "is_you": s["player_id"] == player_id,
                "name": s["name"] if s["player_id"] is not None else None,
                "role": s["role"],
            }
            for s in seats
        ],
        "summary": _summary(row) if row is not None and reviewed else None,
        "decisions": decisions,
        "ai_model": row["ai_model"] if row is not None else None,
        "newer_game_id": newer,
        "older_game_id": older,
    }
