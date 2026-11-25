"""AI review: a player's decisions against the AI's, read back from storage.

Everything here is a query over reviews the background worker has already
made (runtime.review_worker); no endpoint runs the model.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from server.api.auth import PlayerIdentity, current_player, verified_account
from server.api.schemas import (
    AccountReviewResponse,
    HandReviewResponse,
    ReviewQueueResponse,
)
from server.config import get_settings
from server.runtime import review_worker
from server.services.hand_review import is_notable
from server.services.high_trump import high_trump, played_before
from server.services.persistence import accounts as accounts_db
from server.services.persistence import history as history_db
from server.services.persistence import review_reads
from server.services.persistence.hand_records import load_hands
from server.services.persistence.pool import get_db_pool
from server.services.persistence.review_reads import decode

router = APIRouter()


@router.get("/api/account/review", response_model=AccountReviewResponse)
async def account_review(identity: PlayerIdentity = Depends(current_player)):
    """The caller's reviewed hands, totalled: how often they made the AI's
    choice, by kind of decision, and the largest differences."""
    pool = get_db_pool()
    await verified_account(identity)
    return await review_reads.account_review(pool, identity.id)


@router.get("/api/account/hands/{game_id}/review", response_model=HandReviewResponse)
async def hand_review(
    game_id: UUID, identity: PlayerIdentity = Depends(current_player)
):
    """One of the caller's hands, decision by decision, against the AI."""
    pool = get_db_pool()
    await verified_account(identity)
    hand = await history_db.hand_row(pool, identity.id, game_id)
    if hand is None:
        raise HTTPException(status_code=404, detail="hand_not_found")
    stored = await review_reads.hand_review(pool, identity.id, game_id)
    seats = await review_reads.hand_seats(pool, game_id)
    seat = next(s["position"] for s in seats if s["player_id"] == identity.id)
    newer, older = await history_db.neighbours(pool, identity.id, game_id)
    records = await load_hands(pool, [game_id])
    record = records[0].record if records else None

    decisions = []
    reviewed = stored is not None and stored["row"]["status"] == "reviewed"
    if stored is not None and reviewed and record is not None:
        own_bury = record.bury if record.picker == seat else []
        # The card whose play shows the table who the partner is: the
        # called ace, or the jack of diamonds where that names the partner.
        partner_card = record.called_card if record.partner_mode else "JD"
        for d in stored["decisions"]:
            table = [(s, card) for s, card in decode(d["table_cards"])]
            # A picker's own bury is gone for good once the cards are down.
            buried = own_bury if d["kind"] == "play" else []
            played = played_before(record.tricks, d["trick_index"], table)
            trump = high_trump(d["hand"], played, buried)
            decisions.append(
                {
                    "index": d["index"],
                    "kind": d["kind"],
                    "trick": d["trick_index"],
                    "hand": d["hand"],
                    "table": table,
                    "action": d["action"],
                    "ai_action": d["model_action"],
                    "policy": decode(d["policy"]),
                    "standing": d["standing"],
                    "expected_score": d["expected_score"],
                    "oracle_score": d["oracle_score"],
                    "chosen_score": d["chosen_score"],
                    "ai_score": d["model_score"],
                    "difference": d["difference"],
                    "high_trump": {
                        "card": trump.card,
                        "held": trump.held,
                        "unseen": trump.unseen,
                    },
                    "partner_known": partner_card in played,
                }
            )

    row = stored["row"] if stored is not None else None
    return {
        "hand": hand,
        "seat": seat,
        "called_card": record.called_card if record is not None else None,
        "called_under": record is not None and record.under_card is not None,
        "dealt_hand": record.hands[seat] if record is not None else [],
        "blind": record.blind if record is not None else [],
        "partner_mode": record.partner_mode if record is not None else 1,
        "seats": [
            {
                "seat": s["position"],
                "is_ai": s["player_id"] is None,
                "is_you": s["player_id"] == identity.id,
                "name": s["name"] if s["player_id"] is not None else None,
                "role": s["role"],
            }
            for s in seats
        ],
        "summary": (
            {
                key: row[key]
                for key in (
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
            }
            | {
                "notable": is_notable(
                    row["cost"], row["gained"], row["costly"], row["better"]
                )
            }
            if row is not None and reviewed
            else None
        ),
        "decisions": decisions,
        "ai_model": row["ai_model"] if row is not None else None,
        "newer_game_id": newer,
        "older_game_id": older,
    }


@router.get("/api/admin/review-queue", response_model=ReviewQueueResponse)
async def review_queue(identity: PlayerIdentity = Depends(current_player)):
    """The state of the AI review queue, for administrators."""
    pool = get_db_pool()
    account = await accounts_db.get_account(pool, identity.id)
    # Not found, rather than forbidden: the admin area is not advertised.
    if account is None or not account.is_admin:
        raise HTTPException(status_code=404, detail="not_found")
    label = get_settings().sheepshead_model_label
    stats = await review_reads.queue_stats(pool, label)
    return {
        **stats,
        "paused": review_worker.is_paused(),
        "tables_in_play": review_worker.tables_in_play(),
        "ai_model": label,
    }
