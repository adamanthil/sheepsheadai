"""AI review: a player's decisions against the AI's, read back from storage.

Everything here is a query over reviews the background worker has already
made (runtime.review_worker); no endpoint runs the model.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from server.api.auth import (
    PlayerIdentity,
    admin_account,
    current_player,
    verified_account,
)
from server.api.schemas import (
    AccountReviewResponse,
    HandReviewResponse,
    ReviewQueueResponse,
)
from server.config import get_settings
from server.runtime import review_worker
from server.services.persistence import review_reads
from server.services.persistence.pool import get_db_pool
from server.services.review_view import hand_review_view

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
    await verified_account(identity)
    view = await hand_review_view(get_db_pool(), identity.id, game_id)
    if view is None:
        raise HTTPException(status_code=404, detail="hand_not_found")
    return view


@router.get("/api/admin/review-queue", response_model=ReviewQueueResponse)
async def review_queue(identity: PlayerIdentity = Depends(current_player)):
    """The state of the AI review queue, for administrators."""
    await admin_account(identity)
    pool = get_db_pool()
    label = get_settings().sheepshead_model_label
    stats = await review_reads.queue_stats(pool, label)
    return {
        **stats,
        "paused": review_worker.is_paused(),
        "tables_in_play": review_worker.tables_in_play(),
        "ai_model": label,
    }
