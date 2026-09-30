"""Personal stats (verified accounts) and the public leaderboard."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player, optional_player
from server.api.schemas import AccountStatsResponse, LeaderboardResponse
from server.services.persistence import accounts as accounts_db
from server.services.persistence import stats as stats_db
from server.services.persistence.pool import get_db_pool
from server.services.persistence.stats import (
    LEADERBOARD_MIN_HANDS,
    LEADERBOARD_SIZE,
    SortKey,
)

router = APIRouter()


def _public_row(row: dict, you: bool) -> dict:
    return {
        "rank": row["rank"],
        "username": row["username"],
        "hands": row["hands"],
        "total": row["total"],
        "sph": row["sph"],
        "win_pct": row["win_pct"],
        "pick_pct": row["pick_pct"],
        "is_you": you,
    }


@router.get("/api/leaderboard", response_model=LeaderboardResponse)
async def leaderboard(request: Request, sort: SortKey = "total"):
    # Always best-first and a single page: the bottom of the board is never
    # served, so nobody can sort the weakest players to the top.
    ranking = await stats_db.ranking(get_db_pool(), sort)
    identity = await optional_player(request)
    player_id = identity.id if identity is not None else None
    mine = stats_db.row_of(ranking, player_id)
    return {
        "sort": sort,
        "min_hands": LEADERBOARD_MIN_HANDS,
        "rows": [
            _public_row(r, r["player_id"] == player_id)
            for r in ranking[:LEADERBOARD_SIZE]
        ],
        "you": (
            _public_row(mine, True)
            if mine is not None and mine["rank"] > LEADERBOARD_SIZE
            else None
        ),
    }


@router.get("/api/account/stats", response_model=AccountStatsResponse)
async def account_stats(identity: PlayerIdentity = Depends(current_player)):
    pool = get_db_pool()
    account = await accounts_db.get_account(pool, identity.id)
    if account is None:
        raise HTTPException(status_code=403, detail="account_required")
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="email_unverified")
    stats = await stats_db.player_stats(pool, identity.id)
    mine = stats_db.row_of(await stats_db.ranking(pool, "total"), identity.id)
    qualifies_in: Optional[int] = None
    if mine is None:
        qualifies_in = max(0, LEADERBOARD_MIN_HANDS - stats["hands"])
    return {
        "username": account.username,
        **stats,
        "rank": mine["rank"] if mine is not None else None,
        "qualifies_in": qualifies_in,
        "min_hands": LEADERBOARD_MIN_HANDS,
    }
