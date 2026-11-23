"""Personal stats (verified accounts) and the public leaderboard."""

from __future__ import annotations

from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player, optional_player
from server.api.schemas import (
    AccountRunningScoreResponse,
    AccountStatsResponse,
    HandHistoryResponse,
    LeaderboardResponse,
)
from server.services.persistence import accounts as accounts_db
from server.services.persistence import history as history_db
from server.services.persistence import stats as stats_db
from server.services.persistence.history import BadCursor
from server.services.persistence.pool import get_db_pool
from server.services.persistence.stats import (
    ABANDON_AI_ACTIONS,
    LEADERBOARD_SIZE,
    MODE_BOARD,
    SPLIT_MARGIN_MIN_HANDS,
    BoardKey,
    GameMode,
    PeriodKey,
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
async def leaderboard(
    request: Request,
    board: BoardKey = "overall",
    sort: SortKey = "total",
    period: PeriodKey = "all_time",
):
    # Always best-first and a single page: the bottom of the board is never
    # served, so nobody can sort the weakest players to the top.
    pool = get_db_pool()
    month = await stats_db.month_bounds(pool, period)
    ranking = await stats_db.ranking(pool, sort, board, month)
    identity = await optional_player(request)
    player_id = identity.id if identity is not None else None
    mine = stats_db.row_of(ranking, player_id)
    min_hands = stats_db.board_min_hands(board, monthly=month is not None)

    # How far the caller is from this board; a finished month can't be
    # qualified for any more.
    qualifies_in: Optional[int] = None
    if player_id is not None and mine is None and period != "last_month":
        hands = await stats_db.board_hands(pool, player_id, board, month)
        qualifies_in = max(0, min_hands - hands)

    return {
        "board": board,
        "sort": sort,
        "period": period,
        "period_start": month.start if month else None,
        "period_end": month.end if month else None,
        "min_hands": min_hands,
        "rows": [
            _public_row(r, r["player_id"] == player_id)
            for r in ranking[:LEADERBOARD_SIZE]
        ],
        "you": (
            _public_row(mine, True)
            if mine is not None and mine["rank"] > LEADERBOARD_SIZE
            else None
        ),
        "qualifies_in": qualifies_in,
    }


async def _verified_account(pool: asyncpg.Pool, identity: PlayerIdentity):
    account = await accounts_db.get_account(pool, identity.id)
    if account is None:
        raise HTTPException(status_code=403, detail="account_required")
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="email_unverified")
    return account


@router.get("/api/account/stats", response_model=AccountStatsResponse)
async def account_stats(
    mode: GameMode = "all",
    identity: PlayerIdentity = Depends(current_player),
):
    """The caller's stats over every finished hand, or one partner mode's;
    the rank is on that mode's all-time board, by score per hand (the
    leaderboard's default sort)."""
    pool = get_db_pool()
    account = await _verified_account(pool, identity)
    stats = await stats_db.player_stats(pool, identity.id, mode)
    board = MODE_BOARD[mode]
    ranking = await stats_db.ranking(pool, "sph", board)
    mine = stats_db.row_of(ranking, identity.id)
    min_hands = stats_db.board_min_hands(board)
    qualifies_in: Optional[int] = None
    if mine is None:
        qualifies_in = max(0, min_hands - stats["hands"])
    hands = stats["hands"]
    return {
        "username": account.username,
        **stats,
        "completion_rate": (1 - stats["abandoned_hands"] / hands if hands else None),
        "abandon_threshold": ABANDON_AI_ACTIONS,
        "rank": mine["rank"] if mine is not None else None,
        "qualifies_in": qualifies_in,
        "min_hands": min_hands,
        **await stats_db.player_splits(pool, identity.id, mode),
        "split_margin_min_hands": SPLIT_MARGIN_MIN_HANDS,
        **await stats_db.player_picking(pool, identity.id, mode),
        **await stats_db.player_pick_offers(pool, identity.id, mode),
        "roles": await stats_db.player_roles(pool, identity.id, mode),
    }


@router.get("/api/account/running-score", response_model=AccountRunningScoreResponse)
async def account_running_score(
    mode: GameMode = "all",
    identity: PlayerIdentity = Depends(current_player),
):
    """Every finished hand of the caller's (or one partner mode's), oldest
    first, for the running score chart."""
    pool = get_db_pool()
    await _verified_account(pool, identity)
    return await stats_db.player_running_score(pool, identity.id, mode)


@router.get("/api/account/hands", response_model=HandHistoryResponse)
async def account_hands(
    before: Optional[str] = None,
    identity: PlayerIdentity = Depends(current_player),
):
    """The caller's finished hands, newest first, a page at a time; pass
    the previous page's next_cursor as ``before`` for the next."""
    pool = get_db_pool()
    await _verified_account(pool, identity)
    try:
        return await history_db.hand_history(pool, identity.id, before)
    except BadCursor:
        raise HTTPException(status_code=400, detail="invalid_cursor")
