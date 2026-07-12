"""Persistence hooks 5-6: trick completion and hand finalization.

Each hook is its own transaction and logs-then-continues on failure so that
DB errors never break live play.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict
from uuid import UUID

import asyncpg

from sheepshead import DECK_IDS, UNDER_TOKEN

if TYPE_CHECKING:
    from server.runtime.tables import Table

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook 5: trick completed
# ---------------------------------------------------------------------------


async def persist_trick_completed(
    pool: asyncpg.Pool,
    table: "Table",
    post: Dict[str, Any],
    trick_idx: int,
) -> None:
    if not table.current_game_id:
        return
    try:
        leader = post["leaders"][trick_idx]
        winner = post["trick_winners"][trick_idx]
        points = post["trick_points"][trick_idx]
        lead_gp_id = table.game_player_ids[leader]
        winner_gp_id = table.game_player_ids[winner]
        trick_history = post["history"][trick_idx]
        under_card = post["under_card"]

        async with pool.acquire() as conn:
            async with conn.transaction():
                trick_id = await conn.fetchval(
                    """
                    INSERT INTO trick (game_id, index, lead_player_id, winning_player_id, points)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING trick_id
                    """,
                    UUID(table.current_game_id),
                    trick_idx,
                    lead_gp_id,
                    winner_gp_id,
                    points,
                )

                rows = []
                for k in range(5):
                    seat = (leader - 1 + k) % 5 + 1  # 1-based seat
                    history_idx = (leader - 1 + k) % 5  # 0-based history index
                    card_str = trick_history[history_idx]
                    if card_str == UNDER_TOKEN:
                        card_id = DECK_IDS[under_card]
                    else:
                        card_id = DECK_IDS[card_str]
                    gp_id = table.game_player_ids[seat]
                    rows.append((trick_id, card_id, gp_id, k))

                await conn.executemany(
                    """
                    INSERT INTO trick_card (trick_id, card_id, game_player_id, index)
                    VALUES ($1, $2, $3, $4)
                    """,
                    rows,
                )
    except Exception:
        logger.exception(
            "persist_trick_completed failed (table=%s game=%s trick=%s)",
            table.id,
            table.current_game_id,
            trick_idx,
            extra={
                "table_id": table.id,
                "game_id": table.current_game_id,
                "trick_idx": trick_idx,
            },
        )


# ---------------------------------------------------------------------------
# Hook 6: finalize game
# ---------------------------------------------------------------------------


async def persist_finalize_game(
    pool: asyncpg.Pool,
    table: "Table",
    scores: list,
) -> None:
    if not table.current_game_id:
        return
    game_id = table.current_game_id
    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "UPDATE game SET time_closed = now() WHERE game_id = $1",
                    UUID(game_id),
                )
                for seat in range(1, 6):
                    gp_id = table.game_player_ids[seat]
                    await conn.execute(
                        "UPDATE game_player SET score = $1 WHERE game_player_id = $2",
                        scores[seat - 1],
                        gp_id,
                    )
        # Clear so the next hand starts fresh.
        table.current_game_id = None
        table.game_player_ids = {}
    except Exception:
        logger.exception(
            "persist_finalize_game failed (table=%s game=%s)",
            table.id,
            game_id,
            extra={"table_id": table.id, "game_id": game_id},
        )
