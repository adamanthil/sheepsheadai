"""game_table row bookkeeping (created on first start, stamped on close)."""

from __future__ import annotations

import logging
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)


async def ensure_game_table(pool: asyncpg.Pool, table_id: str, table_name: str) -> None:
    """Create game_table row on first start; no-op if the row already exists."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO game_table (game_table_id, name, time_created, time_closed)
                VALUES ($1, $2, now(), NULL)
                ON CONFLICT (game_table_id) DO NOTHING
                """,
                UUID(table_id),
                table_name,
            )
    except Exception:
        logger.exception(
            "ensure_game_table failed (table=%s)",
            table_id,
            extra={"table_id": table_id},
        )


async def close_game_table(pool: asyncpg.Pool, table_id: str) -> None:
    """Stamp time_closed on the game_table row when the table is closed."""
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE game_table SET time_closed = now() WHERE game_table_id = $1",
                UUID(table_id),
            )
    except Exception:
        logger.exception(
            "close_game_table failed (table=%s)",
            table_id,
            extra={"table_id": table_id},
        )
