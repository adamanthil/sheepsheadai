"""Player persistence (Phase 4).

`player` rows are minted lazily on first `POST /api/tables/:id/join`. The
`name` column is NULL until the user explicitly chooses a display name via
`PATCH /api/players/:id`.
"""

from __future__ import annotations

from typing import Optional
from uuid import UUID

import asyncpg


async def get_player(pool: asyncpg.Pool, player_id: UUID) -> Optional[dict]:
    row = await pool.fetchrow(
        "SELECT player_id, name FROM player WHERE player_id = $1",
        player_id,
    )
    if row is None:
        return None
    return {"player_id": str(row["player_id"]), "name": row["name"]}


async def ensure_player(pool: asyncpg.Pool, player_id: UUID) -> None:
    """Idempotently insert a player row with NULL name.

    Used when a client presents a `player_id` the server has no record of —
    e.g. after a DB reset. Bumps `last_updated` only on first insert.
    """
    await pool.execute(
        """
        INSERT INTO player (player_id, name, time_created, last_updated)
        VALUES ($1, NULL, now(), now())
        ON CONFLICT (player_id) DO NOTHING
        """,
        player_id,
    )


async def create_player(pool: asyncpg.Pool, player_id: UUID) -> None:
    """Insert a freshly-minted player row."""
    await pool.execute(
        """
        INSERT INTO player (player_id, name, time_created, last_updated)
        VALUES ($1, NULL, now(), now())
        """,
        player_id,
    )


async def set_player_name(
    pool: asyncpg.Pool, player_id: UUID, name: Optional[str]
) -> Optional[dict]:
    row = await pool.fetchrow(
        """
        UPDATE player
        SET name = $2, last_updated = now()
        WHERE player_id = $1
        RETURNING player_id, name
        """,
        player_id,
        name,
    )
    if row is None:
        return None
    return {"player_id": str(row["player_id"]), "name": row["name"]}
