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


async def ensure_player(
    pool: asyncpg.Pool, player_id: UUID, ip: Optional[str] = None
) -> None:
    """Idempotently insert a player row with NULL name, recording ``ip``.

    Used when a client presents a `player_id` the server has no record of —
    e.g. after a DB reset. Bumps `last_updated` only on first insert; an
    existing row is written only when its `last_ip` actually changes.
    """
    await pool.execute(
        """
        INSERT INTO player (player_id, name, last_ip, time_created, last_updated)
        VALUES ($1, NULL, $2::inet, now(), now())
        ON CONFLICT (player_id) DO UPDATE SET last_ip = EXCLUDED.last_ip
        WHERE EXCLUDED.last_ip IS NOT NULL
          AND player.last_ip IS DISTINCT FROM EXCLUDED.last_ip
        """,
        player_id,
        ip,
    )


async def touch_player_ip(pool: asyncpg.Pool, player_id: UUID, ip: str) -> None:
    """Record the address an existing player was last seen from."""
    await pool.execute(
        """
        UPDATE player SET last_ip = $2::inet
        WHERE player_id = $1 AND last_ip IS DISTINCT FROM $2::inet
        """,
        player_id,
        ip,
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
