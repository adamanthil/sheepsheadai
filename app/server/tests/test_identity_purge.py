"""Expired sessions and orphaned anonymous players are purged; players with
recorded hands, live sessions, or brand-new rows are kept.

Opt-in like test_api_flow: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import os
import uuid

import pytest

from server.services.persistence.pool import open_pool
from server.services.persistence.sessions import purge_expired_identities

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


async def _player(conn, *, age: str) -> uuid.UUID:
    pid = uuid.uuid4()
    await conn.execute(
        "INSERT INTO player (player_id, name, time_created, last_updated) "
        f"VALUES ($1, NULL, now() - interval '{age}', now())",
        pid,
    )
    return pid


async def _session(conn, pid: uuid.UUID, *, expires_in: str) -> None:
    await conn.execute(
        "INSERT INTO session (player_id, token_hash, time_created, last_seen, "
        f"expires_at) VALUES ($1, $2, now(), now(), now() + interval '{expires_in}')",
        pid,
        uuid.uuid4().hex,
    )


async def _hand_for(conn, pid: uuid.UUID) -> None:
    table_id, game_id = uuid.uuid4(), uuid.uuid4()
    cardset = await conn.fetchval(
        "INSERT INTO cardset (cards_hash) VALUES ($1) RETURNING cardset_id",
        uuid.uuid4().hex,
    )
    await conn.execute(
        "INSERT INTO game_table (game_table_id, name, time_created) "
        "VALUES ($1, 'purge', now())",
        table_id,
    )
    await conn.execute(
        "INSERT INTO game (game_id, game_table_id, is_double_on_the_bump, "
        "is_called_partner, time_created, blind_id) "
        "VALUES ($1, $2, true, true, now(), $3)",
        game_id,
        table_id,
        cardset,
    )
    await conn.execute(
        "INSERT INTO game_player (game_id, player_id, name, position, "
        "starting_hand_id) VALUES ($1, $2, 'p', 1, $3)",
        game_id,
        pid,
        cardset,
    )


async def test_purge_keeps_live_played_and_new_players():
    pool = await open_pool(TEST_DB)
    try:
        async with pool.acquire() as conn:
            gone = await _player(conn, age="2 days")
            await _session(conn, gone, expires_in="-1 day")
            live = await _player(conn, age="2 days")
            await _session(conn, live, expires_in="1 day")
            played = await _player(conn, age="2 days")
            await _session(conn, played, expires_in="-1 day")
            await _hand_for(conn, played)
            fresh = await _player(conn, age="1 minute")

        sessions, players = await purge_expired_identities(pool)

        async with pool.acquire() as conn:
            left = {
                r["player_id"]
                for r in await conn.fetch(
                    "SELECT player_id FROM player WHERE player_id = ANY($1)",
                    [gone, live, played, fresh],
                )
            }
            played_sessions = await conn.fetchval(
                "SELECT count(*) FROM session WHERE player_id = $1", played
            )
        assert left == {live, played, fresh}
        assert played_sessions == 0
        assert sessions >= 2 and players >= 1
    finally:
        await pool.close()
