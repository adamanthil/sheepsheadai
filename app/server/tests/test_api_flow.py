"""End-to-end API flow against a real Postgres.

Opt-in: set TEST_DATABASE_URL to a migrated database. CI points this at its
postgres service; locally:

    docker exec sheepshead_postgres psql -U sheepshead -c "CREATE DATABASE sheepshead_test;"
    (cd db && DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5433/sheepshead_test npx graphile-migrate migrate)
    TEST_DATABASE_URL=postgres://sheepshead:sheepshead@localhost:5433/sheepshead_test uv run pytest server/tests/test_api_flow.py
"""

from __future__ import annotations

import os

import httpx
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


class StubAgent:
    """Deterministic stand-in for PPOAgent: always the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass

    def reset_recurrent_state(self):
        pass


@pytest.fixture
async def db_app(app, monkeypatch):
    """The hermetic app fixture, but with a live pool wired to TEST_DATABASE_URL
    (httpx's ASGITransport does not run the lifespan, so do its DB work here)."""
    import server.api.games as games_module
    import server.app as app_module
    from server.services.persistence.pool import (
        close_pool,
        open_pool,
        set_db_state,
    )

    monkeypatch.setattr(games_module, "load_agent", lambda path: StubAgent())

    pool = await open_pool(TEST_DB)
    async with pool.acquire() as conn:
        ai_model_id = await app_module._upsert_ai_model(conn, "test-model")
        ai_player_id = await app_module._upsert_ai_player(conn, ai_model_id)
    set_db_state(pool, ai_player_id)
    try:
        yield app, pool
    finally:
        # Cancel background tasks owned by tables created in this test.
        from server.runtime.tables import tables

        for table in list(tables.tables.values()):
            for task in (table.ai_task, table.autoclose_task):
                if task and not task.done():
                    task.cancel()
        tables.tables.clear()
        await close_pool()


async def test_create_join_start_flow(db_app):
    app, pool = db_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/tables", json={"name": "flow"})
        assert created.status_code == 200
        table_id = created.json()["id"]

        joined = await client.post(
            f"/api/tables/{table_id}/join", json={"display_name": "Flo"}
        )
        assert joined.status_code == 200
        j = joined.json()
        token, client_id, player_id = (
            j["session_token"],
            j["client_id"],
            j["player_id"],
        )
        assert token and j["is_host"]
        auth = {"Authorization": f"Bearer {token}"}

        # The minted identity is real DB state.
        assert await pool.fetchval(
            "SELECT count(*) FROM session s JOIN player p USING (player_id) "
            "WHERE p.player_id = $1",
            __import__("uuid").UUID(player_id),
        ) == 1

        # Re-joining with the token is idempotent: same player, same
        # connection — never a second seat for one player.
        rejoined = await client.post(
            f"/api/tables/{table_id}/join",
            json={"display_name": "Flo2"},
            headers=auth,
        )
        assert rejoined.json()["player_id"] == player_id
        assert rejoined.json()["client_id"] == client_id
        assert rejoined.json()["session_token"] is None

        started = await client.post(
            f"/api/tables/{table_id}/start",
            json={"client_id": client_id},
            headers=auth,
        )
        assert started.status_code == 200, started.text
        assert started.json()["status"] == "playing"

        # The started hand is persisted: one game with five seats.
        game_row = await pool.fetchrow(
            "SELECT game_id FROM game WHERE game_table_id = $1", table_id
        )
        assert game_row is not None
        n_players = await pool.fetchval(
            "SELECT count(*) FROM game_player WHERE game_id = $1",
            game_row["game_id"],
        )
        assert n_players == 5

        # Double-start is refused (status re-checked under the state lock).
        again = await client.post(
            f"/api/tables/{table_id}/start",
            json={"client_id": client_id},
            headers=auth,
        )
        assert again.status_code == 400
        assert again.json()["detail"] == "already_started"


async def test_expired_session_is_rejected(db_app):
    app, pool = db_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/tables", json={"name": "exp"})
        table_id = created.json()["id"]
        joined = await client.post(
            f"/api/tables/{table_id}/join", json={"display_name": "Old"}
        )
        token = joined.json()["session_token"]
        client_id = joined.json()["client_id"]

        await pool.execute(
            "UPDATE session SET expires_at = now() - interval '1 day' "
            "WHERE token_hash = $1",
            __import__("hashlib").sha256(token.encode()).hexdigest(),
        )
        # Bypass the in-process auth cache so the DB expiry is consulted.
        from server.api import auth

        auth.clear_cache()

        resp = await client.post(
            f"/api/tables/{table_id}/start",
            json={"client_id": client_id},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
        assert resp.json()["detail"] == "invalid_token"
