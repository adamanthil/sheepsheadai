"""Open tables are capped per creating player and per creating IP."""

from __future__ import annotations

import time
import uuid

import httpx

from server.api import auth
from server.config import get_settings
from server.runtime.tables import tables
from server.services.persistence.sessions import hash_token


async def _create(client: httpx.AsyncClient, headers: dict | None = None):
    return await client.post("/api/tables", json={"name": "t"}, headers=headers)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


async def test_per_ip_cap(app):
    async with _client(app) as client:
        for _ in range(get_settings().sheepshead_max_tables_per_ip):
            assert (await _create(client)).status_code == 200
        over = await _create(client)

    assert over.status_code == 429
    assert over.json()["detail"] == "table_limit_per_ip"


async def test_per_player_cap_and_closing_frees_a_slot(app):
    player = uuid.uuid4()
    auth._cache[hash_token("tok")] = (player, time.monotonic())
    headers = {"Authorization": "Bearer tok"}
    async with _client(app) as client:
        created = []
        for _ in range(get_settings().sheepshead_max_tables_per_player):
            r = await _create(client, headers)
            assert r.status_code == 200
            created.append(r.json()["id"])
        over = await _create(client, headers)
        assert over.status_code == 429
        assert over.json()["detail"] == "table_limit_per_player"

        tables.delete_table(created[0])
        assert (await _create(client, headers)).status_code == 200
