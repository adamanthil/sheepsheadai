"""Mutating routes demand a valid bearer token bound to the acting client."""

from __future__ import annotations

import uuid

import httpx

from server.api.auth import PlayerIdentity, current_player


async def test_mutating_routes_require_token(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cases = [
            client.post("/api/tables/x/seat", json={"client_id": "c", "seat": 1}),
            client.post("/api/tables/x/start", json={"client_id": "c"}),
            client.post("/api/tables/x/action", json={"client_id": "c", "action_id": 1}),
            client.post("/api/tables/x/close", json={"client_id": "c"}),
            client.patch(f"/api/players/{uuid.uuid4()}", json={"name": "n"}),
        ]
        for coro in cases:
            resp = await coro
            assert resp.status_code == 401, resp.url


async def test_players_patch_requires_ownership(app):
    app.dependency_overrides[current_player] = lambda: PlayerIdentity(id=uuid.uuid4())
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.patch(
                f"/api/players/{uuid.uuid4()}", json={"name": "n"}
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "not_your_player"
    finally:
        app.dependency_overrides.clear()


async def test_foreign_client_id_is_403(app):
    """Even with a valid token, acting through someone else's client_id fails."""
    from server.runtime.tables import ClientConn, Table, tables

    victim = uuid.uuid4()
    attacker = uuid.uuid4()
    table = Table(id="t1", name="t")
    table.clients["victim-conn"] = ClientConn(
        client_id="victim-conn", display_name="v", seat=1, player_id=str(victim)
    )
    table.host_client_id = "victim-conn"
    tables.tables["t1"] = table

    app.dependency_overrides[current_player] = lambda: PlayerIdentity(id=attacker)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/tables/t1/close", json={"client_id": "victim-conn"}
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "client_mismatch"
    finally:
        app.dependency_overrides.clear()
