"""Backstop caps on tables and websocket connections."""

from __future__ import annotations

import uuid

import httpx
from starlette.testclient import TestClient

import server.realtime.websocket as ws_module
import server.runtime.manager as manager_module
import server.runtime.tables as tables_module
from server.api.auth import PlayerIdentity


async def test_table_cap_returns_503(app, monkeypatch):
    # Patch the module the manager actually reads, not the tables facade.
    monkeypatch.setattr(manager_module, "MAX_TABLES", 2)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/api/tables", json={"name": "a"})).status_code == 200
        assert (await client.post("/api/tables", json={"name": "b"})).status_code == 200
        resp = await client.post("/api/tables", json={"name": "c"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "table_limit_reached"


def _table_with_client(player_id: uuid.UUID) -> tables_module.Table:
    table = tables_module.Table(id="t1", name="capped")
    table.clients["c1"] = tables_module.ClientConn(
        client_id="c1", display_name="x", player_id=str(player_id)
    )
    tables_module.tables.tables["t1"] = table
    return table


def test_ws_per_ip_cap_closes_4429(app, monkeypatch):
    monkeypatch.setattr(ws_module, "MAX_SOCKETS_PER_IP", 0)
    player_id = uuid.uuid4()

    async def fake_resolve(token):
        return PlayerIdentity(id=player_id)

    monkeypatch.setattr(ws_module, "resolve_player", fake_resolve)
    _table_with_client(player_id)

    client = TestClient(app)
    with client.websocket_connect(
        "/ws/table/t1",
        subprotocols=["sheepshead.client.c1", "sheepshead.token.tok"],
    ) as ws:
        msg = ws.receive()
    assert msg["type"] == "websocket.close"
    assert msg["code"] == 4429


def test_ws_without_token_closes_4401(app):
    _table_with_client(uuid.uuid4())
    client = TestClient(app)
    with client.websocket_connect(
        "/ws/table/t1", subprotocols=["sheepshead.client.c1"]
    ) as ws:
        msg = ws.receive()
    assert msg["type"] == "websocket.close"
    assert msg["code"] == 4401


def test_ws_with_foreign_token_closes_4403(app, monkeypatch):
    async def fake_resolve(token):
        return PlayerIdentity(id=uuid.uuid4())  # not the table client's player

    monkeypatch.setattr(ws_module, "resolve_player", fake_resolve)
    _table_with_client(uuid.uuid4())

    client = TestClient(app)
    with client.websocket_connect(
        "/ws/table/t1",
        subprotocols=["sheepshead.client.c1", "sheepshead.token.tok"],
    ) as ws:
        msg = ws.receive()
    assert msg["type"] == "websocket.close"
    assert msg["code"] == 4403
