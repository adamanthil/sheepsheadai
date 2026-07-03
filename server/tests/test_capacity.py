"""Backstop caps on tables and websocket connections."""

from __future__ import annotations

import httpx
from starlette.testclient import TestClient

import server.realtime.websocket as ws_module
import server.runtime.tables as tables_module


async def test_table_cap_returns_503(app, monkeypatch):
    monkeypatch.setattr(tables_module, "MAX_TABLES", 2)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/api/tables", json={"name": "a"})).status_code == 200
        assert (await client.post("/api/tables", json={"name": "b"})).status_code == 200
        resp = await client.post("/api/tables", json={"name": "c"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "table_limit_reached"


def test_ws_per_ip_cap_closes_4429(app, monkeypatch):
    monkeypatch.setattr(ws_module, "MAX_SOCKETS_PER_IP", 0)
    # The cap check needs a joined client on a real table; fabricate both.
    table = tables_module.Table(id="t1", name="capped")
    table.clients["c1"] = tables_module.ClientConn(client_id="c1", display_name="x")
    tables_module.tables.tables["t1"] = table

    client = TestClient(app)
    with client.websocket_connect(
        "/ws/table/t1", subprotocols=["sheepshead.client.c1"]
    ) as ws:
        msg = ws.receive()
    assert msg["type"] == "websocket.close"
    assert msg["code"] == 4429
