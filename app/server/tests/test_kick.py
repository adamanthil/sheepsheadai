"""The host can remove a player: AI takes the seat, and they're banned."""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest
from fastapi import Request

from server.api import auth
from server.api import tables as tables_api
from server.api.auth import PlayerIdentity, current_player
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import ClientConn, Occupant, Table, tables
from server.services.persistence.sessions import hash_token


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed_with: int | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


def _join(table: Table, cid: str, seat: int | None) -> ClientConn:
    conn = ClientConn(
        client_id=cid, display_name=cid, seat=seat, player_id=str(uuid.uuid4())
    )
    if seat is not None:
        table.seats[seat] = cid
    table.clients[cid] = conn
    return conn


def _table_in_play() -> Table:
    table = Table(id="t", name="kick")
    host = _join(table, "host", seat=5)
    table.host_client_id = host.client_id
    _join(table, "pest", seat=2)
    for seat in (1, 3, 4):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name="AI", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    table.game = new_game_for_table(table)
    table.status = "playing"
    tables.tables[table.id] = table
    return table


@pytest.fixture
async def client(app):
    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        for table in tables.tables.values():
            if table.ai_task:
                table.ai_task.cancel()


def _as(conn: ClientConn) -> dict:
    return {"x-test-player": conn.player_id or ""}


async def test_host_removes_a_seated_player(client):
    table = _table_in_play()
    host, pest = table.clients["host"], table.clients["pest"]
    ws = _RecordingSocket()
    pest.sockets.add(ws)  # type: ignore[arg-type]

    r = await client.post(
        "/api/tables/t/kick",
        json={"client_id": "host", "target_client_id": "pest"},
        headers=_as(host),
    )

    assert r.status_code == 200, r.text
    assert "pest" not in table.clients
    assert table.occupants[table.seats[2] or ""].is_ai
    assert pest.player_id in table.banned_player_ids
    assert ws.sent == [{"type": "kicked"}]
    assert ws.closed_with == 4403
    last = table.chat_log[-1]
    assert (last["author"], last["body"]) == ("pest", "was removed by the host")


async def test_only_the_host_can_remove(client):
    table = _table_in_play()
    pest = table.clients["pest"]

    r = await client.post(
        "/api/tables/t/kick",
        json={"client_id": "pest", "target_client_id": "host"},
        headers=_as(pest),
    )

    assert r.status_code == 403
    assert "host" in table.clients


async def test_host_cannot_remove_themselves(client):
    table = _table_in_play()

    r = await client.post(
        "/api/tables/t/kick",
        json={"client_id": "host", "target_client_id": "host"},
        headers=_as(table.clients["host"]),
    )

    assert r.status_code == 400


async def test_removed_player_cannot_rejoin(client, monkeypatch):
    # The ban is checked before the join touches the database.
    monkeypatch.setattr(tables_api, "get_db_pool", lambda: object())
    table = _table_in_play()
    banned = str(uuid.uuid4())
    table.banned_player_ids.add(banned)
    # Their session token resolves from the auth cache, so no DB is needed.
    auth._cache[hash_token("banned-token")] = (uuid.UUID(banned), time.monotonic())

    r = await client.post(
        "/api/tables/t/join",
        json={"display_name": "back again"},
        headers={"Authorization": "Bearer banned-token"},
    )

    assert r.status_code == 403
    assert r.json()["detail"] == "removed_from_table"
