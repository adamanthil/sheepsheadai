"""Leaving, or closing the table, once the hand in play ends."""

from __future__ import annotations

import asyncio
import json
import uuid

import httpx2
import pytest
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime.dealing import new_game_for_table
from server.runtime.departures import LEFT_AFTER_HAND_WS_CODE
from server.runtime.lifecycle import finish_hand
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Occupant, Table


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
    conn.sockets.add(_RecordingSocket())  # type: ignore[arg-type]
    if seat is not None:
        table.seats[seat] = cid
    table.clients[cid] = conn
    return conn


def _table_in_play() -> Table:
    table = Table(id="t", name="after hand")
    table.host_client_id = _join(table, "host", seat=5).client_id
    _join(table, "guest", seat=2)
    for seat in (1, 3, 4):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name="AI", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    table.game = new_game_for_table(table)
    table.status = "playing"
    tables.tables[table.id] = table
    return table


def _socket(conn: ClientConn) -> _RecordingSocket:
    return next(iter(conn.sockets))  # type: ignore[return-value]


@pytest.fixture
async def client(app):
    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()
        for table in tables.tables.values():
            if table.ai_task:
                table.ai_task.cancel()


def _as(conn: ClientConn) -> dict:
    return {"x-test-player": conn.player_id or ""}


async def test_guest_leaves_when_the_hand_ends(client):
    table = _table_in_play()
    guest = table.clients["guest"]
    ws = _socket(guest)

    r = await client.post(
        "/api/tables/t/leave_after_hand",
        json={"client_id": "guest"},
        headers=_as(guest),
    )
    assert r.status_code == 200, r.text
    assert table.to_public_dict()["seatLeavingAfterHand"][2] is True
    # Nothing happens until the hand is over.
    assert table.seats[2] == "guest"

    await finish_hand(table)

    assert "guest" not in table.clients
    assert table.occupants[table.seats[2] or ""].is_ai
    left = next(m for m in ws.sent if m["type"] == "left_after_hand")
    assert len(left["table"]["resultsHistory"]) == 1
    assert ws.closed_with == LEFT_AFTER_HAND_WS_CODE
    assert table.results_history[-1]["bySeat"][2]["id"] == "guest"
    last = table.chat_log[-1]
    assert (last["author"], last["body"]) == (
        "guest",
        "left after the hand. Seat 2 went to the AI.",
    )


async def test_request_can_be_taken_back(client):
    table = _table_in_play()
    guest = table.clients["guest"]
    for on in (True, False):
        r = await client.post(
            "/api/tables/t/leave_after_hand",
            json={"client_id": "guest", "on": on},
            headers=_as(guest),
        )
        assert r.status_code == 200, r.text

    await finish_hand(table)

    assert table.seats[2] == "guest"
    assert _socket(guest).closed_with is None


async def test_no_leave_after_hand_between_hands(client):
    table = _table_in_play()
    table.status = "finished"

    r = await client.post(
        "/api/tables/t/leave_after_hand",
        json={"client_id": "guest"},
        headers=_as(table.clients["guest"]),
    )

    assert (r.status_code, r.json()["detail"]) == (409, "no_hand_in_play")


async def test_leaving_host_hands_off_to_a_staying_player(client):
    table = _table_in_play()
    _join(table, "also_leaving", seat=None)
    table.clients["host"].leave_after_hand = True
    # Joined before guest, but leaving too, so passed over.
    table.clients = {
        cid: table.clients[cid] for cid in ("host", "also_leaving", "guest")
    }
    table.clients["also_leaving"].leave_after_hand = True

    await finish_hand(table)

    assert table.host_client_id == "guest"
    assert set(table.clients) == {"guest"}


async def test_leaving_host_with_no_one_to_take_over(client):
    table = _table_in_play()
    del table.clients["guest"]
    table.seats[2] = "ai3"
    table.clients["host"].leave_after_hand = True

    await finish_hand(table)

    assert table.clients == {}
    assert table.occupants[table.seats[5] or ""].is_ai


async def test_host_closes_the_table_after_the_hand(client):
    table = _table_in_play()
    host = table.clients["host"]
    guest_ws = _socket(table.clients["guest"])

    r = await client.post(
        "/api/tables/t/close_after_hand",
        json={"client_id": "host"},
        headers=_as(host),
    )
    assert r.status_code == 200, r.text
    assert table.to_public_dict()["closingAfterHand"] is True

    await finish_hand(table)
    for _ in range(20):
        if table.id not in tables.tables:
            break
        await asyncio.sleep(0)

    assert table.id not in tables.tables
    closed = {
        "type": "table_closed",
        "reason": "host_closed_after_hand",
        "tableId": "t",
    }
    assert closed in guest_ws.sent


async def test_only_the_host_can_close_after_the_hand(client):
    table = _table_in_play()

    r = await client.post(
        "/api/tables/t/close_after_hand",
        json={"client_id": "guest"},
        headers=_as(table.clients["guest"]),
    )

    assert r.status_code == 403
    assert table.close_after_hand is False
