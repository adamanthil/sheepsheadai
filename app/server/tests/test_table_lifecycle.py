"""Table teardown: the idle autoclose must actually remove the table.

Every autoclose trigger is edge-driven off a websocket connect/disconnect, so
a table that survives its own close is unreachable by all of them and lives
until the process restarts. These tests pin the removal itself, not just the
"finished" status.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from fastapi import WebSocketDisconnect
from starlette.testclient import TestClient

import server.realtime.websocket as ws_module
import server.runtime.lifecycle as lifecycle
from server.api.auth import PlayerIdentity
from server.realtime.broadcast import broadcast_table_update
from server.runtime.tables import ClientConn, Occupant, Table, tables


@pytest.fixture
def registry():
    """Isolated table registry (module-level state shared across tests)."""
    tables.tables.clear()
    yield tables
    for table in tables.tables.values():
        for task in (table.ai_task, table.autoclose_task):
            if task and not task.done():
                task.cancel()
    tables.tables.clear()


@pytest.fixture
def stub_db(monkeypatch):
    """Stand in for the Postgres close stamp with a genuinely suspending call.

    asyncpg's acquire/execute both yield to the event loop; a non-suspending
    stub would hide any cancellation delivered at that await.
    """
    calls: list[str] = []

    async def fake_close_game_table(pool, table_id):
        await asyncio.sleep(0)
        calls.append(table_id)

    monkeypatch.setattr(lifecycle, "close_game_table", fake_close_game_table)
    monkeypatch.setattr(lifecycle, "get_db_pool", lambda: object())
    return calls


async def test_autoclose_removes_table_from_registry(registry, stub_db):
    table = Table(id="idle", name="idle")
    registry.tables[table.id] = table

    lifecycle.schedule_autoclose_if_no_humans(table, delay_seconds=0.01)
    await table.autoclose_task

    assert table.id not in registry.tables
    assert table.status == "finished"
    assert stub_db == ["idle"]


async def test_autoclose_suppressed_while_a_human_is_connected(registry, stub_db):
    table = Table(id="busy", name="busy")
    conn = ClientConn(client_id="c1", display_name="human", player_id="p1")
    conn.sockets.add(object())  # only the set's emptiness is inspected
    table.clients["c1"] = conn
    registry.tables[table.id] = table

    lifecycle.schedule_autoclose_if_no_humans(table, delay_seconds=0.01)
    await asyncio.sleep(0.05)

    assert table.id in registry.tables
    assert table.autoclose_task is None or table.autoclose_task.cancelled()


async def test_close_table_removes_table_even_when_db_stamp_raises(
    registry, monkeypatch
):
    async def boom(pool, table_id):
        raise RuntimeError("pool is not initialised")

    monkeypatch.setattr(lifecycle, "close_game_table", boom)
    monkeypatch.setattr(lifecycle, "get_db_pool", lambda: object())

    table = Table(id="dbdown", name="dbdown")
    registry.tables[table.id] = table

    with pytest.raises(RuntimeError):
        await lifecycle.close_table(table)

    assert table.id not in registry.tables


async def test_close_table_from_a_request_still_cancels_the_autoclose(
    registry, stub_db
):
    table = Table(id="host", name="host")
    registry.tables[table.id] = table
    lifecycle.schedule_autoclose_if_no_humans(table, delay_seconds=60.0)
    pending = table.autoclose_task

    await lifecycle.close_table(table, reason="host_closed")

    assert table.id not in registry.tables
    assert pending.cancelled() or pending.cancelling()


class _FakeWebSocket:
    """Minimal WebSocket stand-in whose first send fails.

    Models a client that vanishes during the post-accept handshake burst,
    before the receive loop is ever entered.
    """

    def __init__(self):
        self.closed = False

    async def accept(self, subprotocol=None):
        pass

    async def send_text(self, text):
        raise RuntimeError("client went away mid-handshake")

    async def receive_text(self):
        raise AssertionError("receive loop should never be reached")

    async def close(self, code=1000):
        self.closed = True


async def test_failed_handshake_send_does_not_leave_a_phantom_connection(registry):
    table = Table(id="phantom", name="phantom")
    conn = ClientConn(client_id="c1", display_name="ghost", player_id="p1")
    table.clients["c1"] = conn
    registry.tables[table.id] = table

    with pytest.raises(RuntimeError):
        await ws_module._serve_connection(_FakeWebSocket(), table, "c1", None)

    # A lingering socket here reads as a connected human to
    # any_human_connected(), which would suppress the idle autoclose forever.
    assert not conn.sockets
    assert conn.disconnected_at is not None
    assert table.autoclose_task is not None and not table.autoclose_task.done()


class _ControllableWebSocket:
    """WebSocket stand-in that stays open until ``disconnect`` is set."""

    def __init__(self):
        self.sent: list[str] = []
        self.disconnect = asyncio.Event()
        self.close_code: int | None = None

    async def accept(self, subprotocol=None):
        pass

    async def send_text(self, text):
        self.sent.append(text)

    async def receive_text(self):
        await self.disconnect.wait()
        raise WebSocketDisconnect(1000)

    async def close(self, code=1000):
        self.close_code = code
        self.disconnect.set()

    async def settled(self):
        """Resolve once the handshake burst is done and the receive loop is live."""
        while not self.sent:
            await asyncio.sleep(0)


async def _open_tab(table, client_id="c1"):
    ws = _ControllableWebSocket()
    task = asyncio.create_task(ws_module._serve_connection(ws, table, client_id, None))
    await ws.settled()
    return ws, task


def _table_with_client(table_id: str) -> tuple[Table, ClientConn]:
    table = Table(id=table_id, name=table_id)
    conn = ClientConn(client_id="c1", display_name="two-tabs", player_id="p1")
    table.clients["c1"] = conn
    tables.tables[table_id] = table
    return table, conn


async def test_tabs_of_one_player_are_all_live(registry):
    table, conn = _table_with_client("tabs")
    first, first_task = await _open_tab(table)
    second, second_task = await _open_tab(table)

    assert conn.sockets == {first, second}
    before = (len(first.sent), len(second.sent))
    await broadcast_table_update(table)
    # Both tabs receive the broadcast, not just the most recent one.
    assert len(first.sent) == before[0] + 1
    assert len(second.sent) == before[1] + 1

    first.disconnect.set()
    await first_task

    # Closing one tab must not mark the player disconnected, hand their seat
    # to an AI, or arm the idle autoclose.
    assert conn.sockets == {second}
    assert conn.connected
    assert conn.disconnected_at is None
    assert table.autoclose_task is None

    second.disconnect.set()
    await second_task

    assert not conn.sockets
    assert conn.disconnected_at is not None
    assert table.autoclose_task is not None


async def test_seat_reclaim_runs_only_on_the_first_tab(registry):
    table, conn = _table_with_client("reclaim")
    table.occupants["ai-1"] = Occupant(id="ai-1", display_name="Dan", is_ai=True)
    table.seats[3] = "ai-1"
    table.reserved_ai_by_human["c1"] = "ai-1"

    first, first_task = await _open_tab(table)
    assert conn.seat == 3
    assert table.seats[3] == "c1"

    # A second tab must not replay the reclaim and re-announce it to the table.
    table.seats[3] = "ai-1"
    second, second_task = await _open_tab(table)
    assert table.seats[3] == "ai-1"

    for ws, task in ((first, first_task), (second, second_task)):
        ws.disconnect.set()
        await task


async def test_socket_cap_refuses_the_extra_tab(registry, monkeypatch):
    monkeypatch.setattr(ws_module, "MAX_SOCKETS_PER_CLIENT", 2)
    table, conn = _table_with_client("capped")

    opened = [await _open_tab(table) for _ in range(2)]
    assert len(conn.sockets) == 2

    extra = _ControllableWebSocket()
    await ws_module._serve_connection(extra, table, "c1", None)

    assert extra.close_code == 4429
    assert extra not in conn.sockets
    assert len(conn.sockets) == 2

    for ws, task in opened:
        ws.disconnect.set()
        await task


def test_two_real_sockets_share_one_client(app, monkeypatch):
    """End-to-end through the ASGI stack, not the _ControllableWebSocket fake."""
    player_id = uuid.uuid4()

    async def fake_resolve(token):
        return PlayerIdentity(id=player_id)

    monkeypatch.setattr(ws_module, "resolve_player", fake_resolve)
    table, conn = _table_with_client("real")
    conn.player_id = str(player_id)

    client = TestClient(app)
    subs = ["sheepshead.client.c1", "sheepshead.token.tok"]
    with client.websocket_connect("/ws/table/real", subprotocols=subs) as first:
        assert first.receive_json()["type"] == "chat:init"
        assert first.receive_json()["type"] == "table_update"
        with client.websocket_connect("/ws/table/real", subprotocols=subs) as second:
            assert second.receive_json()["type"] == "chat:init"
            assert second.receive_json()["type"] == "table_update"
            assert len(conn.sockets) == 2
            second.send_text(json.dumps({"type": "chat:send", "message": "hi"}))
            # The chat append fans out to both tabs of the same player.
            assert first.receive_json()["type"] == "chat:append"
            assert second.receive_json()["type"] == "chat:append"
        assert conn.connected
