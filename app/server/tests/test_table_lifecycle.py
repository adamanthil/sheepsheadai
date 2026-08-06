"""Table teardown: the idle autoclose must actually remove the table.

Every autoclose trigger is edge-driven off a websocket connect/disconnect, so
a table that survives its own close is unreachable by all of them and lives
until the process restarts. These tests pin the removal itself, not just the
"finished" status.
"""

from __future__ import annotations

import asyncio

import pytest

import server.realtime.websocket as ws_module
import server.runtime.lifecycle as lifecycle
from server.runtime.tables import ClientConn, Table, tables


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
    conn.websocket = object()  # only its non-None-ness is inspected
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

    # A lingering websocket here reads as a connected human to
    # any_human_connected(), which would suppress the idle autoclose forever.
    assert conn.websocket is None
    assert conn.disconnected_at is not None
    assert table.autoclose_task is not None and not table.autoclose_task.done()
