"""Idle tables close on schedule, even with players connected."""

from __future__ import annotations

import time

import pytest

import server.runtime.lifecycle as lifecycle
from server.runtime.lifecycle import (
    HAND_IDLE_SECONDS,
    LOBBY_EXPIRY_SECONDS,
    idle_tables,
    sweep_idle_tables,
)
from server.runtime.tables import ClientConn, Table, tables


class _Socket:
    async def send_text(self, text: str) -> None:
        pass

    async def close(self, code: int = 1000) -> None:
        pass


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    async def no_db(pool, table_id):
        pass

    monkeypatch.setattr(lifecycle, "close_game_table", no_db)
    monkeypatch.setattr(lifecycle, "get_db_pool", lambda: object())
    tables.tables.clear()
    yield
    tables.tables.clear()


def _table(tid: str, *, age: float, dealt: bool, finished_ago: float | None) -> Table:
    now = time.monotonic()
    table = Table(id=tid, name=tid, created_at=now - age, ever_dealt=dealt)
    if finished_ago is not None:
        table.hand_finished_at = now - finished_ago
    # A connected player doesn't keep an idle table open.
    conn = ClientConn(client_id=f"{tid}-c", display_name="p")
    conn.sockets.add(_Socket())  # type: ignore[arg-type]
    table.clients[conn.client_id] = conn
    tables.tables[tid] = table
    return table


def test_which_tables_are_idle():
    _table("fresh-lobby", age=60, dealt=False, finished_ago=None)
    _table("stale-lobby", age=LOBBY_EXPIRY_SECONDS + 1, dealt=False, finished_ago=None)
    _table("long-game", age=LOBBY_EXPIRY_SECONDS * 3, dealt=True, finished_ago=None)
    _table("just-finished", age=3600, dealt=True, finished_ago=30)
    _table("left-on-scores", age=3600, dealt=True, finished_ago=HAND_IDLE_SECONDS + 1)

    due = {t.id: reason for t, reason in idle_tables(time.monotonic())}

    assert due == {"stale-lobby": "lobby_expired", "left-on-scores": "hand_idle"}


async def test_sweep_closes_idle_tables():
    _table("stale-lobby", age=LOBBY_EXPIRY_SECONDS + 1, dealt=False, finished_ago=None)
    _table("fresh-lobby", age=60, dealt=False, finished_ago=None)

    await sweep_idle_tables()

    assert set(tables.tables) == {"fresh-lobby"}
