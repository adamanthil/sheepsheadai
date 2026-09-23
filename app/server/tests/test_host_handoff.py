"""Host passes to the longest-present connected human after a grace."""

from __future__ import annotations

import asyncio

import pytest

from server.runtime import host as host_mod
from server.runtime.host import cancel_host_handoff, schedule_host_handoff
from server.runtime.tables import ClientConn, Table


class _Socket:
    async def send_text(self, text: str) -> None:
        pass


def _join(table: Table, cid: str, seat: int | None, connected: bool) -> ClientConn:
    conn = ClientConn(client_id=cid, display_name=cid, seat=seat)
    if connected:
        conn.sockets.add(_Socket())  # type: ignore[arg-type]
    if seat is not None:
        table.seats[seat] = cid
    table.clients[cid] = conn
    return conn


@pytest.fixture(autouse=True)
def no_grace(monkeypatch):
    monkeypatch.setattr(host_mod, "HOST_HANDOFF_GRACE_SECONDS", 0.0)


async def _run_handoff(table: Table) -> None:
    schedule_host_handoff(table)
    task = table.host_handoff_task
    assert task is not None
    await task


async def test_seated_human_preferred_over_earlier_spectator():
    table = Table(id="t", name="h")
    _join(table, "host", seat=5, connected=False)
    _join(table, "watcher", seat=None, connected=True)
    _join(table, "late", seat=3, connected=True)
    _join(table, "later", seat=1, connected=True)
    table.host_client_id = "host"

    await _run_handoff(table)

    assert table.host_client_id == "late"
    assert table.chat_log[-1]["body"] == "late is now the host"


async def test_spectator_takes_host_when_no_one_is_seated():
    table = Table(id="t", name="h")
    _join(table, "host", seat=5, connected=False)
    _join(table, "watcher", seat=None, connected=True)
    table.host_client_id = "host"

    await _run_handoff(table)

    assert table.host_client_id == "watcher"


async def test_host_back_within_grace_keeps_host(monkeypatch):
    monkeypatch.setattr(host_mod, "HOST_HANDOFF_GRACE_SECONDS", 60.0)
    table = Table(id="t", name="h")
    _join(table, "host", seat=5, connected=False)
    _join(table, "other", seat=1, connected=True)
    table.host_client_id = "host"

    schedule_host_handoff(table)
    task = table.host_handoff_task
    assert task is not None
    cancel_host_handoff(table)  # the host's tab reconnected
    await asyncio.gather(task, return_exceptions=True)

    assert table.host_client_id == "host"


async def test_no_one_connected_leaves_host_unchanged():
    table = Table(id="t", name="h")
    _join(table, "host", seat=5, connected=False)
    _join(table, "other", seat=1, connected=False)
    table.host_client_id = "host"

    await _run_handoff(table)

    assert table.host_client_id == "host"
    assert table.host_handoff_task is None
