"""Typing in the chat is relayed to everyone else at the table, throttled,
and stopped when the typist's message arrives or their last tab closes."""

from __future__ import annotations

import json

import pytest

from server.realtime import chat as chat_module
from server.realtime.chat import relay_typing
from server.runtime.models import ClientConn, Table


class _Socket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _table() -> tuple[Table, dict[str, _Socket]]:
    table = Table(id="t", name="typing")
    sockets = {}
    for cid, name in (("c1", "Kyle"), ("c2", "Ann"), ("c3", "Dan")):
        sock = _Socket()
        conn = ClientConn(client_id=cid, display_name=name)
        conn.sockets.add(sock)  # type: ignore[arg-type]
        table.clients[cid] = conn
        sockets[cid] = sock
    return table, sockets


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(chat_module.time, "monotonic", lambda: now[0])
    return now


async def test_typing_reaches_everyone_but_the_typist(clock):
    table, sockets = _table()
    await relay_typing(table, table.clients["c1"], True)
    expected = {
        "type": "chat:typing",
        "client_id": "c1",
        "name": "Kyle",
        "typing": True,
    }
    assert sockets["c2"].sent == [expected]
    assert sockets["c3"].sent == [expected]
    assert sockets["c1"].sent == []


async def test_repeats_are_throttled(clock):
    table, sockets = _table()
    kyle = table.clients["c1"]
    await relay_typing(table, kyle, True)
    clock[0] += 1.0
    await relay_typing(table, kyle, True)
    assert len(sockets["c2"].sent) == 1
    clock[0] += chat_module.TYPING_RELAY_INTERVAL_SECONDS
    await relay_typing(table, kyle, True)
    assert len(sockets["c2"].sent) == 2


async def test_stopped_is_relayed_only_after_typing(clock):
    table, sockets = _table()
    kyle = table.clients["c1"]
    await relay_typing(table, kyle, False)
    assert sockets["c2"].sent == []
    await relay_typing(table, kyle, True)
    await relay_typing(table, kyle, False)
    assert [m["typing"] for m in sockets["c2"].sent] == [True, False]
    # Typing again straight after stopping is not throttled.
    await relay_typing(table, kyle, True)
    assert [m["typing"] for m in sockets["c2"].sent] == [True, False, True]
