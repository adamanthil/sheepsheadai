"""A client that hangs up before the websocket handshake finishes.

uvicorn raises ClientDisconnected when the server sends the accept to a
peer that already left. That is routine (a stale tab, a backgrounded
phone) and must neither escape the handler nor leave bookkeeping behind.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from uvicorn.protocols.utils import ClientDisconnected

import server.realtime.websocket as ws_module
import server.runtime.manager as manager_module
from server.api.auth import PlayerIdentity
from server.runtime.models import ClientConn, Table

_SUBPROTOCOLS = ["sheepshead.client.c1", "sheepshead.token.tok"]


class _GoneSocket:
    """A websocket whose peer disconnected before the accept went out."""

    def __init__(self) -> None:
        self.scope = {"subprotocols": _SUBPROTOCOLS}
        self.client = SimpleNamespace(host="203.0.113.5")

    async def accept(self, subprotocol: str | None = None) -> None:
        raise ClientDisconnected()

    async def close(self, code: int = 1000) -> None:
        raise ClientDisconnected()


def _identity(monkeypatch) -> uuid.UUID:
    player_id = uuid.uuid4()

    async def fake_resolve(token):
        return PlayerIdentity(id=player_id)

    async def fake_touch(pool, pid, address):
        return None

    monkeypatch.setattr(ws_module, "resolve_player", fake_resolve)
    monkeypatch.setattr(ws_module, "get_db_pool", lambda: None)
    monkeypatch.setattr(ws_module.players_db, "touch_player_ip", fake_touch)
    return player_id


async def test_rejection_to_departed_client_is_quiet(app, monkeypatch):
    _identity(monkeypatch)
    # No such table: the handler accepts only to close with 4404.
    await ws_module.table_ws(_GoneSocket(), "missing")  # type: ignore[arg-type]


async def test_departure_during_accept_leaves_no_trace(app, monkeypatch):
    player_id = _identity(monkeypatch)
    table = Table(id="t1", name="t")
    conn = ClientConn(client_id="c1", display_name="x", player_id=str(player_id))
    table.clients["c1"] = conn
    manager_module.tables.tables["t1"] = table

    await ws_module.table_ws(_GoneSocket(), "t1")  # type: ignore[arg-type]

    assert not conn.sockets
    assert not ws_module._sockets_by_ip
