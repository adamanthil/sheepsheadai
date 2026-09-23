"""Unseated clients receive the public game view, never private cards."""

from __future__ import annotations

import json

from server.realtime.broadcast import broadcast_table_state
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import ClientConn, Occupant, Table


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _connect(table: Table, client_id: str, seat: int | None) -> _RecordingSocket:
    ws = _RecordingSocket()
    conn = ClientConn(client_id=client_id, display_name=client_id, seat=seat)
    conn.sockets.add(ws)  # type: ignore[arg-type]
    table.clients[client_id] = conn
    if seat is not None:
        table.seats[seat] = client_id
    return ws


def _table_into_play() -> Table:
    """A hand advanced past pick and bury, so the picker holds private
    blind and bury cards and one card has been led."""
    table = Table(id="t", name="spectate")
    for i in range(1, 6):
        table.occupants[f"ai{i}"] = Occupant(id=f"ai{i}", display_name="AI", is_ai=True)
        table.seats[i] = f"ai{i}"
    table.game = new_game_for_table(table)
    table.status = "playing"
    game = table.game
    while game.cards_played < 1:
        player = next(p for p in game.players if p.get_valid_action_ids())
        player.act(min(player.get_valid_action_ids()))
    return table


async def test_spectator_gets_public_view_only():
    table = _table_into_play()
    game = table.game
    assert game is not None and game.picker
    seated = _connect(table, "seated", seat=2)
    watcher = _connect(table, "watcher", seat=None)

    await broadcast_table_state(table)

    (own,) = seated.sent
    assert own["yourSeat"] == 2
    assert own["view"]["hand"]
    assert set(own["view"]["hand"]) == set(game.players[1].hand)

    (public,) = watcher.sent
    assert public["type"] == "state"
    assert public["yourSeat"] is None
    assert public["valid_actions"] == []
    view = public["view"]
    assert view["hand"] == view["blind"] == view["bury"] == []
    assert "player" not in view
    assert view["picker"] == game.picker
    # Every card the spectator can see has actually been played.
    played = {c for trick in game.history for c in trick if c}
    seen = {c for trick in view["history"] for c in trick if c}
    seen |= {c for c in view["current_trick"] if c}
    assert seen <= played
    # The observation dict is per-seat and stays private.
    assert set(public["state"]) == {"play_started"}
