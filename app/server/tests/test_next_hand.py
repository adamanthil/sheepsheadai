"""Next hand: the host passes the deal on and deals, in one step."""

from __future__ import annotations

import uuid

import httpx2
import pytest
from fastapi import Request

import server.api.games as games
import server.runtime.dealing as dealing
from server.api.auth import PlayerIdentity, current_player
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Occupant, Table
from server.runtime.occupants import give_seat_to_ai


class _Socket:
    """A connected tab that accepts whatever the table broadcasts."""

    async def send_text(self, text: str) -> None:
        pass


def _join(table: Table, cid: str, seat: int) -> ClientConn:
    conn = ClientConn(
        client_id=cid, display_name=cid, seat=seat, player_id=str(uuid.uuid4())
    )
    table.seats[seat] = cid
    table.clients[cid] = conn
    return conn


def _table(status: str) -> Table:
    """Host in seat 5, a guest in seat 2, the AI in the rest."""
    table = Table(id="t", name="next")
    table.host_client_id = _join(table, "host", seat=5).client_id
    _join(table, "guest", seat=2)
    for seat in (1, 3, 4):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name="AI", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    if status != "open":
        table.game = dealing.new_game_for_table(table)
        table.ever_dealt = True
    table.status = status
    tables.tables[table.id] = table
    return table


@pytest.fixture
async def client(app, monkeypatch):
    """Requests as the host, with the DB and the AI turn loop stubbed out."""

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(games, "get_db_pool", lambda: object())
    monkeypatch.setattr(games, "ensure_game_table", noop)
    monkeypatch.setattr(games, "persist_started_game", noop)
    monkeypatch.setattr(games, "schedule_ai_turns", lambda *a, **k: None)
    monkeypatch.setattr(dealing, "build_table_agent", lambda s: object())

    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


async def _next_hand(client, table: Table) -> httpx2.Response:
    host = table.clients["host"]
    return await client.post(
        "/api/tables/t/next-hand",
        json={"client_id": host.client_id},
        headers={"x-test-player": host.player_id or ""},
    )


async def test_next_hand_passes_the_deal_and_deals(client):
    table = _table("finished")
    finished = table.game
    table.doublings = 2

    r = await _next_hand(client, table)

    assert r.status_code == 200, r.text
    assert r.json()["status"] == "playing"
    # Everyone moves down a seat, seat 1's into seat 5.
    assert [table.seats[i] for i in range(1, 6)] == [
        "guest",
        "ai3",
        "ai4",
        "host",
        "ai1",
    ]
    assert table.clients["host"].seat == 4
    assert table.clients["guest"].seat == 1
    assert table.game is not None and table.game is not finished
    assert table.status == "playing"
    assert table.score_multiplier == 1


async def test_a_refused_next_hand_leaves_the_finished_hand_alone(client):
    table = _table("finished")
    finished = table.game
    seats = dict(table.seats)
    # A player left between hands and nobody has taken the seat.
    table.seats[3] = None

    r = await _next_hand(client, table)

    assert (r.status_code, r.json()["detail"]) == (400, "not_enough_players")
    assert table.seats == {**seats, 3: None}
    assert table.clients["host"].seat == 5
    assert table.game is finished and table.status == "finished"


async def test_host_cannot_deal_over_a_hand_in_play(client):
    # Mid-hand this would discard the deal unrecorded, erasing a hand the
    # host was losing. It is also what a double-submitted Next hand meets.
    table = _table("playing")
    game = table.game

    r = await _next_hand(client, table)

    assert (r.status_code, r.json()["detail"]) == (409, "hand_in_progress")
    assert table.game is game and table.status == "playing"
    assert table.clients["host"].seat == 5


async def test_next_hand_needs_a_hand_to_follow(client):
    table = _table("open")

    r = await _next_hand(client, table)

    assert (r.status_code, r.json()["detail"]) == (400, "no_hand_finished")
    assert table.game is None and table.status == "open"


async def test_a_host_moved_out_by_the_turn_clock_is_sat_back_down(client):
    table = _table("finished")
    host = table.clients["host"]
    host.sockets.add(_Socket())  # type: ignore[arg-type]  # connected
    give_seat_to_ai(table, host)
    host.home_occupant = table.seats[5]
    host.timeout_strikes = 3

    r = await _next_hand(client, table)

    assert r.status_code == 200, r.text
    # Back in seat 5, then down a seat with everyone else.
    assert table.seats[4] == "host" and host.seat == 4
    assert host.home_occupant is None and host.timeout_strikes == 0
    assert "host" not in table.disconnect_tasks


async def test_a_host_whose_phone_slept_is_sat_back_down(client):
    # Their socket dropped on the hand summary and the AI took the seat;
    # Next hand arrives before the reconnect that would reclaim it.
    table = _table("finished")
    host = table.clients["host"]
    table.occupants["stand-in"] = Occupant(id="stand-in", display_name="AI", is_ai=True)
    table.reserved_ai_by_human["host"] = "stand-in"
    table.seats[5] = "stand-in"
    host.seat = None

    r = await _next_hand(client, table)

    assert r.status_code == 200, r.text
    assert table.seats[4] == "host" and host.seat == 4
    # Still away: the AI takes over again unless they reconnect in time.
    assert "host" in table.disconnect_tasks
    table.disconnect_tasks["host"].cancel()


async def test_a_host_without_a_free_seat_deals_and_watches(client):
    table = _table("finished")
    host = table.clients["host"]
    host.sockets.add(_Socket())  # type: ignore[arg-type]
    give_seat_to_ai(table, host)
    # Other players have since taken every AI seat.
    for seat in (1, 3, 4, 5):
        _join(table, f"p{seat}", seat=seat)

    r = await _next_hand(client, table)

    assert r.status_code == 200, r.text
    assert host.seat is None and "host" not in table.seats.values()
