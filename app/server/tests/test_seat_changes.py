"""Seat changes once a hand is dealt: AI takeover only, never a move."""

from __future__ import annotations

import uuid

import httpx
import pytest
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import ClientConn, Occupant, Table, tables


def _table_in_play() -> tuple[Table, uuid.UUID, uuid.UUID]:
    """A live hand: human "seated" at seat 2, AIs elsewhere, and an
    unseated "spectator" client at the table."""
    table = Table(id="t", name="seats")
    seated, spectator = uuid.uuid4(), uuid.uuid4()
    table.clients["seated"] = ClientConn(
        client_id="seated", display_name="s", seat=2, player_id=str(seated)
    )
    table.clients["spectator"] = ClientConn(
        client_id="spectator", display_name="v", player_id=str(spectator)
    )
    table.seats[2] = "seated"
    for i in (1, 3, 4, 5):
        table.occupants[f"ai{i}"] = Occupant(id=f"ai{i}", display_name="AI", is_ai=True)
        table.seats[i] = f"ai{i}"
    table.game = new_game_for_table(table)
    table.status = "playing"
    tables.tables["t"] = table
    return table, seated, spectator


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


async def test_seated_player_cannot_hop_seats_mid_hand(client):
    table, seated, _ = _table_in_play()

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "seated", "seat": 4},
        headers={"x-test-player": str(seated)},
    )

    assert r.status_code == 409
    assert r.json()["detail"] == "seat_locked_in_play"
    # Neither seat 4's hand exposed nor seat 2 stranded empty.
    assert table.seats[2] == "seated"
    assert table.seats[4] == "ai4"
    assert table.clients["seated"].seat == 2


async def test_unseated_client_can_take_over_ai_seat_mid_hand(client):
    table, _, spectator = _table_in_play()

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 4},
        headers={"x-test-player": str(spectator)},
    )

    assert r.status_code == 200
    assert table.seats[4] == "spectator"
    assert table.clients["spectator"].seat == 4
    # The displaced AI is held for the takeover like any other.
    assert table.reserved_ai_by_human["spectator"] == "ai4"


async def test_unseated_client_cannot_take_human_seat_mid_hand(client):
    table, _, spectator = _table_in_play()

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 2},
        headers={"x-test-player": str(spectator)},
    )

    assert r.status_code == 409
    assert table.seats[2] == "seated"
    assert table.clients["spectator"].seat is None


async def test_timed_out_player_may_only_take_back_their_own_seat(client):
    table, _, spectator = _table_in_play()
    # The turn timer moved them out of the seat ai4 now holds.
    table.clients["spectator"].home_occupant = "ai4"
    headers = {"x-test-player": str(spectator)}

    elsewhere = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 1},
        headers=headers,
    )
    assert elsewhere.status_code == 409
    assert elsewhere.json()["detail"] == "not_your_seat"

    home = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 4},
        headers=headers,
    )
    assert home.status_code == 200
    assert table.seats[4] == "spectator"
    assert table.clients["spectator"].home_occupant is None


async def test_timed_out_player_whose_seat_was_taken_may_take_any_ai_seat(client):
    table, _, spectator = _table_in_play()
    # Their seat's AI was since taken over by someone else.
    table.clients["spectator"].home_occupant = "ai-no-longer-seated"

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 1},
        headers={"x-test-player": str(spectator)},
    )

    assert r.status_code == 200
    assert table.seats[1] == "spectator"
