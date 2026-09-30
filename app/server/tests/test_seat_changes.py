"""Seat changes once a hand is dealt: AI takeover only, never a move."""

from __future__ import annotations

import time
import uuid

import httpx
import pytest
from fastapi import Request

from server.api import auth
from server.api import tables as tables_api
from server.api.auth import PlayerIdentity, current_player
from server.runtime.dealing import new_game_for_table
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Occupant, Table
from server.services.persistence.sessions import hash_token


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
        for table in tables.tables.values():
            if table.ai_task:
                table.ai_task.cancel()


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


def _with_human_dealt_ai_seat(table: Table) -> None:
    """Seat 3 was dealt to "gone", who disconnected: its AI plays their
    row out and is held for their reclaim. Every other AI seat was dealt
    to the AI."""
    table.clients["gone"] = ClientConn(
        client_id="gone", display_name="g", player_id=str(uuid.uuid4())
    )
    table.reserved_ai_by_human["gone"] = "ai3"
    table.game_player_is_ai = {1: True, 2: False, 3: False, 4: True, 5: True}


async def _join_as_newcomer(client, monkeypatch) -> dict:
    async def no_db(*_args, **_kwargs):
        return None

    monkeypatch.setattr(tables_api.players_db, "ensure_player", no_db)
    monkeypatch.setattr(tables_api.accounts_db, "verified_username", no_db)
    monkeypatch.setattr(tables_api, "get_db_pool", lambda: object())
    auth._cache[hash_token("newcomer-token")] = (uuid.uuid4(), time.monotonic())
    r = await client.post(
        "/api/tables/t/join",
        json={"display_name": "late"},
        headers={"Authorization": "Bearer newcomer-token"},
    )
    assert r.status_code == 200
    return r.json()


async def test_public_table_marks_human_dealt_seats_untakeable():
    table, _, _ = _table_in_play()
    _with_human_dealt_ai_seat(table)

    takeable = table.to_public_dict()["seatTakeable"]

    assert takeable == {1: True, 2: False, 3: False, 4: True, 5: True}


async def test_mid_hand_joiner_skips_human_dealt_seat(client, monkeypatch):
    table, _, _ = _table_in_play()
    _with_human_dealt_ai_seat(table)
    table.game_player_is_ai[1] = False  # seat 1 is another absent human's

    joined = await _join_as_newcomer(client, monkeypatch)

    assert table.clients[joined["client_id"]].seat == 4
    assert table.seats[1] == "ai1" and table.seats[3] == "ai3"


async def test_mid_hand_joiner_watches_when_every_ai_seat_is_human_dealt(
    client, monkeypatch
):
    table, _, _ = _table_in_play()
    _with_human_dealt_ai_seat(table)
    table.game_player_is_ai = {i: False for i in range(1, 6)}

    joined = await _join_as_newcomer(client, monkeypatch)

    assert table.clients[joined["client_id"]].seat is None
    assert [table.seats[i] for i in (1, 3, 4, 5)] == ["ai1", "ai3", "ai4", "ai5"]


async def test_unseated_client_cannot_take_another_humans_dealt_seat(client):
    table, _, spectator = _table_in_play()
    _with_human_dealt_ai_seat(table)

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "spectator", "seat": 3},
        headers={"x-test-player": str(spectator)},
    )

    assert (r.status_code, r.json()["detail"]) == (409, "seat_owned_by_player")
    assert table.seats[3] == "ai3"


async def test_owner_may_take_back_their_human_dealt_seat(client):
    table, _, _ = _table_in_play()
    _with_human_dealt_ai_seat(table)
    owner = table.clients["gone"]

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "gone", "seat": 3},
        headers={"x-test-player": owner.player_id or ""},
    )

    assert r.status_code == 200
    assert table.seats[3] == "gone" and owner.seat == 3


async def test_timed_out_owner_may_take_back_their_human_dealt_seat(client):
    table, seated, _ = _table_in_play()
    table.game_player_is_ai = {1: True, 2: False, 3: True, 4: True, 5: True}
    # The turn timer moved the seat-2 human out; ai2 now plays their row.
    table.occupants["ai2"] = Occupant(id="ai2", display_name="AI", is_ai=True)
    table.seats[2] = "ai2"
    conn = table.clients["seated"]
    conn.seat = None
    conn.home_occupant = "ai2"

    r = await client.post(
        "/api/tables/t/seat",
        json={"client_id": "seated", "seat": 2},
        headers={"x-test-player": str(seated)},
    )

    assert r.status_code == 200
    assert table.seats[2] == "seated"
