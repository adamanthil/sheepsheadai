"""Before the deal: a table opens with an AI in every seat, and players
take seats over from them."""

from __future__ import annotations

import uuid

import httpx2
import pytest
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime.manager import TableManager, tables
from server.runtime.models import ClientConn


async def test_new_table_opens_with_an_ai_in_every_seat():
    table = await TableManager().create_table("full", {})

    occupants = [table.occupants[table.seats[i] or ""] for i in range(1, 6)]
    assert all(occ.is_ai for occ in occupants)
    assert [occ.display_name for occ in occupants] == [
        "Dan",
        "Kyle",
        "John",
        "Trevor",
        "Tim",
    ]


@pytest.fixture
async def client(app):
    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


async def test_moving_seats_hands_the_old_seat_back_to_its_ai(client):
    table = await tables.create_table("move", {})
    player = uuid.uuid4()
    table.clients["p"] = ClientConn(
        client_id="p", display_name="P", seat=5, player_id=str(player)
    )
    ai5, ai1 = table.seats[5], table.seats[1]
    table.seats[5] = "p"
    table.reserved_ai_by_human["p"] = ai5 or ""

    r = await client.post(
        f"/api/tables/{table.id}/seat",
        json={"client_id": "p", "seat": 1},
        headers={"x-test-player": str(player)},
    )

    assert r.status_code == 200
    assert table.seats[1] == "p"
    assert table.seats[5] == ai5
    assert table.reserved_ai_by_human["p"] == ai1
    tables.delete_table(table.id)
