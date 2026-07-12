"""Seat-mutation atomicity and bookkeeping pruning."""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime.tables import (
    ClientConn,
    Occupant,
    Table,
    prune_table_state,
    tables,
)


async def test_concurrent_seat_grab_yields_one_winner(app):
    table = Table(id="t1", name="race")
    p1, p2 = uuid.uuid4(), uuid.uuid4()
    table.clients["c1"] = ClientConn(client_id="c1", display_name="a", player_id=str(p1))
    table.clients["c2"] = ClientConn(client_id="c2", display_name="b", player_id=str(p2))
    tables.tables["t1"] = table

    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r1, r2 = await asyncio.gather(
                client.post(
                    "/api/tables/t1/seat",
                    json={"client_id": "c1", "seat": 3},
                    headers={"x-test-player": str(p1)},
                ),
                client.post(
                    "/api/tables/t1/seat",
                    json={"client_id": "c2", "seat": 3},
                    headers={"x-test-player": str(p2)},
                ),
            )
    finally:
        app.dependency_overrides.clear()

    assert sorted([r1.status_code, r2.status_code]) == [200, 409]
    # Exactly one of the two owns the seat.
    assert table.seats[3] in {"c1", "c2"}


def test_prune_table_state_sweeps_stale_bookkeeping():
    table = Table(id="t2", name="prune")
    # A long-gone, unseated client with an AI reservation.
    stale = ClientConn(client_id="gone", display_name="g", player_id="p")
    stale.disconnected_at = time.time() - 3600
    table.clients["gone"] = stale
    table.reserved_ai_by_human["gone"] = "ai-1"
    table.occupants["ai-1"] = Occupant(id="ai-1", display_name="AI", is_ai=True)
    # An orphaned AI occupant (no seat, no reservation).
    table.occupants["ai-orphan"] = Occupant(id="ai-orphan", display_name="AI", is_ai=True)
    # A seated AI and a recently-disconnected client must both survive.
    table.occupants["ai-seated"] = Occupant(id="ai-seated", display_name="AI", is_ai=True)
    table.seats[1] = "ai-seated"
    recent = ClientConn(client_id="recent", display_name="r", player_id="p2")
    recent.disconnected_at = time.time() - 5
    table.clients["recent"] = recent

    prune_table_state(table)

    assert "gone" not in table.clients
    assert "gone" not in table.reserved_ai_by_human
    assert "ai-1" not in table.occupants
    assert "ai-orphan" not in table.occupants
    assert "ai-seated" in table.occupants
    assert "recent" in table.clients
