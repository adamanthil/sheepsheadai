"""A human who runs out of time gets an AI move and a strike; three
strikes in a row move them to spectator."""

from __future__ import annotations

import json
import uuid

import httpx
import pytest
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime import turn_timer
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import (
    ClientConn,
    Occupant,
    Table,
    get_actor_seat,
    get_valid_action_ids_for_seat,
    tables,
)


class StubAgent:
    """Always the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def _table_with_human_to_act() -> tuple[Table, ClientConn, _RecordingSocket]:
    """Seat 1 (first to bid) is a connected human; seats 2-5 are AIs."""
    table = Table(id="t", name="clock")
    ws = _RecordingSocket()
    conn = ClientConn(
        client_id="human", display_name="Pat", seat=1, player_id=str(uuid.uuid4())
    )
    conn.sockets.add(ws)  # type: ignore[arg-type]
    table.clients["human"] = conn
    table.seats[1] = "human"
    for seat in range(2, 6):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name=f"AI{seat}", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    table.game = new_game_for_table(table)
    table.status = "playing"
    table.ai_agent = StubAgent()
    tables.tables[table.id] = table
    assert get_actor_seat(table) == 1
    return table, conn, ws


@pytest.fixture
def hooks(monkeypatch) -> list[tuple[int, bool]]:
    monkeypatch.setattr(turn_timer, "turn_timeout_seconds", lambda: 0.01)
    calls: list[tuple[int, bool]] = []

    async def record(table, pre, post, seat, by_ai):
        calls.append((seat, by_ai))

    monkeypatch.setattr(turn_timer, "fire_game_hooks", record)
    return calls


async def _arm_and_expire(table: Table, continued: list[Table]) -> None:
    await turn_timer.arm_turn_timer(table, on_expired=continued.append)
    task = table.turn_timer_task
    assert task is not None
    await task


async def test_timeout_plays_for_the_human_and_strikes(hooks):
    table, conn, ws = _table_with_human_to_act()
    continued: list[Table] = []

    await _arm_and_expire(table, continued)

    assert table.move_seq == 1  # seat 1's bid was made for them
    assert hooks == [(1, True)]  # recorded as the AI's move
    assert conn.timeout_strikes == 1
    assert conn.seat == 1
    assert continued == [table]
    assert ws.sent[0] == {"type": "turn_timer", "seat": 1, "secondsLeft": 0.01}
    last = table.chat_log[-1]
    assert (last["author"], last["body"]) == (
        "Pat",
        "ran out of time; the AI played for them",
    )


async def test_a_move_in_time_leaves_the_timer_stale(hooks):
    table, conn, _ = _table_with_human_to_act()
    continued: list[Table] = []
    await turn_timer.arm_turn_timer(table, on_expired=continued.append)
    task = table.turn_timer_task
    assert task is not None

    # The human moves before the clock runs out (as post_action does).
    table.game.players[0].act(min(get_valid_action_ids_for_seat(table, 1)))  # type: ignore[union-attr]
    table.move_seq += 1
    await task

    assert hooks == []
    assert conn.timeout_strikes == 0
    assert continued == []


async def test_third_strike_in_a_row_moves_the_player_to_spectator(hooks):
    table, conn, _ = _table_with_human_to_act()
    conn.timeout_strikes = 2
    table.reserved_ai_by_human["human"] = "ai-held"
    table.occupants["ai-held"] = Occupant(id="ai-held", display_name="Tom", is_ai=True)

    await _arm_and_expire(table, [])

    assert conn.seat is None
    assert table.seats[1] == "ai-held"
    assert table.occupants["ai-held"].is_ai
    # No reservation left, so reconnecting doesn't seat them again.
    assert "human" not in table.reserved_ai_by_human
    # ...but the seat stays theirs to take back by hand.
    assert conn.home_occupant == "ai-held"
    assert conn.timeout_strikes == 0
    assert "is now watching" in table.chat_log[-1]["body"]


async def test_own_move_clears_strikes(app, hooks):
    table, conn, _ = _table_with_human_to_act()
    conn.timeout_strikes = 2
    player_id = uuid.UUID(conn.player_id or "")

    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=player_id)

    app.dependency_overrides[current_player] = fake_player
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            r = await client.post(
                f"/api/tables/{table.id}/action",
                json={
                    "client_id": "human",
                    "action_id": min(get_valid_action_ids_for_seat(table, 1)),
                },
            )
    finally:
        app.dependency_overrides.clear()
        if table.ai_task:
            table.ai_task.cancel()

    assert r.status_code == 200, r.text
    assert conn.timeout_strikes == 0
