"""The AI turn loop plays AI-held seats."""

from __future__ import annotations

import types

from server.runtime import ai_loop, dealing, turn_timer
from server.runtime.ai_loop import TRICK_COLLECT_SECONDS, ai_take_turns
from server.runtime.dealing import new_game_for_table
from server.runtime.models import ClientConn, Occupant, Table
from server.runtime.views import get_actor_seat


class StubAgent:
    """Always the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass


async def test_seat_lost_to_ai_on_an_all_human_deal_is_played(monkeypatch):
    # Dealt with five humans; seat 1 (first to bid) then disconnects and an
    # AI occupant takes it over. The deal still got a table AI to play it.
    monkeypatch.setattr(
        dealing, "build_table_agent", lambda settings, table_id: StubAgent()
    )
    monkeypatch.setattr(dealing, "get_settings", lambda: None)
    table = Table(id="t", name="humans")
    for seat in range(1, 6):
        cid = f"h{seat}"
        table.clients[cid] = ClientConn(client_id=cid, display_name=cid, seat=seat)
        table.seats[seat] = cid
    table.game = new_game_for_table(table)
    table.status = "playing"
    dealing.refresh_table_agent(table)
    assert isinstance(table.ai_agent, StubAgent)

    table.occupants["ai1"] = Occupant(id="ai1", display_name="AI", is_ai=True)
    table.seats[1] = "ai1"
    table.clients["h1"].seat = None
    # The loop arms the human's turn clock when it hands the turn back.
    monkeypatch.setattr(turn_timer, "turn_timeout_seconds", lambda: 60.0)

    await ai_take_turns(table)

    # Seat 1's bid was made; the turn has moved on to the humans.
    assert get_actor_seat(table) != 1
    if table.turn_timer_task:
        table.turn_timer_task.cancel()


async def test_ai_lead_waits_out_the_collect_animation(monkeypatch):
    # A human just completed a trick: the next lead must not be played under
    # the client's collect animation of that trick.
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def no_turns(table: Table) -> None:
        return None

    monkeypatch.setattr(
        ai_loop,
        "asyncio",
        types.SimpleNamespace(
            sleep=record_sleep,
            create_task=ai_loop.asyncio.create_task,
            CancelledError=ai_loop.asyncio.CancelledError,
        ),
    )
    monkeypatch.setattr(ai_loop, "ai_take_turns", no_turns)
    table = Table(id="t", name="collect")
    table.game = types.SimpleNamespace(was_trick_just_completed=True)  # type: ignore[assignment]

    ai_loop.schedule_ai_turns(table)
    assert table.ai_task is not None
    await table.ai_task
    assert sleeps == [TRICK_COLLECT_SECONDS]

    sleeps.clear()
    table.game.was_trick_just_completed = False
    ai_loop.schedule_ai_turns(table)
    await table.ai_task
    assert sleeps == []
