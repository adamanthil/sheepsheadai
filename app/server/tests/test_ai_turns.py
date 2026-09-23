"""The AI turn loop plays AI-held seats, loading the table's AI if needed."""

from __future__ import annotations

from server.runtime import dealing, turn_timer
from server.runtime.ai_loop import ai_take_turns
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import ClientConn, Occupant, Table, get_actor_seat


class StubAgent:
    """Always the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass


async def test_seat_lost_to_ai_on_an_all_human_deal_is_played(monkeypatch):
    # Dealt with five humans, so the table has no AI. Seat 1 (first to
    # bid) then disconnects and an AI occupant takes it over.
    table = Table(id="t", name="humans")
    for seat in range(1, 6):
        cid = f"h{seat}"
        table.clients[cid] = ClientConn(client_id=cid, display_name=cid, seat=seat)
        table.seats[seat] = cid
    table.game = new_game_for_table(table)
    table.status = "playing"
    dealing.refresh_table_agent(table)
    assert table.ai_agent is None

    table.occupants["ai1"] = Occupant(id="ai1", display_name="AI", is_ai=True)
    table.seats[1] = "ai1"
    table.clients["h1"].seat = None
    monkeypatch.setattr(
        dealing, "build_table_agent", lambda settings, table_id: StubAgent()
    )
    monkeypatch.setattr(dealing, "get_settings", lambda: None)
    # The loop arms the human's turn clock when it hands the turn back.
    monkeypatch.setattr(turn_timer, "turn_timeout_seconds", lambda: 60.0)

    await ai_take_turns(table)

    assert isinstance(table.ai_agent, StubAgent)
    # Seat 1's bid was made; the turn has moved on to the humans.
    assert get_actor_seat(table) != 1
    if table.turn_timer_task:
        table.turn_timer_task.cancel()
