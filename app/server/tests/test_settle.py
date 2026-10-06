"""A hand in play when its table closes or the server stops is played out
by the AI and recorded, never dropped."""

from __future__ import annotations

import uuid

import httpx2
import pytest
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime import lifecycle
from server.runtime.ai_loop import schedule_ai_turns
from server.runtime.dealing import new_game_for_table
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Occupant, Table
from server.runtime.settle import settle_hand
from server.services.persistence import games as games_hooks
from server.tests.recording_pool import RecordingPool
from sheepshead.game import ACTION_IDS


class StubAgent:
    """Passes when it may, else the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None
        if ACTION_IDS["PASS"] in valid_actions:
            return (ACTION_IDS["PASS"], None, None)
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass


class PickingAgent(StubAgent):
    """Picks when it may, so a hand is played rather than passed out."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None
        if ACTION_IDS["PICK"] in valid_actions:
            return (ACTION_IDS["PICK"], None, None)
        return super().act(state, valid_actions, player_id, deterministic)


@pytest.fixture
def pool(monkeypatch) -> RecordingPool:
    pool = RecordingPool()
    monkeypatch.setattr(games_hooks, "get_db_pool", lambda: pool)

    async def no_close(*_args) -> None:
        return None

    monkeypatch.setattr(lifecycle, "get_db_pool", lambda: pool)
    monkeypatch.setattr(lifecycle, "close_game_table", no_close)
    return pool


def _table_in_play(rules: dict | None = None, agent=None) -> Table:
    """Humans h2 (the host, seat 2) and h3 (seat 3) on their own rows; the
    AI dealt seats 1, 4 and 5. Persisted as game_player rows 101-105."""
    table = Table(id="t", name="settle", rules=rules or {})
    for seat in (2, 3):
        cid = f"h{seat}"
        table.clients[cid] = ClientConn(
            client_id=cid, display_name=cid, seat=seat, player_id=str(uuid.uuid4())
        )
        table.seats[seat] = cid
    for seat in (1, 4, 5):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name="AI", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"
    table.host_client_id = "h2"
    table.game = new_game_for_table(table)
    table.status = "playing"
    table.ai_agent = agent or PickingAgent()  # type: ignore[assignment]
    table.current_game_id = str(uuid.uuid4())
    table.game_player_ids = {s: 100 + s for s in range(1, 6)}
    table.game_player_is_ai = {s: s not in (2, 3) for s in range(1, 6)}
    tables.tables[table.id] = table
    return table


def _finalized_counters(pool: RecordingPool) -> dict[int, tuple[int, int]]:
    """game_player id -> (ai_actions, ai_actions_excused) as finalized."""
    return {args[3]: (args[1], args[2]) for sql, args in pool.log if "SET score" in sql}


async def test_settlement_plays_the_hand_out_and_records_it(pool):
    table = _table_in_play()

    await settle_hand(table, charge=lambda seat: seat == 2)

    assert table.game is not None and table.game.is_done()
    assert table.status == "finished"
    assert len(table.results_history) == 1
    counters = _finalized_counters(pool)
    # Seat 2's moves count against its player; seat 3's are excused.
    charged, excused = counters[102], counters[103]
    assert charged[0] > 0 and charged[1] == 0
    assert excused[0] > 0 and excused[1] == excused[0]
    # AI-dealt rows carry no counters.
    assert counters[101] == (0, 0)


async def test_settlement_stops_at_a_passed_out_doublers_deal(pool):
    table = _table_in_play(rules={"allPassMode": "doublers"}, agent=StubAgent())
    dealt = table.game

    await settle_hand(table, charge=lambda seat: True)

    # Thrown in, not redealt: nobody is left to play a new deal.
    assert table.game is dealt and dealt is not None and dealt.is_leaster
    assert table.score_multiplier == 1
    assert table.results_history == []
    assert any("SET time_closed" in sql for sql, _ in pool.log)


async def test_settlement_is_a_no_op_between_hands(pool):
    table = _table_in_play()
    table.status = "finished"

    await settle_hand(table, charge=lambda seat: True)

    assert table.game is not None and not table.game.is_done()
    assert not table.settling


async def test_no_ai_turn_loop_starts_during_settlement(pool):
    table = _table_in_play()
    table.settling = True

    schedule_ai_turns(table)

    assert table.ai_task is None


async def test_drain_settles_every_live_hand_excusing_everyone(pool):
    table = _table_in_play()

    await lifecycle.settle_all_for_restart()

    assert table.game is not None and table.game.is_done()
    counters = _finalized_counters(pool)
    for gp_id in (102, 103):
        assert counters[gp_id][0] > 0 and counters[gp_id][1] == counters[gp_id][0]


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


async def test_host_close_settles_the_hand_charging_only_the_host(client, pool):
    table = _table_in_play()
    host = table.clients["h2"]

    r = await client.post(
        "/api/tables/t/close",
        json={"client_id": "h2"},
        headers={"x-test-player": host.player_id or ""},
    )

    assert r.status_code == 200, r.text
    assert "t" not in tables.tables
    counters = _finalized_counters(pool)
    assert counters[102][0] > 0 and counters[102][1] == 0
    assert counters[103][0] > 0 and counters[103][1] == counters[103][0]


async def test_no_move_is_accepted_while_a_hand_settles(client, pool):
    table = _table_in_play()
    table.settling = True
    conn = table.clients["h2"]

    r = await client.post(
        "/api/tables/t/action",
        json={"client_id": "h2", "action_id": ACTION_IDS["PASS"]},
        headers={"x-test-player": conn.player_id or ""},
    )

    assert (r.status_code, r.json()["detail"]) == (409, "table_closing")


async def test_autoclose_after_everyone_left_charges_everyone(pool):
    table = _table_in_play()

    await lifecycle.close_table(table, reason="idle_all_disconnected")

    counters = _finalized_counters(pool)
    for gp_id in (102, 103):
        assert counters[gp_id][0] > 0 and counters[gp_id][1] == 0
