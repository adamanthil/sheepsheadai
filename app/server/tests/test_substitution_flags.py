"""Moves made by someone other than the game_player row's owner are
flagged: bidding decisions on the row, card plays on each trick_card."""

from __future__ import annotations

import uuid
from typing import Callable

import pytest

from server.runtime.dealing import new_game_for_table
from server.runtime.models import Table
from server.services.persistence import games as games_hooks
from server.services.persistence.hand_play import persist_finalize_game
from server.services.persistence.snapshots import (
    capture_post_state,
    capture_pre_state,
)
from server.tests.recording_pool import RecordingPool
from sheepshead.game import ACTION_LOOKUP


@pytest.fixture
def pool(monkeypatch) -> RecordingPool:
    pool = RecordingPool()
    monkeypatch.setattr(games_hooks, "get_db_pool", lambda: pool)
    return pool


def _persisted_table(ai_rows: set[int]) -> Table:
    """A dealt hand whose game_player rows (ids 101-105) exist, with the
    rows for ``ai_rows`` owned by the AI and the rest by humans."""
    table = Table(id=str(uuid.uuid4()), name="flags")
    table.game = new_game_for_table(table)
    table.current_game_id = str(uuid.uuid4())
    table.game_player_ids = {s: 100 + s for s in range(1, 6)}
    table.game_player_is_ai = {s: s in ai_rows for s in range(1, 6)}
    return table


async def _play_first_trick(
    table: Table, picker: int, by_ai: Callable[[int, bool], bool]
) -> None:
    """Everyone passes to ``picker``, who picks; then play one full trick.
    ``by_ai(seat, play_started)`` says who chose each move."""
    game = table.game
    assert game is not None
    while game.current_trick == 0:
        seat = next(p.position for p in game.players if p.get_valid_action_ids())
        player = game.players[seat - 1]
        valid = player.get_valid_action_ids()
        labels = {ACTION_LOOKUP[a]: a for a in valid}
        if "PICK" in labels:
            action = labels["PICK" if seat == picker else "PASS"]
        else:
            action = min(valid)
        pre = capture_pre_state(game)
        assert player.act(action)
        post = capture_post_state(game)
        await games_hooks.fire_game_hooks(
            table, pre, post, seat=seat, by_ai=by_ai(seat, pre["play_started"])
        )


def _pick_flags(pool: RecordingPool) -> set[int]:
    return {args[0] for sql, args in pool.log if "SET is_substituted_pick" in sql}


def _trick_card_flags(pool: RecordingPool) -> dict[int, bool]:
    (rows,) = [args for sql, args in pool.log if "INSERT INTO trick_card" in sql]
    return {gp_id: substituted for _, _, gp_id, _, substituted in rows}


async def test_ai_moves_for_a_human_row_are_flagged(pool):
    # Seat 2 is a human's row, but the AI makes every one of its moves
    # (a disconnect or a timeout); the other seats are AIs playing as AIs.
    table = _persisted_table(ai_rows={1, 3, 4, 5})

    await _play_first_trick(table, picker=2, by_ai=lambda seat, _: True)

    assert _pick_flags(pool) == {102}
    flags = _trick_card_flags(pool)
    assert flags[102] is True
    assert not any(v for gp, v in flags.items() if gp != 102)


async def test_human_moves_on_an_ai_row_are_flagged(pool):
    # Seat 3 is an AI's row, taken over by a human once play started.
    table = _persisted_table(ai_rows={1, 2, 3, 4, 5})

    await _play_first_trick(
        table,
        picker=1,
        by_ai=lambda seat, started: not (seat == 3 and started),
    )

    assert _pick_flags(pool) == set()
    flags = _trick_card_flags(pool)
    assert flags[103] is True
    assert not any(v for gp, v in flags.items() if gp != 103)


async def test_owners_playing_their_own_seats_flag_nothing(pool):
    table = _persisted_table(ai_rows={1, 3, 4, 5})

    await _play_first_trick(table, picker=2, by_ai=lambda seat, _: seat != 2)

    assert _pick_flags(pool) == set()
    assert not any(_trick_card_flags(pool).values())


async def test_ai_decisions_for_a_human_row_are_counted(pool):
    # Seat 2 is a human's row the AI plays throughout: every decision it
    # makes there counts (pick, bury cards, call, card play), and no other
    # seat's does.
    table = _persisted_table(ai_rows={1, 3, 4, 5})
    seat2_moves = []

    def by_ai(seat: int, _started: bool) -> bool:
        if seat == 2:
            seat2_moves.append(seat)
        return True

    await _play_first_trick(table, picker=2, by_ai=by_ai)

    assert len(seat2_moves) >= 4  # pick, two bury cards, a card play
    assert table.ai_actions == {2: len(seat2_moves)}
    assert table.ai_actions_excused == {}


async def test_ai_decisions_on_an_excused_seat_are_excused(pool):
    table = _persisted_table(ai_rows={1, 3, 4, 5})
    table.excused_seats = {2}

    await _play_first_trick(table, picker=2, by_ai=lambda seat, _: True)

    assert table.ai_actions_excused == table.ai_actions
    assert table.ai_actions[2] > 0


async def test_human_moves_are_never_counted_as_ai_decisions(pool):
    # A human on their own row, and a human who took over an AI row.
    table = _persisted_table(ai_rows={1, 3, 4, 5})

    await _play_first_trick(
        table, picker=1, by_ai=lambda seat, started: seat not in (2, 3)
    )

    assert table.ai_actions == {}


async def test_finalize_writes_the_counters(pool):
    table = _persisted_table(ai_rows={1, 3, 4, 5})
    table.ai_actions = {2: 5}
    table.ai_actions_excused = {2: 3}

    await persist_finalize_game(pool, table, [2, -4, 1, 1, 0])

    updates = {args[-1]: args[:-1] for sql, args in pool.log if "SET score" in sql}
    assert updates[102] == (-4, 5, 3)
    assert updates[101] == (2, 0, 0)
    assert table.ai_actions == {} and table.excused_seats == set()
