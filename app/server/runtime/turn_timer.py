"""Turn timer: a human seat that doesn't move in time gets an AI move.

The AI turn loop arms the timer whenever it hands the turn to a human.
When it runs out, the table's AI plays that one move (flagged as a
substitution in the hand record) and the player takes a strike; any move
of their own clears their strikes. After MAX_TIMEOUT_STRIKES in a row the
player is moved to spectator and an AI takes the seat, which they can
take back with the spectator view's "Take seat" button.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from server.config import get_settings
from server.realtime.broadcast import (
    broadcast_table_event,
    broadcast_table_state,
    broadcast_table_update,
)
from server.realtime.chat import (
    add_chat_message,
    broadcast_chat_append,
    emit_bid_chat_message,
)
from server.runtime.ai_move import ai_act_for_seat, ai_observe_all
from server.runtime.dealing import ensure_table_agent, redeal_passed_out_hand
from server.runtime.models import ClientConn, Table
from server.runtime.occupants import give_seat_to_ai
from server.runtime.tasks import spawn
from server.runtime.views import get_actor_seat
from server.services.persistence.games import fire_game_hooks

MAX_TIMEOUT_STRIKES = 3


def turn_timeout_seconds() -> float:
    return get_settings().sheepshead_turn_timeout_seconds


def _human_to_act(table: Table) -> Optional[tuple[int, ClientConn]]:
    seat = get_actor_seat(table)
    if seat is None:
        return None
    conn = table.clients.get(table.seats.get(seat) or "")
    return (seat, conn) if conn is not None else None


def cancel_turn_timer(table: Table) -> None:
    task = table.turn_timer_task
    if task and not task.done() and task is not asyncio.current_task():
        task.cancel()
    table.turn_timer_task = None
    table.turn_timer_key = None


async def arm_turn_timer(table: Table, on_expired: Callable[[Table], None]) -> None:
    """Start the clock for the human whose turn it is, if any.

    Idempotent for the same turn. ``on_expired`` continues play after the
    timeout move (the AI turn loop passes its scheduler).
    """
    turn = _human_to_act(table) if table.game else None
    if turn is None:
        cancel_turn_timer(table)
        return
    seat, _ = turn
    key = (id(table.game), table.move_seq, seat)
    live = table.turn_timer_task is not None and not table.turn_timer_task.done()
    if table.turn_timer_key == key and live:
        return
    cancel_turn_timer(table)
    seconds = turn_timeout_seconds()
    table.turn_timer_key = key
    table.turn_deadline = time.monotonic() + seconds
    table.turn_timer_task = spawn(
        _expire(table, key, on_expired, seconds), f"turn-timer:{table.id}"
    )
    await broadcast_table_event(
        table,
        {"type": "turn_timer", "seat": seat, "secondsLeft": seconds},
    )


async def _expire(
    table: Table, key: tuple, on_expired: Callable[[Table], None], seconds: float
) -> None:
    try:
        await asyncio.sleep(seconds)
        async with table.game_lock:
            # Any move, deal, or seat change since arming makes this stale.
            turn = _human_to_act(table) if table.game else None
            if turn is None or table.turn_timer_key != key:
                return
            seat, conn = turn
            if key != (id(table.game), table.move_seq, seat):
                return
            ensure_table_agent(table)
            move = await ai_act_for_seat(table, seat)
        table.turn_timer_key = None
        if move is None:
            return

        await ai_observe_all(table, except_seat=seat)
        await fire_game_hooks(table, move.pre, move.post, seat=seat, by_ai=True)
        await emit_bid_chat_message(table, move.action_str, conn.display_name)
        await redeal_passed_out_hand(table)

        conn.timeout_strikes += 1
        if conn.timeout_strikes >= MAX_TIMEOUT_STRIKES:
            await _move_to_spectator(table, conn, seat)
        else:
            msg_dict = await add_chat_message(
                table,
                "system",
                "ran out of time; the AI played for them",
                author=conn.display_name,
            )
            await broadcast_chat_append(table, msg_dict)
        await broadcast_table_state(table)
        on_expired(table)
    except asyncio.CancelledError:
        return


async def _move_to_spectator(table: Table, conn: ClientConn, seat: int) -> None:
    async with table.state_lock:
        if give_seat_to_ai(table, conn) is None:
            return
        conn.timeout_strikes = 0
        conn.home_occupant = table.seats[seat]
    msg_dict = await add_chat_message(
        table,
        "system",
        f"missed {MAX_TIMEOUT_STRIKES} turns in a row and is now watching. "
        f"Seat {seat} went to the AI.",
        author=conn.display_name,
    )
    await broadcast_chat_append(table, msg_dict)
    await broadcast_table_update(table)
