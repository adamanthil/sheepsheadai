"""Settle a live hand: the AI plays it out, unpaced, and it is recorded.

No path that closes a table (host close, everyone leaving) or stops the
server may drop a hand in progress -- a dropped hand vanishes from stats,
which would let a player erase a hand they were losing. Settlement moves
go through the same persistence hooks as any AI move, so they are counted
against each human row; ``charge`` decides whose settlement moves count
against them rather than being excused.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from server.realtime.broadcast import broadcast_table_state
from server.runtime.ai_move import ai_act_for_seat
from server.runtime.dealing import hand_passed_out
from server.runtime.models import Table
from server.runtime.turn_timer import cancel_turn_timer
from server.runtime.views import get_actor_seat, record_hand_result
from server.services.persistence.games import fire_game_hooks


async def _stop(task: Optional[asyncio.Task]) -> None:
    """Cancel ``task`` and wait for it to unwind, so it releases the game
    lock and never applies a move alongside settlement."""
    if task is None or task.done() or task is asyncio.current_task():
        return
    task.cancel()
    await asyncio.wait({task})


async def settle_hand(table: Table, charge: Callable[[int], bool]) -> None:
    """Play the hand in progress to its end with the AI in every seat.

    ``charge(seat)`` says whether the AI's settlement moves for a human
    seat count against that player; every other human seat is excused
    for the rest of the hand. A no-op unless a hand is in play.
    """
    if not table.hand_in_play:
        return
    # Stays set: the table closes, or the server exits, once this returns.
    table.settling = True
    await _stop(table.ai_task)
    timer = table.turn_timer_task
    cancel_turn_timer(table)
    await _stop(timer)
    for seat, is_ai in table.game_player_is_ai.items():
        if not is_ai and not charge(seat):
            table.excused_seats.add(seat)

    while table.game is not None and not table.game.is_done():
        async with table.game_lock:
            seat = get_actor_seat(table)
            move = await ai_act_for_seat(table, seat) if seat else None
        if seat is None or move is None:
            logging.warning("table %s: hand could not be settled", table.id)
            return
        await fire_game_hooks(table, move.pre, move.post, seat=seat, by_ai=True)
        # A passed-out doublers deal is closed by the fifth PASS's hook and
        # never played; redealing it would start a hand nobody is there to
        # play.
        if hand_passed_out(table):
            return

    table.status = "finished"
    record_hand_result(table)
    await broadcast_table_state(table)
