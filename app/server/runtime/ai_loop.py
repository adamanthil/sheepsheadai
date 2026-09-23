from __future__ import annotations

import asyncio
import logging

from server.realtime.broadcast import broadcast_table_state
from server.realtime.chat import emit_bid_chat_message
from server.runtime.ai_move import ai_act_for_seat
from server.runtime.dealing import redeal_passed_out_hand
from server.runtime.tables import (
    Table,
    get_actor_seat,
    record_hand_result,
)
from server.runtime.turn_timer import arm_turn_timer
from server.services.persistence.games import fire_game_hooks


async def ai_take_turns(table: Table) -> None:
    """Loop AI moves until a human is the actor or the game ends.
    Avoid holding the game lock across sleeps and network IO.
    """
    if not table.game:
        return
    while table.game and not table.game.is_done():
        actor = None
        move = None
        ai_occupant = None
        async with table.game_lock:
            if not table.game:
                break
            actor = get_actor_seat(table)
            if actor is None:
                break
            occupant = table.seats.get(actor)
            if not occupant:
                break
            occ = table.occupants.get(occupant)
            if not occ or not occ.is_ai:
                # Human's turn
                break
            move = await ai_act_for_seat(table, actor)
            ai_occupant = occ

        if actor is None or move is None:
            break

        await fire_game_hooks(table, move.pre, move.post, seat=actor, by_ai=True)

        action_str = move.action_str
        display_name = ai_occupant.display_name if ai_occupant else f"Seat {actor}"
        await emit_bid_chat_message(table, action_str, display_name, is_ai=True)

        # A doublers table that just passed out swaps in a fresh deal before
        # any state goes out, so the momentary leaster state is never
        # broadcast. Loop round onto the new deal from the top.
        if await redeal_passed_out_hand(table):
            await broadcast_table_state(table)
            # Beat before the new deal's first bid, so the throw-in callout
            # is readable rather than being overrun by the next PICK.
            await asyncio.sleep(1.2)
            continue

        if isinstance(action_str, str) and action_str.startswith("PLAY "):
            await asyncio.sleep(0.5)
        await broadcast_table_state(table)
        if getattr(table.game, "was_trick_just_completed", False):
            await asyncio.sleep(3.3)
        else:
            if isinstance(action_str, str) and action_str == "PASS":
                await asyncio.sleep(0.5)

    # If game ended via AI actions, mark finished, tally results, broadcast.
    if table.game and table.game.is_done():
        table.status = "finished"
        record_hand_result(table)
        await broadcast_table_state(table)

    # Whenever the loop stops on a human's turn, their clock starts.
    await arm_turn_timer(table, on_expired=schedule_ai_turns)


def schedule_ai_turns(table: Table, initial_delay: float = 0.0) -> None:
    """Schedule background AI turns for a table, cancelling any prior task."""

    async def _runner():
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        try:
            await ai_take_turns(table)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Fire-and-forget task: without this, a failure surfaces only as
            # an unretrieved-task warning at GC time.
            logging.exception("AI turn loop failed for table %s", table.id)

    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    table.ai_task = asyncio.create_task(_runner())
