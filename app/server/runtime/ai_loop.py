from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sheepshead import ACTION_LOOKUP

from server.realtime.broadcast import broadcast_table_state
from server.realtime.chat import emit_bid_chat_message
from server.runtime.tables import (
    Table,
    get_actor_seat,
    record_hand_result,
)
from server.services.ai_loader import inference_limit
from server.services.persistence.games import (
    capture_post_state,
    capture_pre_state,
    fire_game_hooks,
)


def _observe_seats(agent, observations: list[tuple[dict, int]]) -> None:
    for state, seat in observations:
        agent.observe(state, player_id=seat)


async def ai_observe_all(table: Table, except_seat: Optional[int] = None) -> None:
    """Update the AI's recurrent memory for every AI seat.

    observe() is a torch forward pass; run the batch in a worker thread so
    the event loop (all tables, websockets, /health) never blocks on it.
    """
    if not table.ai_agent or not table.game:
        return
    observations: list[tuple[dict, int]] = []
    for seat, occupant in table.seats.items():
        if not occupant:
            continue
        occ = table.occupants.get(occupant)
        if not occ or not occ.is_ai:
            continue
        if seat == except_seat:
            continue
        player = table.game.players[seat - 1]
        observations.append((player.get_state_dict(), seat))
    if not observations:
        return
    async with inference_limit:
        await asyncio.to_thread(_observe_seats, table.ai_agent, observations)


async def ai_take_turns(table: Table) -> None:
    """Loop AI moves until a human is the actor or the game ends.
    Avoid holding the game lock across sleeps and network IO.
    """
    if not table.game or not table.ai_agent:
        return
    while table.game and not table.game.is_done():
        actor = None
        action_id = None
        ai_occupant = None
        pre = None
        post = None
        async with table.game_lock:
            if not table.game or not table.ai_agent:
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
            player = table.game.players[actor - 1]
            state = player.get_state_dict()
            valid = player.get_valid_action_ids()
            if not valid:
                break
            pre = capture_pre_state(table.game)
            # Torch inference runs in a worker thread while game_lock is held:
            # this table stays consistent, every other table (and the event
            # loop itself) keeps moving.
            async with inference_limit:
                action_id, _, _ = await asyncio.to_thread(
                    table.ai_agent.act,
                    state,
                    valid_actions=valid,
                    player_id=actor,
                    deterministic=True,
                )
            ok = player.act(int(action_id))
            if not ok:
                raise RuntimeError(
                    f"AI produced invalid action_id {action_id} for seat {actor}; valid set: {sorted(list(valid))}"
                )
            post = capture_post_state(table.game)
            ai_occupant = occ

        if actor is None or action_id is None:
            break
        await ai_observe_all(table, except_seat=actor)

        if pre is not None and post is not None:
            await fire_game_hooks(table, pre, post)

        action_str = ACTION_LOOKUP.get(action_id, "")
        display_name = ai_occupant.display_name if ai_occupant else f"Seat {actor}"
        await emit_bid_chat_message(table, action_str, display_name)
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
