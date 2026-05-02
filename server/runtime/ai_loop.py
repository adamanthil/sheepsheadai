from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from sheepshead import ACTION_LOOKUP, CARD_FULL_NAMES

from server.runtime.tables import Table, get_actor_seat, get_valid_action_ids_for_seat
from server.realtime.broadcast import broadcast_table_state
from server.realtime.chat import add_chat_message, broadcast_chat_append


async def ai_observe_all(table: Table, except_seat: Optional[int] = None) -> None:
    if not table.ai_agent or not table.game:
        return
    for seat, occupant in table.seats.items():
        if not occupant:
            continue
        occ = table.occupants.get(occupant)
        if not occ or not occ.is_ai:
            continue
        if seat == except_seat:
            continue
        player = table.game.players[seat - 1]
        state = player.get_state_dict()
        table.ai_agent.observe(state, player_id=seat)


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
            action_id, _, _ = table.ai_agent.act(
                state, valid_actions=valid, player_id=actor, deterministic=True
            )
            ok = player.act(int(action_id))
            if not ok:
                raise RuntimeError(
                    f"AI produced invalid action_id {action_id} for seat {actor}; valid set: {sorted(list(valid))}"
                )
            ai_occupant = occ

        if actor is None or action_id is None:
            break
        await ai_observe_all(table, except_seat=actor)

        action_str = ACTION_LOOKUP.get(action_id, "")
        if action_str in (
            "PICK",
            "PASS",
            "ALONE",
            "JD PARTNER",
        ) or action_str.startswith("CALL "):
            display_name = ai_occupant.display_name if ai_occupant else f"Seat {actor}"
            if action_str == "PICK":
                msg_dict = await add_chat_message(
                    table, "system", f"{display_name} picked"
                )
                await broadcast_chat_append(table, msg_dict)
            elif action_str == "PASS":
                msg_dict = await add_chat_message(
                    table, "system", f"{display_name} passed"
                )
                await broadcast_chat_append(table, msg_dict)
            elif action_str == "ALONE":
                msg_dict = await add_chat_message(
                    table, "system", f"{display_name} goes alone"
                )
                await broadcast_chat_append(table, msg_dict)
            elif action_str == "JD PARTNER":
                msg_dict = await add_chat_message(
                    table, "system", f"{display_name} chose JD partner"
                )
                await broadcast_chat_append(table, msg_dict)
            elif action_str.startswith("CALL "):
                parts = action_str.split()
                called_card = parts[1] if len(parts) > 1 else ""
                under = "under" if len(parts) > 2 and parts[2] == "UNDER" else ""
                card_display = CARD_FULL_NAMES.get(called_card, called_card)
                call_msg = f"{display_name} calls {card_display}"
                if under:
                    call_msg += " under"
                msg_dict = await add_chat_message(table, "system", call_msg)
                await broadcast_chat_append(table, msg_dict)
        if isinstance(action_str, str) and action_str.startswith("PLAY "):
            await asyncio.sleep(0.5)
        await broadcast_table_state(table)
        if getattr(table.game, "was_trick_just_completed", False):
            await asyncio.sleep(3.3)
        else:
            if isinstance(action_str, str) and action_str == "PASS":
                await asyncio.sleep(0.5)

    # If game ended via AI actions, mark finished, tally results, record history, and broadcast
    if table.game and table.game.is_done():
        table.status = "finished"
        if not table.results_counted:
            for i in range(1, 6):
                occ = table.seats[i]
                if not occ:
                    continue
                pscore = table.game.players[i - 1].get_score()
                table.running_scores[occ] = table.running_scores.get(occ, 0) + int(
                    pscore
                )
            table.results_counted = True
            try:
                entry: Dict[str, Any] = {
                    "hand": len(table.results_history) + 1,
                    "timestamp": time.time(),
                    "bySeat": {},
                    "sum": 0,
                }
                pub = table.to_public_dict()
                for i in range(1, 6):
                    name = pub["seats"][i]
                    occ_id = pub["seatOccupants"][i]
                    score = int(table.game.players[i - 1].get_score())
                    entry["bySeat"][i] = {"name": name, "id": occ_id, "score": score}
                    entry["sum"] += score
                table.results_history.append(entry)
            except Exception:
                logging.exception(
                    "failed to append results history for table %s", table.id
                )
        await broadcast_table_state(table)


def schedule_ai_turns(table: Table, initial_delay: float = 0.0) -> None:
    """Schedule background AI turns for a table, cancelling any prior task."""

    async def _runner():
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        await ai_take_turns(table)

    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    table.ai_task = asyncio.create_task(_runner())
