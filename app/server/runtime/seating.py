from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional, Set

from server.runtime.tables import Table, Occupant
from server.realtime.broadcast import broadcast_table_event, broadcast_table_update
from server.realtime.chat import add_chat_message, broadcast_chat_append
from server.runtime.ai_loop import schedule_ai_turns

# Name pool for auto-generated AI occupants (disconnect replacement here,
# and table auto-fill in server.api.games). Content and order are shared;
# each call site keeps its own indexing scheme (time-indexed here,
# seat-indexed in games.py) so behavior is unchanged.
AI_NAME_POOL = ("Dan", "Kyle", "John", "Trevor", "Tim", "Tom")

# Fixed seat-label names for the /analyze simulate trace (server.services.
# analyze). This list has different content/order than AI_NAME_POOL above
# (it includes "Andrew" and is always exactly 5 long, one per seat) -- kept
# as a separate constant rather than unified, since changing it would alter
# user-visible analyze output.
ANALYZE_SEAT_NAMES = ["Dan", "Kyle", "Trevor", "John", "Andrew"]


def is_ai_occupant(table: Table, occ_id: Optional[str]) -> bool:
    if not occ_id:
        return False
    occ = table.occupants.get(occ_id)
    return bool(occ and occ.is_ai)


def reserved_ai_ids(table: Table) -> Set[str]:
    return {v for v in table.reserved_ai_by_human.values() if v}


def pick_join_ai_seat(table: Table) -> Optional[int]:
    """Pick an AI seat for a newcomer, preferring AIs not reserved for disconnected humans."""
    reserved_ids = reserved_ai_ids(table)
    non_reserved: list = []
    reserved: list = []
    for i in range(1, 6):
        occ = table.seats.get(i)
        if is_ai_occupant(table, occ):
            if occ in reserved_ids:
                reserved.append(i)
            else:
                non_reserved.append(i)
    if non_reserved:
        return non_reserved[0]
    if reserved:
        return reserved[0]
    return None


def lowest_non_human_seat(table: Table) -> Optional[int]:
    """Return the lowest seat index that is either empty or occupied by an AI."""
    for i in range(1, 6):
        occ = table.seats.get(i)
        if not occ or is_ai_occupant(table, occ):
            return i
    return None


def find_seat_of_occupant(table: Table, occ_id: str) -> Optional[int]:
    for i in range(1, 6):
        if table.seats.get(i) == occ_id:
            return i
    return None


def allocate_ai_occupant(display_name: Optional[str] = None) -> Occupant:
    occ_id = str(uuid.uuid4())
    return Occupant(
        id=occ_id,
        display_name=display_name
        or AI_NAME_POOL[int(time.time()) % len(AI_NAME_POOL)],
        is_ai=True,
    )


async def replace_ai_with_human_and_reserve(
    table: Table, seat: int, client_id: str
) -> None:
    """Replace AI at seat with human client and remember the AI for future reclaim."""
    prev_occ = table.seats.get(seat)
    if not is_ai_occupant(table, prev_occ):
        return
    table.seats[seat] = client_id
    if client_id in table.clients:
        table.clients[client_id].seat = seat
    if prev_occ:
        if prev_occ in reserved_ai_ids(table):
            placeholder = allocate_ai_occupant()
            table.occupants[placeholder.id] = placeholder
            table.reserved_ai_by_human[client_id] = placeholder.id
        else:
            table.reserved_ai_by_human[client_id] = prev_occ
    display_name = (
        table.clients.get(client_id).display_name
        if client_id in table.clients
        else "A player"
    )
    msg_dict = await add_chat_message(
        table, "system", f"{display_name} joined and took seat {seat}"
    )
    await broadcast_chat_append(table, msg_dict)
    await broadcast_table_event(
        table,
        {
            "type": "lobby_event",
            "message": f"{display_name} joined and took seat {seat}",
            "table": table.to_public_dict(),
        },
    )
    await broadcast_table_update(table)
    schedule_ai_turns(table)


def cancel_disconnect_task(table: Table, client_id: str) -> None:
    task = table.disconnect_tasks.get(client_id)
    if task and not task.done():
        task.cancel()
    table.disconnect_tasks.pop(client_id, None)


def schedule_ai_replacement_for_disconnected_human(
    table: Table, client_id: str
) -> None:
    """After a grace period, replace the disconnected human's seat with AI and broadcast."""

    async def _runner():
        try:
            # In waiting room (open), fill immediately; during play, wait 10s
            delay = 0.0 if table.status == "open" else 10.0
            if delay > 0:
                await asyncio.sleep(delay)
            async with table.state_lock:
                conn = table.clients.get(client_id)
                if not conn:
                    return
                if conn.websocket is not None:
                    return
                seat_idx: Optional[int] = None
                for i in range(1, 6):
                    if table.seats.get(i) == client_id:
                        seat_idx = i
                        break
                if not seat_idx:
                    return
                ai_id = table.reserved_ai_by_human.get(client_id)
                if not ai_id:
                    occ = allocate_ai_occupant()
                    table.occupants[occ.id] = occ
                    ai_id = occ.id
                    table.reserved_ai_by_human[client_id] = ai_id
                else:
                    if ai_id not in table.occupants:
                        table.occupants[ai_id] = Occupant(
                            id=ai_id, display_name="AI", is_ai=True
                        )
                table.seats[seat_idx] = ai_id
                conn.seat = None
            msg_dict = await add_chat_message(
                table, "system", f"{conn.display_name} disconnected. Seat filled by AI."
            )
            await broadcast_chat_append(table, msg_dict)
            await broadcast_table_event(
                table,
                {
                    "type": "lobby_event",
                    "message": f"{conn.display_name} disconnected. Seat filled by AI.",
                    "table": table.to_public_dict(),
                },
            )
            await broadcast_table_update(table)
            schedule_ai_turns(table)
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception(
                "disconnect replacement failed for table %s client %s",
                table.id,
                client_id,
            )
        finally:
            table.disconnect_tasks.pop(client_id, None)

    cancel_disconnect_task(table, client_id)
    table.disconnect_tasks[client_id] = asyncio.create_task(_runner())
