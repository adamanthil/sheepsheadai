"""AI occupants: the name pool and allocation of a fresh AI seat-holder.

A leaf module (it imports only the data model) so the seating, table API,
and AI-turn code can all allocate AIs without importing one another.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from server.runtime.models import ClientConn, Occupant, Table

# Name pool for auto-generated AI occupants: seat-indexed for the AIs a
# table opens with, time-indexed for any allocated later.
AI_NAME_POOL = ("Dan", "Kyle", "John", "Trevor", "Tim", "Tom")


def allocate_ai_occupant(display_name: Optional[str] = None) -> Occupant:
    occ_id = str(uuid.uuid4())
    return Occupant(
        id=occ_id,
        display_name=display_name or AI_NAME_POOL[int(time.time()) % len(AI_NAME_POOL)],
        is_ai=True,
    )


def seat_ai_in_every_seat(table: Table) -> None:
    """Seat an AI in every empty seat: a table opens full of AIs, and
    players take seats over from them. Caller holds ``table.state_lock``
    or owns a table nobody else can see yet."""
    for seat in range(1, 6):
        if not table.seats[seat]:
            occ = allocate_ai_occupant(AI_NAME_POOL[(seat - 1) % len(AI_NAME_POOL)])
            table.occupants[occ.id] = occ
            table.seats[seat] = occ.id


def give_seat_to_ai(table: Table, conn: ClientConn) -> Optional[int]:
    """Seat an AI where ``conn`` sits, for good, and return the seat.

    The player's AI reservation is dropped rather than kept for a reclaim,
    so a reconnect doesn't quietly seat them again; the reserved AI (if
    any) is the one that takes the seat. Caller holds ``table.state_lock``.
    """
    reserved = table.reserved_ai_by_human.pop(conn.client_id, None)
    seat = conn.seat
    if seat is None or table.seats.get(seat) != conn.client_id:
        return None
    occ = table.occupants.get(reserved) if reserved else None
    if occ is None:
        occ = allocate_ai_occupant()
        table.occupants[occ.id] = occ
    table.seats[seat] = occ.id
    conn.seat = None
    return seat
