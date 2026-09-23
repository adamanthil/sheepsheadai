"""AI occupants: the name pool and allocation of a fresh AI seat-holder.

A leaf module (it imports only the data model) so the seating, table API,
and AI-turn code can all allocate AIs without importing one another.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from server.runtime.models import ClientConn, Occupant, Table

# Name pool for auto-generated AI occupants (disconnect replacement and
# table auto-fill). Each call site keeps its own indexing scheme (time-
# indexed in allocate_ai_occupant, seat-indexed in server.api.games).
AI_NAME_POOL = ("Dan", "Kyle", "John", "Trevor", "Tim", "Tom")


def allocate_ai_occupant(display_name: Optional[str] = None) -> Occupant:
    occ_id = str(uuid.uuid4())
    return Occupant(
        id=occ_id,
        display_name=display_name or AI_NAME_POOL[int(time.time()) % len(AI_NAME_POOL)],
        is_ai=True,
    )


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
