"""Table registry: creation (with a hard cap), lookup, and state pruning."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List

from server.runtime.models import Table

# DoS backstop: hard ceiling on concurrently open tables. Each table holds an
# in-memory Game plus (once started) its own PPOAgent, so unbounded creation
# is a memory exhaustion vector.
MAX_TABLES = 200


class TableLimitError(RuntimeError):
    """Raised when creating a table would exceed MAX_TABLES."""


class TableManager:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
        self._lock = asyncio.Lock()

    async def create_table(
        self, name: str, fill_with_ai: bool, rules: Dict[str, Any]
    ) -> Table:
        async with self._lock:
            if len(self.tables) >= MAX_TABLES:
                raise TableLimitError()
            tid = str(uuid.uuid4())
            table = Table(id=tid, name=name, fill_with_ai=fill_with_ai, rules=rules)
            self.tables[tid] = table
            return table

    def get_table(self, table_id: str) -> Table:
        if table_id not in self.tables:
            raise KeyError("table_not_found")
        return self.tables[table_id]

    def list_tables(self) -> List[Dict[str, Any]]:
        return [t.to_public_dict() for t in self.tables.values()]

    def delete_table(self, table_id: str) -> None:
        if table_id in self.tables:
            del self.tables[table_id]


tables = TableManager()

# A disconnected, unseated client keeps its reclaim rights this long; after
# that its ClientConn (and any AI seat reservation) is swept.
STALE_CLIENT_SECONDS = 30 * 60


def prune_table_state(table: Table) -> None:
    """Sweep bookkeeping that only grows during a table's lifetime.

    Caller must hold ``table.state_lock``. Removes clients that disconnected
    long ago without a seat, their AI-seat reservations, and AI occupants no
    longer referenced by a seat or reservation.
    """
    now = time.time()
    for cid, conn in list(table.clients.items()):
        if (
            conn.websocket is None
            and conn.seat is None
            and conn.disconnected_at is not None
            and now - conn.disconnected_at > STALE_CLIENT_SECONDS
        ):
            del table.clients[cid]
            table.reserved_ai_by_human.pop(cid, None)

    seated = {occ for occ in table.seats.values() if occ}
    reserved = set(table.reserved_ai_by_human.values())
    for occ_id, occ in list(table.occupants.items()):
        if occ.is_ai and occ_id not in seated and occ_id not in reserved:
            del table.occupants[occ_id]
