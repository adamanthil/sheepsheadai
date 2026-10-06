"""Host handoff: when the host leaves, the table gets a new one.

Without this a table whose host disconnected could never be started,
redealt, or closed again by anyone left at it.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from server.realtime.broadcast import broadcast_table_state, broadcast_table_update
from server.realtime.chat import add_chat_message, broadcast_chat_append
from server.runtime.models import ClientConn, Table
from server.runtime.tasks import spawn

# Same grace as a disconnected player's seat, so a page refresh keeps host.
HOST_HANDOFF_GRACE_SECONDS = 10.0


def _successor(table: Table, excluding: set[str]) -> Optional[ClientConn]:
    """The connected human who has been at the table longest, preferring
    one with a seat. ``table.clients`` keeps join order."""
    connected = [
        c
        for cid, c in table.clients.items()
        if c.connected and cid != table.host_client_id and cid not in excluding
    ]
    seated = [c for c in connected if c.seat is not None]
    return (seated or connected or [None])[0]


def pass_host(table: Table, excluding: set[str]) -> Optional[ClientConn]:
    """Make the successor host now, passing over the clients in
    ``excluding``; None (and no change) when there is nobody to take it.
    Caller holds ``table.state_lock``."""
    successor = _successor(table, excluding)
    if successor is not None:
        table.host_client_id = successor.client_id
    return successor


async def announce_new_host(table: Table, successor: ClientConn) -> None:
    msg_dict = await add_chat_message(
        table, "system", "is now the host", author=successor.display_name
    )
    await broadcast_chat_append(table, msg_dict)
    await broadcast_table_update(table)
    await broadcast_table_state(table)


def cancel_host_handoff(table: Table) -> None:
    task = table.host_handoff_task
    if task and not task.done():
        task.cancel()
    table.host_handoff_task = None


def schedule_host_handoff(table: Table) -> None:
    """Hand host to a successor once the grace runs out, unless the host
    is back by then. Idempotent while a handoff is pending."""
    if table.host_handoff_task and not table.host_handoff_task.done():
        return

    async def _runner() -> None:
        try:
            await asyncio.sleep(HOST_HANDOFF_GRACE_SECONDS)
            async with table.state_lock:
                host = table.clients.get(table.host_client_id or "")
                if host is not None and host.connected:
                    return
                successor = pass_host(table, excluding=set())
                if successor is None:
                    # Nobody to hand to; the next connection re-arms this.
                    return
            await announce_new_host(table, successor)
        except asyncio.CancelledError:
            return
        finally:
            if table.host_handoff_task is asyncio.current_task():
                table.host_handoff_task = None

    table.host_handoff_task = spawn(_runner(), f"host-handoff:{table.id}")


def host_is_away(table: Table) -> bool:
    """True when the table has a host who has no tab open. A table whose
    host has not joined yet has no one to hand off from."""
    if not table.host_client_id:
        return False
    host = table.clients.get(table.host_client_id)
    return host is None or not host.connected
