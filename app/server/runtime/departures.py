"""Leaving a table on purpose: at once, or once the hand in play ends.

A player who leaves says so, so their seat goes to the AI straight away:
a socket that merely drops is held for them (runtime.seating), and the
table would wait out that grace on someone who isn't coming back.

A player who leaves mid-hand forfeits it to the AI; one who leaves between
hands can be dealt into the next one before they get the chance if the host
deals it first. Asking ahead closes that gap: the server unseats them the
moment the hand ends, before the next hand can reach their seat.

A passed-out doublers deal is not the end of the hand -- the redeal is the
same hand at double the stake -- so it does not set anyone off.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocketDisconnect

from server.realtime.broadcast import broadcast_table_update
from server.realtime.chat import add_chat_message, broadcast_chat_append
from server.runtime.host import announce_new_host, pass_host
from server.runtime.models import ClientConn, Table
from server.runtime.occupants import give_seat_to_ai
from server.runtime.views import json_default

# Close code for a player's tabs once they have left; the client treats it
# as terminal, like a kick.
LEFT_WS_CODE = 4410


async def leave_now(table: Table, conn: ClientConn) -> int | None:
    """Unseat ``conn`` for good, hand their seat to the AI and close their
    tabs; the seat it went to, if any. The caller sets the AI moving on
    that seat (runtime.ai_loop imports this module, by way of lifecycle)."""
    async with table.state_lock:
        if table.clients.get(conn.client_id) is not conn:
            return None
        departed, new_host = _unseat(table, [conn])
    # Every tab goes home: the one that asked is on its way already.
    await _see_off(table, departed, new_host, "left", {"type": "left"})
    return departed[0][1]


async def see_off_departures(table: Table) -> None:
    """Unseat everyone who asked to leave after this hand and close their
    tabs. Call once the hand has finished and its result is recorded."""
    async with table.state_lock:
        leaving = [c for c in table.clients.values() if c.leave_after_hand]
        if not leaving:
            return
        for conn in leaving:
            conn.leave_after_hand = False
        departed, new_host = _unseat(table, leaving)
    # The table rides along: the leaver is gone before the finished hand is
    # broadcast, and their final scores card needs its result.
    farewell = {
        "type": "left_after_hand",
        "tableId": table.id,
        "table": table.to_public_dict(),
    }
    await _see_off(table, departed, new_host, "left after the hand", farewell)


def _unseat(
    table: Table, leaving: list[ClientConn]
) -> tuple[list[tuple[ClientConn, int | None]], ClientConn | None]:
    """Give the leavers' seats to the AI, pass host on if it is leaving and
    forget them. Caller holds ``table.state_lock``."""
    departed = [(conn, give_seat_to_ai(table, conn)) for conn in leaving]
    leaving_ids = {c.client_id for c in leaving}
    new_host = (
        pass_host(table, excluding=leaving_ids)
        if table.host_client_id in leaving_ids
        else None
    )
    for conn in leaving:
        del table.clients[conn.client_id]
    return departed, new_host


async def _see_off(
    table: Table,
    departed: list[tuple[ClientConn, int | None]],
    new_host: ClientConn | None,
    event: str,
    farewell: dict[str, Any],
) -> None:
    """Close the leavers' tabs and tell the table they went."""
    for conn, seat in departed:
        await _close_tabs(table, conn, farewell)
        note = event
        if seat is not None:
            note += f". Seat {seat} went to the AI."
        msg_dict = await add_chat_message(
            table, "system", note, author=conn.display_name
        )
        await broadcast_chat_append(table, msg_dict)
    if new_host is not None:
        await announce_new_host(table, new_host)
    else:
        await broadcast_table_update(table)


async def _close_tabs(table: Table, conn: ClientConn, farewell: dict[str, Any]) -> None:
    sockets = list(conn.sockets)
    conn.sockets.clear()
    msg = json.dumps(farewell, default=json_default)
    for ws in sockets:
        try:
            await ws.send_text(msg)
            await ws.close(code=LEFT_WS_CODE)
        # Same failures close_table tolerates: the tab is already gone.
        except WebSocketDisconnect, OSError, RuntimeError:
            logging.debug(
                "failed to close websocket for client %s on table %s",
                conn.client_id,
                table.id,
            )
