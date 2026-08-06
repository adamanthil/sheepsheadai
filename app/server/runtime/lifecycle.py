from __future__ import annotations

import asyncio
import json
import logging

from server.realtime.broadcast import broadcast_table_event
from server.runtime.tables import Table, tables
from server.services.persistence.games import close_game_table
from server.services.persistence.pool import get_db_pool

# Set on SIGTERM (deploy/restart): new tables/games are refused with a 503
# while in-flight hands get a heads-up broadcast before the process exits.
_draining = False


def set_draining() -> None:
    global _draining
    _draining = True


def is_draining() -> bool:
    return _draining


async def close_table(table: Table, reason: str = "closed") -> None:
    """Gracefully close a table: cancel AI, notify clients, close websockets, and remove from manager."""
    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    # The idle path calls this from *inside* autoclose_task. Cancelling that
    # task here would throw CancelledError into this coroutine at the next
    # await and strand the table in the registry forever.
    current = asyncio.current_task()
    if (
        table.autoclose_task
        and table.autoclose_task is not current
        and not table.autoclose_task.done()
    ):
        table.autoclose_task.cancel()
    for cid, task in list(table.disconnect_tasks.items()):
        try:
            if task and not task.done():
                task.cancel()
        except Exception:
            logging.debug(
                "failed to cancel disconnect task for client %s on table %s",
                cid,
                table.id,
            )
        finally:
            table.disconnect_tasks.pop(cid, None)
    table.status = "finished"
    try:
        try:
            await broadcast_table_event(
                table, {"type": "table_closed", "reason": reason, "tableId": table.id}
            )
        except Exception:
            logging.debug("failed to broadcast table_closed for table %s", table.id)
        for cid, conn in list(table.clients.items()):
            ws = conn.websocket
            if not ws:
                continue
            try:
                await ws.send_text(
                    json.dumps(
                        {"type": "table_closed", "reason": reason, "tableId": table.id}
                    )
                )
                await ws.close()
            except Exception:
                logging.debug(
                    "failed to close websocket for client %s on table %s", cid, table.id
                )
                conn.websocket = None
        await close_game_table(get_db_pool(), table.id)
    finally:
        # Registry removal is the one step that must not be skippable: a table
        # left here is unreachable by every autoclose trigger (they all key off
        # websocket edges on a table nobody can reach) and survives until the
        # process restarts.
        try:
            tables.delete_table(table.id)
        except Exception:
            logging.exception("failed deleting table %s", table.id)


def schedule_autoclose_if_no_humans(table: Table, delay_seconds: float = 30.0) -> None:
    """If there are no human players connected, schedule an auto-close after delay."""

    def any_human_connected() -> bool:
        for cid, conn in table.clients.items():
            if conn.websocket is not None:
                return True
        return False

    if any_human_connected():
        if table.autoclose_task and not table.autoclose_task.done():
            table.autoclose_task.cancel()
        return

    async def _auto():
        try:
            await asyncio.sleep(delay_seconds)
            for _cid, _conn in table.clients.items():
                if _conn.websocket is not None:
                    return
            await close_table(table, reason="idle_all_disconnected")
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception("autoclose task failed for table %s", table.id)

    if table.autoclose_task and not table.autoclose_task.done():
        table.autoclose_task.cancel()
    table.autoclose_task = asyncio.create_task(_auto())
