from __future__ import annotations

import asyncio
import json
import logging

from server.runtime.tables import Table, tables, _json_default
from server.realtime.broadcast import broadcast_table_event


async def close_table(table: Table, reason: str = "closed") -> None:
    """Gracefully close a table: cancel AI, notify clients, close websockets, and remove from manager."""
    if table.ai_task and not table.ai_task.done():
        table.ai_task.cancel()
    if table.autoclose_task and not table.autoclose_task.done():
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
