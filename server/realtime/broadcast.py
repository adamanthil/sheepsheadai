from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import WebSocketDisconnect

from server.runtime.tables import (
    Table,
    _json_default,
    build_player_state,
    get_actor_seat,
    get_valid_action_ids_for_seat,
)


async def broadcast_table_event(table: Table, payload: Dict[str, Any]) -> None:
    """Broadcast any table-related event payload to all connected clients."""
    msg_txt = json.dumps(payload, default=_json_default)
    for cid, conn in list(table.clients.items()):
        ws = conn.websocket
        if not ws:
            continue
        try:
            await ws.send_text(msg_txt)
        except WebSocketDisconnect:
            conn.websocket = None
        except Exception:
            logging.exception(
                "broadcast_table_event send failed for table %s client %s",
                table.id,
                cid,
            )


async def broadcast_table_update(table: Table) -> None:
    """Send per-client table_update events, each including the client's isHost status."""
    table_dict = table.to_public_dict()
    for cid, conn in list(table.clients.items()):
        ws = conn.websocket
        if not ws:
            continue
        payload = {
            "type": "table_update",
            "table": table_dict,
            "isHost": cid == table.host_client_id,
        }
        try:
            await ws.send_text(json.dumps(payload, default=_json_default))
        except WebSocketDisconnect:
            conn.websocket = None
        except Exception:
            logging.exception(
                "broadcast_table_update send failed for table %s client %s",
                table.id,
                cid,
            )


async def broadcast_table_state(table: Table) -> None:
    """Send each connected human client their own masked state + valid actions."""
    if not table.game:
        return
    actor_seat = get_actor_seat(table)
    for cid, conn in list(table.clients.items()):
        if not conn.websocket:
            continue
        if not conn.seat:
            continue
        player = table.game.players[conn.seat - 1]
        payload = build_player_state(player)
        valid_actions = get_valid_action_ids_for_seat(table, conn.seat)
        msg = {
            "type": "state",
            "table": table.to_public_dict(),
            "yourSeat": conn.seat,
            "actorSeat": actor_seat,
            "isHost": cid == table.host_client_id,
            "state": payload["state"],
            "view": payload["view"],
            "valid_actions": valid_actions if conn.seat == actor_seat else [],
        }
        try:
            await conn.websocket.send_text(json.dumps(msg, default=_json_default))
        except WebSocketDisconnect:
            conn.websocket = None
        except Exception as exc:
            logging.exception(
                "broadcast send failed for table %s client %s: %s", table.id, cid, exc
            )
