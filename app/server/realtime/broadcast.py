from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import WebSocketDisconnect

from server.runtime.tables import (
    ClientConn,
    Table,
    _json_default,
    build_player_state,
    get_actor_seat,
    get_valid_action_ids_for_seat,
)


async def send_to_client(table: Table, conn: ClientConn, text: str) -> None:
    """Fan one payload out to every tab the client has open.

    A socket that fails to send is evicted rather than logged-and-kept: a dead
    entry left in the set keeps the player reading as connected, which
    suppresses the idle autoclose (server.runtime.lifecycle) and strands the
    table with no players.
    """
    for ws in list(conn.sockets):
        try:
            await ws.send_text(text)
        except WebSocketDisconnect:
            conn.sockets.discard(ws)
        except Exception:
            conn.sockets.discard(ws)
            logging.exception(
                "send failed for table %s client %s", table.id, conn.client_id
            )


async def broadcast_table_event(table: Table, payload: Dict[str, Any]) -> None:
    """Broadcast any table-related event payload to all connected clients."""
    msg_txt = json.dumps(payload, default=_json_default)
    for conn in list(table.clients.values()):
        await send_to_client(table, conn, msg_txt)


async def broadcast_table_update(table: Table) -> None:
    """Send per-client table_update events, each including the client's isHost status."""
    table_dict = table.to_public_dict()
    for cid, conn in list(table.clients.items()):
        payload = {
            "type": "table_update",
            "table": table_dict,
            "isHost": cid == table.host_client_id,
        }
        await send_to_client(table, conn, json.dumps(payload, default=_json_default))


async def broadcast_table_state(table: Table) -> None:
    """Send each connected human client their own masked state + valid actions.

    State is masked by ``conn.seat``, which is per-client, so every tab of one
    player receives identical content -- multi-tab reveals nothing a single
    tab would not.
    """
    if not table.game:
        return
    actor_seat = get_actor_seat(table)
    for cid, conn in list(table.clients.items()):
        if not conn.connected:
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
        await send_to_client(table, conn, json.dumps(msg, default=_json_default))
