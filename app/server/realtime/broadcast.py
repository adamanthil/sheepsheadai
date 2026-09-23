from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import WebSocketDisconnect

from server.runtime.tables import (
    ClientConn,
    Table,
    build_player_state,
    get_actor_seat,
    get_valid_action_ids_for_seat,
    json_default,
)
from server.runtime.views import build_spectator_state


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
    msg_txt = json.dumps(payload, default=json_default)
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
        await send_to_client(table, conn, json.dumps(payload, default=json_default))


def _turn_seconds_left(table: Table, actor_seat: Optional[int]) -> Optional[float]:
    """What is left of the running turn timer, if it is for this turn."""
    key = table.turn_timer_key
    if key is None or table.turn_deadline is None or actor_seat is None:
        return None
    if key != (id(table.game), table.move_seq, actor_seat):
        return None
    return max(0.0, table.turn_deadline - time.monotonic())


async def broadcast_table_state(table: Table) -> None:
    """Send each connected client their own masked state + valid actions.

    State is masked by ``conn.seat``, which is per-client, so every tab of one
    player receives identical content -- multi-tab reveals nothing a single
    tab would not. Unseated clients (spectators) get the public view with
    ``yourSeat`` null.
    """
    if not table.game:
        return
    actor_seat = get_actor_seat(table)
    seconds_left = _turn_seconds_left(table, actor_seat)
    spectator_payload = None
    for cid, conn in list(table.clients.items()):
        if not conn.connected:
            continue
        if conn.seat:
            player = table.game.players[conn.seat - 1]
            payload = build_player_state(player, table.score_multiplier)
            valid_actions = get_valid_action_ids_for_seat(table, conn.seat)
        else:
            if spectator_payload is None:
                spectator_payload = build_spectator_state(
                    table.game, table.score_multiplier
                )
            payload = spectator_payload
            valid_actions = []
        msg = {
            "type": "state",
            "table": table.to_public_dict(),
            "yourSeat": conn.seat,
            "actorSeat": actor_seat,
            "isHost": cid == table.host_client_id,
            "state": payload["state"],
            "view": payload["view"],
            "valid_actions": valid_actions if conn.seat == actor_seat else [],
            "turnSecondsLeft": seconds_left,
        }
        await send_to_client(table, conn, json.dumps(msg, default=json_default))
