from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.api.auth import resolve_player
from server.realtime.broadcast import (
    broadcast_table_event,
    broadcast_table_state,
    broadcast_table_update,
)
from server.realtime.chat import (
    CHAT_MAX_LEN,
    add_chat_message,
    broadcast_chat_append,
    is_chat_rate_limited,
    send_chat_init,
)
from server.runtime.ai_loop import schedule_ai_turns
from server.runtime.lifecycle import schedule_autoclose_if_no_humans
from server.runtime.seating import (
    cancel_disconnect_task,
    find_seat_of_occupant,
    schedule_ai_replacement_for_disconnected_human,
)
from server.runtime.tables import _json_default, tables

router = APIRouter()


_CLIENT_SUBPROTO_PREFIX = "sheepshead.client."
_TOKEN_SUBPROTO_PREFIX = "sheepshead.token."

# DoS backstop: one IP may hold at most this many concurrent sockets. Counted
# only for connections that pass validation; check+increment happen with no
# await in between, so the asyncio event loop makes them atomic.
MAX_SOCKETS_PER_IP = 20
_sockets_by_ip: dict[str, int] = {}

# One player may have the table open in this many tabs at once. Each socket
# multiplies the fan-out of every broadcast, so this is the backstop that
# used to be implicit in "one socket per ClientConn". Rejected connections
# close with 4429, which the client treats as terminal (no reconnect storm).
MAX_SOCKETS_PER_CLIENT = 4


def _release_ip_slot(ip: str) -> None:
    remaining = _sockets_by_ip.get(ip, 1) - 1
    if remaining <= 0:
        _sockets_by_ip.pop(ip, None)
    else:
        _sockets_by_ip[ip] = remaining


@router.websocket("/ws/table/{table_id}")
async def table_ws(websocket: WebSocket, table_id: str):
    # client_id arrives via Sec-WebSocket-Protocol rather than the URL so it
    # does not end up in proxy access logs / browser history.
    offered = websocket.scope.get("subprotocols", []) or []
    client_id: str | None = None
    token: str | None = None
    chosen_subproto: str | None = None
    for sp in offered:
        if not isinstance(sp, str):
            continue
        if sp.startswith(_CLIENT_SUBPROTO_PREFIX) and client_id is None:
            client_id = sp[len(_CLIENT_SUBPROTO_PREFIX) :]
            # Echo the client entry; the token must never appear in the
            # response headers.
            chosen_subproto = sp
        elif sp.startswith(_TOKEN_SUBPROTO_PREFIX) and token is None:
            token = sp[len(_TOKEN_SUBPROTO_PREFIX) :]

    if not client_id or not token:
        await websocket.accept()
        await websocket.close(code=4401)
        return

    identity = await resolve_player(token)
    if identity is None:
        await websocket.accept(subprotocol=chosen_subproto)
        await websocket.close(code=4401)
        return

    try:
        table = tables.get_table(table_id)
    except KeyError:
        await websocket.accept(subprotocol=chosen_subproto)
        await websocket.close(code=4404)
        return

    conn_check = table.clients.get(client_id)
    if conn_check is None or conn_check.player_id != str(identity.id):
        await websocket.accept(subprotocol=chosen_subproto)
        await websocket.close(code=4403)
        return

    ip = websocket.client.host if websocket.client else "unknown"
    if _sockets_by_ip.get(ip, 0) >= MAX_SOCKETS_PER_IP:
        await websocket.accept(subprotocol=chosen_subproto)
        await websocket.close(code=4429)
        return
    _sockets_by_ip[ip] = _sockets_by_ip.get(ip, 0) + 1

    try:
        await _serve_connection(websocket, table, client_id, chosen_subproto)
    finally:
        _release_ip_slot(ip)


async def _serve_connection(
    websocket: WebSocket, table, client_id: str, chosen_subproto: str | None
):
    await websocket.accept(subprotocol=chosen_subproto)

    conn = table.clients[client_id]
    reclaimed_seat = None
    async with table.state_lock:
        # Check and add under one lock hold so two tabs opening at once
        # cannot both pass the cap.
        over_cap = len(conn.sockets) >= MAX_SOCKETS_PER_CLIENT
        if not over_cap:
            # Reconnect bookkeeping is a 0->1 edge, not a per-socket event:
            # re-running it for a second tab would cancel nothing useful and
            # re-broadcast "reclaimed seat N" to the whole table on every
            # tab the player opens.
            first_socket = not conn.sockets
            conn.sockets.add(websocket)
            conn.disconnected_at = None
            if first_socket:
                # Cancel any pending replacement and attempt to reclaim
                # reserved AI seat if needed.
                cancel_disconnect_task(table, client_id)
                ai_id = table.reserved_ai_by_human.get(client_id)
                if ai_id:
                    seat_idx = find_seat_of_occupant(table, ai_id)
                    if seat_idx:
                        table.seats[seat_idx] = client_id
                        conn.seat = seat_idx
                        reclaimed_seat = seat_idx
    if over_cap:
        logging.info(
            "client %s on table %s refused: %d concurrent sockets",
            client_id,
            table.id,
            MAX_SOCKETS_PER_CLIENT,
        )
        await websocket.close(code=4429)
        return

    # Everything past the socket joining the set runs under this try/finally:
    # a client that vanishes during the initial handshake burst
    # (send_chat_init, the isHost table_update) must not leave a dead socket
    # on the conn. Such a phantom reads as a connected human forever, which
    # suppresses the idle autoclose and keeps the table alive with no players.
    try:
        if reclaimed_seat is not None:
            await broadcast_table_event(
                table,
                {
                    "type": "lobby_event",
                    "message": f"{conn.display_name} reconnected and reclaimed seat {reclaimed_seat}",
                    "table": table.to_public_dict(),
                },
            )
            await broadcast_table_update(table)
            schedule_ai_turns(table)
        # On connect, cancel any pending autoclose
        schedule_autoclose_if_no_humans(table)

        await broadcast_table_state(table)
        await send_chat_init(table, websocket)
        # Send a per-client table_update so the client knows their isHost status immediately
        await websocket.send_text(
            json.dumps(
                {
                    "type": "table_update",
                    "table": table.to_public_dict(),
                    "isHost": client_id == table.host_client_id,
                },
                default=_json_default,
            )
        )

        while True:
            try:
                raw_text = await websocket.receive_text()
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, dict) and data.get("type") == "chat:send":
                        message_text = data.get("message", "").strip()
                        if message_text and len(message_text) <= CHAT_MAX_LEN:
                            if not is_chat_rate_limited(conn):
                                msg_dict = await add_chat_message(
                                    table,
                                    "player",
                                    message_text,
                                    author=conn.display_name,
                                )
                                await broadcast_chat_append(table, msg_dict)
                except json.JSONDecodeError:
                    pass
            except ValueError:
                logging.exception(
                    "Received malformed text over ws connection from client %s",
                    client_id,
                )
    except WebSocketDisconnect:
        pass
    finally:
        c = table.clients.get(client_id)
        if c:
            c.sockets.discard(websocket)
            # Only the last tab closing means the player left: handing the
            # seat to an AI because they closed one of several tabs would
            # take the game away from someone still sitting at it.
            if not c.sockets:
                c.disconnected_at = time.time()
                try:
                    if c.seat is not None:
                        schedule_ai_replacement_for_disconnected_human(table, client_id)
                except Exception:
                    logging.exception(
                        "failed to schedule AI replacement for client %s on table %s",
                        client_id,
                        table.id,
                    )
        schedule_autoclose_if_no_humans(table)
