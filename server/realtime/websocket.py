from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from server.realtime.broadcast import broadcast_table_event, broadcast_table_state, broadcast_table_update
from server.realtime.chat import CHAT_MAX_LEN, add_chat_message, broadcast_chat_append, is_chat_rate_limited, send_chat_init
from server.runtime.ai_loop import schedule_ai_turns
from server.runtime.lifecycle import schedule_autoclose_if_no_humans
from server.runtime.seating import (
    _cancel_disconnect_task,
    _find_seat_of_occupant,
    schedule_ai_replacement_for_disconnected_human,
)
from server.runtime.tables import _json_default, tables

router = APIRouter()


@router.websocket("/ws/table/{table_id}")
async def table_ws(websocket: WebSocket, table_id: str):
    await websocket.accept()
    params = websocket.query_params
    client_id = params.get("client_id")
    if not client_id:
        await websocket.close(code=4401)
        return
    try:
        table = tables.get_table(table_id)
    except KeyError:
        await websocket.close(code=4404)
        return

    if client_id not in table.clients:
        await websocket.close(code=4403)
        return

    conn = table.clients[client_id]
    conn.websocket = websocket
    # Cancel any pending replacement and attempt to reclaim reserved AI seat if needed
    _cancel_disconnect_task(table, client_id)
    ai_id = table.reserved_ai_by_human.get(client_id)
    if ai_id:
        seat_idx = _find_seat_of_occupant(table, ai_id)
        if seat_idx:
            table.seats[seat_idx] = client_id
            conn.seat = seat_idx
            await broadcast_table_event(table, {
                "type": "lobby_event",
                "message": f"{conn.display_name} reconnected and reclaimed seat {seat_idx}",
                "table": table.to_public_dict(),
            })
            await broadcast_table_update(table)
            schedule_ai_turns(table)
    # On connect, cancel any pending autoclose
    schedule_autoclose_if_no_humans(table)

    await broadcast_table_state(table)
    await send_chat_init(table, websocket)
    # Send a per-client table_update so the client knows their isHost status immediately
    await websocket.send_text(json.dumps({
        "type": "table_update",
        "table": table.to_public_dict(),
        "isHost": client_id == table.host_client_id,
    }, default=_json_default))

    try:
        while True:
            try:
                raw_text = await websocket.receive_text()
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, dict) and data.get("type") == "chat:send":
                        message_text = data.get("message", "").strip()
                        if message_text and len(message_text) <= CHAT_MAX_LEN:
                            if not is_chat_rate_limited(conn):
                                msg_dict = await add_chat_message(table, "player", message_text, author=conn.display_name)
                                await broadcast_chat_append(table, msg_dict)
                except json.JSONDecodeError:
                    pass
            except ValueError:
                logging.exception("Received malformed text over ws connection from client %s", client_id)
    except WebSocketDisconnect:
        pass
    finally:
        c = table.clients.get(client_id)
        if c:
            c.websocket = None
            try:
                if c.seat is not None:
                    schedule_ai_replacement_for_disconnected_human(table, client_id)
            except Exception:
                logging.exception("failed to schedule AI replacement for client %s on table %s", client_id, table.id)
        schedule_autoclose_if_no_humans(table)
