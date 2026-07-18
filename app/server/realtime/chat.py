from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket

from server.runtime.tables import ClientConn, Table, _json_default
from server.realtime.broadcast import broadcast_table_event
from sheepshead import CARD_FULL_NAMES

CHAT_MAX_LEN = 500
_CHAT_RATE_LIMIT = 5  # max messages
_CHAT_RATE_WINDOW = 5.0  # seconds


def is_chat_rate_limited(conn: ClientConn) -> bool:
    """Return True and log if the client is sending chat messages too fast."""
    now = time.monotonic()
    while conn.chat_timestamps and conn.chat_timestamps[0] < now - _CHAT_RATE_WINDOW:
        conn.chat_timestamps.popleft()
    if len(conn.chat_timestamps) >= _CHAT_RATE_LIMIT:
        logging.debug("chat rate limit exceeded for client %s", conn.client_id)
        return True
    conn.chat_timestamps.append(now)
    return False


async def add_chat_message(
    table: Table, msg_type: str, body: str, author: Optional[str] = None
) -> Dict[str, Any]:
    """Add a chat message to the table's chat log and return the message dict."""
    msg_id = str(uuid.uuid4())
    msg_dict: Dict[str, Any] = {
        "id": msg_id,
        "table_id": table.id,
        "type": msg_type,
        "author": author,
        "body": body,
        "timestamp": time.time(),
    }
    table.chat_log.append(msg_dict)
    return msg_dict


async def broadcast_chat_append(table: Table, msg_dict: Dict[str, Any]) -> None:
    """Broadcast a chat:append event to all connected clients."""
    await broadcast_table_event(
        table,
        {
            "type": "chat:append",
            "message": msg_dict,
        },
    )


async def emit_bid_chat_message(table: Table, action_str: str, display_name: str) -> None:
    """Post + broadcast a system chat message for a bid/partner-call action.

    Covers PICK / PASS / ALONE / JD PARTNER / CALL <card> [UNDER]. No-op for
    any other action string (e.g. PLAY actions), so call sites can invoke
    this unconditionally after resolving an action.
    """
    if action_str == "PICK":
        body = f"{display_name} picked"
    elif action_str == "PASS":
        body = f"{display_name} passed"
    elif action_str == "ALONE":
        body = f"{display_name} goes alone"
    elif action_str == "JD PARTNER":
        body = f"{display_name} chose JD partner"
    elif action_str.startswith("CALL "):
        parts = action_str.split()
        called_card = parts[1] if len(parts) > 1 else ""
        under = "under" if len(parts) > 2 and parts[2] == "UNDER" else ""
        card_display = CARD_FULL_NAMES.get(called_card, called_card)
        body = f"{display_name} calls {card_display}"
        if under:
            body += " under"
    else:
        return

    msg_dict = await add_chat_message(table, "system", body)
    await broadcast_chat_append(table, msg_dict)


async def send_chat_init(table: Table, websocket: WebSocket) -> None:
    """Send the full chat history to a newly connected client."""
    messages = list(table.chat_log)
    await websocket.send_text(
        json.dumps(
            {
                "type": "chat:init",
                "messages": messages,
            },
            default=_json_default,
        )
    )
