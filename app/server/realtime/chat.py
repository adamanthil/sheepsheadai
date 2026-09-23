from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket

from server.realtime.broadcast import broadcast_table_event
from server.runtime.tables import ClientConn, Table, json_default
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
    table: Table,
    msg_type: str,
    body: str,
    author: Optional[str] = None,
    author_id: Optional[str] = None,
    author_is_ai: bool = False,
) -> Dict[str, Any]:
    """Add a chat message to the table's chat log and return the message dict.

    On a player message ``author`` wrote it; ``author_id`` is their client
    id, so other clients can act on them (mute, or the host removing them).
    On a system message ``author`` is who the event is about, kept out of
    ``body`` so clients render the name apart from the event text and a
    display name can never pass as part of the event; ``author_is_ai``
    marks an AI occupant.
    """
    msg_id = str(uuid.uuid4())
    msg_dict: Dict[str, Any] = {
        "id": msg_id,
        "table_id": table.id,
        "type": msg_type,
        "author": author,
        "author_id": author_id,
        "author_is_ai": author_is_ai,
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


async def emit_bid_chat_message(
    table: Table, action_str: str, display_name: str, is_ai: bool = False
) -> None:
    """Post + broadcast a system chat message for a bid/partner-call action.

    Covers PICK / PASS / ALONE / JD PARTNER / CALL <card> [UNDER]. No-op for
    any other action string (e.g. PLAY actions), so call sites can invoke
    this unconditionally after resolving an action.
    """
    if action_str == "PICK":
        body = "picked"
    elif action_str == "PASS":
        body = "passed"
    elif action_str == "ALONE":
        body = "goes alone"
    elif action_str == "JD PARTNER":
        body = "chose JD partner"
    elif action_str.startswith("CALL "):
        parts = action_str.split()
        called_card = parts[1] if len(parts) > 1 else ""
        under = "under" if len(parts) > 2 and parts[2] == "UNDER" else ""
        card_display = CARD_FULL_NAMES.get(called_card, called_card)
        body = f"calls {card_display}"
        if under:
            body += " under"
    else:
        return

    msg_dict = await add_chat_message(
        table, "system", body, author=display_name, author_is_ai=is_ai
    )
    await broadcast_chat_append(table, msg_dict)


async def emit_doubler_redeal_message(table: Table, multiplier: int) -> None:
    """Post + broadcast the system message for a passed-out doublers deal."""
    msg_dict = await add_chat_message(
        table,
        "system",
        f"All passed — hand thrown in, redealing at {multiplier}x",
    )
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
            default=json_default,
        )
    )
