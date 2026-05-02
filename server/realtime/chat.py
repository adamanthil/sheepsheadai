from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import WebSocket

from server.runtime.tables import Table, _json_default
from server.realtime.broadcast import broadcast_table_event


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
    await broadcast_table_event(table, {
        "type": "chat:append",
        "message": msg_dict,
    })


async def send_chat_init(table: Table, websocket: WebSocket) -> None:
    """Send the full chat history to a newly connected client."""
    messages = list(table.chat_log)
    await websocket.send_text(json.dumps({
        "type": "chat:init",
        "messages": messages,
    }, default=_json_default))
