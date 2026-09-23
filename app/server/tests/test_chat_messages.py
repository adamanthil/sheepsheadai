"""System chat lines keep who they're about apart from the event text."""

from __future__ import annotations

from server.realtime.chat import emit_bid_chat_message
from server.runtime.tables import Table


async def test_bid_line_carries_the_actor_separately():
    table = Table(id="t", name="chat")
    # A display name made to read like an event stays just a name.
    await emit_bid_chat_message(table, "PASS", "Dan picked")
    await emit_bid_chat_message(table, "PICK", "Dan", is_ai=True)

    spoof, ai = table.chat_log
    assert (spoof["author"], spoof["author_is_ai"], spoof["body"]) == (
        "Dan picked",
        False,
        "passed",
    )
    assert (ai["author"], ai["author_is_ai"], ai["body"]) == ("Dan", True, "picked")
