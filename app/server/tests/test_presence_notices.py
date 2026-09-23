"""A player's join/leave notices are posted at most once a minute."""

from __future__ import annotations

import time

from server.realtime.chat import (
    PRESENCE_NOTICE_COOLDOWN_SECONDS,
    post_presence_notice,
)
from server.runtime.seating import replace_ai_with_human_and_reserve
from server.runtime.tables import ClientConn, Occupant, Table


def _conn(table: Table, cid: str, player: str) -> ClientConn:
    conn = ClientConn(client_id=cid, display_name=cid, player_id=player)
    table.clients[cid] = conn
    return conn


async def test_repeat_within_the_window_is_dropped():
    table = Table(id="t", name="n")
    pat = _conn(table, "pat", "p1")
    sam = _conn(table, "sam", "p2")

    await post_presence_notice(table, pat, "joined the table")
    await post_presence_notice(table, pat, "disconnected. Seat filled by AI.")
    await post_presence_notice(table, sam, "joined the table")

    assert [(m["author"], m["body"]) for m in table.chat_log] == [
        ("pat", "joined the table"),
        ("sam", "joined the table"),
    ]


async def test_notice_posts_again_once_the_window_passes():
    table = Table(id="t", name="n")
    pat = _conn(table, "pat", "p1")
    await post_presence_notice(table, pat, "joined the table")
    table.presence_notice_at["p1"] = (
        time.monotonic() - PRESENCE_NOTICE_COOLDOWN_SECONDS - 1
    )

    await post_presence_notice(table, pat, "disconnected. Seat filled by AI.")

    assert len(table.chat_log) == 2


async def test_rejoin_loop_through_seating_posts_once():
    # A player bouncing in and out of AI seats announces themselves once.
    table = Table(id="t", name="n")
    _conn(table, "pest", "p1")
    for seat in (1, 2, 3):
        table.occupants[f"ai{seat}"] = Occupant(
            id=f"ai{seat}", display_name="AI", is_ai=True
        )
        table.seats[seat] = f"ai{seat}"

    for seat in (1, 2, 3):
        await replace_ai_with_human_and_reserve(table, seat, "pest")
    if table.ai_task:
        table.ai_task.cancel()

    assert [m["body"] for m in table.chat_log] == ["joined and took seat 1"]
