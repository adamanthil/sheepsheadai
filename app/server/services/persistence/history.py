"""A player's hand history: their finished, scored hands, newest first.

Card points taken are derived from the recorded tricks rather than stored,
mirroring the engine's Game.get_final_picker_points /
get_final_defender_points: each row takes the points of the tricks it won
(the leaster blind and the under are already folded into trick.points);
the picker's team is the picker plus the partner unless alone; the bury
goes to the picker's team if it took any points, else to the defenders. A
leaster row reports only its own points.
"""

from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime
from typing import Optional
from uuid import UUID

import asyncpg

from server.services.persistence.stats import (
    ABANDONED_SQL,
    AI_ASSISTED_SQL,
    COUNTED_SCORE_SQL,
)

PAGE_SIZE = 25

# The row owner is on the picker's team. The engine's JD-buried case marks
# the picker as partner too; the team is the same set either way.
_ON_PICKER_TEAM = (
    "(gp.is_picker IS TRUE OR (gp.is_partner IS TRUE AND g.is_alone IS NOT TRUE))"
)

_PAGE_SQL = f"""
WITH page AS (
    SELECT gp.game_player_id, gp.game_id, g.time_closed, t.name AS table_name,
           g.score_multiplier AS multiplier,
           gp.score, {COUNTED_SCORE_SQL} AS counted_score,
           {AI_ASSISTED_SQL} AS ai_assisted, {ABANDONED_SQL} AS abandoned,
           CASE WHEN g.is_leaster IS TRUE THEN 'leaster'
                WHEN gp.is_picker IS TRUE THEN 'picker'
                WHEN gp.is_partner IS TRUE THEN 'partner'
                ELSE 'defender' END AS role,
           g.is_alone IS TRUE AS alone,
           g.is_leaster IS TRUE AS is_leaster
    FROM game_player gp
    JOIN game g ON g.game_id = gp.game_id
    JOIN game_table t ON t.game_table_id = g.game_table_id
    WHERE gp.player_id = $1
      AND g.time_closed IS NOT NULL
      AND gp.score IS NOT NULL
      AND ($2::timestamp IS NULL OR (g.time_closed, g.game_id) < ($2, $3::uuid))
    ORDER BY g.time_closed DESC, g.game_id DESC
    LIMIT $4
),
seat AS (
    -- Every row of each hand on the page, with the card points it took.
    SELECT gp.game_id, gp.game_player_id, gp.player_id,
           {_ON_PICKER_TEAM} AS on_picker_team,
           (SELECT coalesce(sum(tr.points), 0) FROM trick tr
            WHERE tr.winning_player_id = gp.game_player_id)::int AS taken
    FROM game_player gp
    JOIN game g ON g.game_id = gp.game_id
    WHERE gp.game_id IN (SELECT game_id FROM page)
),
team AS (
    SELECT s.game_id,
           sum(s.taken) FILTER (WHERE s.on_picker_team)::int AS picker_raw,
           sum(s.taken) FILTER (WHERE NOT s.on_picker_team)::int AS defender_raw,
           count(*) FILTER (WHERE s.player_id <> $1)::int AS opponent_humans,
           count(*) FILTER (WHERE s.player_id IS NULL)::int AS opponent_ai
    FROM seat s
    GROUP BY s.game_id
),
bury AS (
    SELECT g.game_id, coalesce(sum(c.points), 0)::int AS points
    FROM game g
    JOIN cardset_card cc ON cc.cardset_id = g.bury_id
    JOIN card c ON c.card_id = cc.card_id
    WHERE g.game_id IN (SELECT game_id FROM page)
    GROUP BY g.game_id
)
SELECT p.*, s.on_picker_team, s.taken, tm.picker_raw, tm.defender_raw,
       tm.opponent_humans, tm.opponent_ai, coalesce(b.points, 0) AS bury
FROM page p
JOIN seat s ON s.game_player_id = p.game_player_id
JOIN team tm ON tm.game_id = p.game_id
LEFT JOIN bury b ON b.game_id = p.game_id
ORDER BY p.time_closed DESC, p.game_id DESC
"""


class BadCursor(ValueError):
    pass


def encode_cursor(time_closed: datetime, game_id: UUID) -> str:
    raw = f"{time_closed.isoformat()}|{game_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        stamp, game_id = base64.urlsafe_b64decode(padded).decode().split("|")
        return datetime.fromisoformat(stamp), UUID(game_id)
    except (binascii.Error, UnicodeDecodeError, ValueError) as e:
        raise BadCursor(cursor) from e


def _points_taken(row: asyncpg.Record) -> int:
    if row["is_leaster"]:
        return row["taken"]
    picker_raw = row["picker_raw"] or 0
    defender_raw = row["defender_raw"] or 0
    if row["on_picker_team"]:
        return picker_raw + (row["bury"] if picker_raw else 0)
    return defender_raw + (0 if picker_raw else row["bury"])


def _hand(row: asyncpg.Record) -> dict:
    return {
        "game_id": row["game_id"],
        # Stored as naive UTC (the database runs in UTC).
        "time_closed": row["time_closed"].replace(tzinfo=UTC),
        "table_name": row["table_name"],
        "role": row["role"],
        "alone": row["alone"],
        "multiplier": row["multiplier"],
        "score": row["score"],
        "counted_score": row["counted_score"],
        "points_taken": _points_taken(row),
        "points_scope": "own" if row["is_leaster"] else "team",
        "ai_assisted": row["ai_assisted"],
        "abandoned": row["abandoned"],
        "opponents": {"humans": row["opponent_humans"], "ai": row["opponent_ai"]},
    }


async def hand_history(
    pool: asyncpg.Pool, player_id: UUID, before: Optional[str] = None
) -> dict:
    """One page of ``player_id``'s hands, newest first, and the cursor for
    the next page (None on the last). Raises BadCursor for a bad ``before``."""
    stamp, game_id = decode_cursor(before) if before else (None, None)
    rows = await pool.fetch(_PAGE_SQL, player_id, stamp, game_id, PAGE_SIZE + 1)
    page = rows[:PAGE_SIZE]
    next_cursor = (
        encode_cursor(page[-1]["time_closed"], page[-1]["game_id"])
        if len(rows) > PAGE_SIZE
        else None
    )
    return {"hands": [_hand(r) for r in page], "next_cursor": next_cursor}
