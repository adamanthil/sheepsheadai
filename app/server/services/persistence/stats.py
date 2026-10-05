"""Per-player hand statistics and the public leaderboard.

Everything derives from game_player rows of finished hands. A hand counts
once it is closed and scored, which leaves out hands still in play and
passed-out doublers deals (closed, but never scored). Scores already
include the doublers stake. Hands the AI finished for a player (timeouts,
disconnects) count as theirs, so walking away can't hide a loss.

Nor can it bank a win: a hand is abandoned when the AI made more than
ABANDON_AI_ACTIONS of the player's decisions in it (excused moves aside --
a kick, a host close, a restart). An abandoned hand's loss counts in full,
but a positive score counts as 0. The rule is applied here, at query time,
so changing the threshold re-grades all history consistently.
"""

from __future__ import annotations

import math
import time
from typing import Literal, Optional
from uuid import UUID

import asyncpg

SortKey = Literal["total", "sph", "win_pct", "pick_pct", "hands"]
SORT_KEYS: tuple[SortKey, ...] = ("total", "sph", "win_pct", "pick_pct", "hands")

# More AI decisions than this in a hand, not excused, abandon it.
ABANDON_AI_ACTIONS = 2

# SQL over a game_player row aliased gp.
AI_ASSISTED_SQL = "(gp.ai_actions > 0)"
ABANDONED_SQL = f"(gp.ai_actions - gp.ai_actions_excused > {ABANDON_AI_ACTIONS})"
COUNTED_SCORE_SQL = (
    f"(CASE WHEN {ABANDONED_SQL} AND gp.score > 0 THEN 0 ELSE gp.score END)"
)

LEADERBOARD_MIN_HANDS = 20
LEADERBOARD_SIZE = 20
_CACHE_TTL = 60.0

_HAND_STATS = f"""
    SELECT gp.player_id,
           count(*)::int                                    AS hands,
           sum({COUNTED_SCORE_SQL})::int                    AS total,
           avg({COUNTED_SCORE_SQL})::float8                 AS sph,
           avg(({COUNTED_SCORE_SQL} > 0)::int)::float8      AS win_pct,
           avg((gp.is_picker IS TRUE)::int)::float8         AS pick_pct,
           (count(*) FILTER (WHERE g.is_leaster))::int      AS leaster_hands,
           (count(*) FILTER (WHERE {ABANDONED_SQL}))::int   AS abandoned_hands,
           (count(*) FILTER (WHERE {AI_ASSISTED_SQL}))::int AS ai_assisted_hands,
           sum(gp.score - {COUNTED_SCORE_SQL})::int         AS forfeited_score
    FROM game_player gp
    JOIN game g ON g.game_id = gp.game_id
    WHERE g.time_closed IS NOT NULL
      AND gp.score IS NOT NULL
      AND gp.player_id IS NOT NULL
"""

# Below this many hands a split's score/hand carries no margin: the
# interval would be too wide to mean anything.
SPLIT_MARGIN_MIN_HANDS = 20

# A hand gp played against the AI alone: every other seat was dealt to
# the AI and the AI made every move in it. Seat ownership is fixed at the
# deal (a human who leaves keeps the row), so a walkout doesn't turn a
# hand into an AI hand; and a person who takes over an AI-dealt seat
# mid-hand leaves substitution flags on its row, which disqualify it.
VS_AI_SQL = """
    NOT EXISTS (
        SELECT 1 FROM game_player o
        WHERE o.game_id = gp.game_id
          AND o.game_player_id <> gp.game_player_id
          AND (o.player_id IS NOT NULL
               OR o.is_substituted_pick
               OR EXISTS (
                   SELECT 1 FROM trick_card tc
                   WHERE tc.game_player_id = o.game_player_id
                     AND tc.is_substituted
               ))
    )
"""

# One row per company for the caller: vs_ai or with other people.
_SPLITS = f"""
    WITH mine AS (
        SELECT {COUNTED_SCORE_SQL} AS counted,
               {VS_AI_SQL} AS vs_ai
        FROM game_player gp
        JOIN game g ON g.game_id = gp.game_id
        WHERE g.time_closed IS NOT NULL
          AND gp.score IS NOT NULL
          AND gp.player_id = $1
    )
    SELECT vs_ai,
           count(*)::int                        AS hands,
           avg(counted)::float8                 AS sph,
           stddev_samp(counted)::float8         AS sd,
           avg((counted > 0)::int)::float8      AS win_pct
    FROM mine
    GROUP BY vs_ai
"""

# Trump and queens in the six cards dealt to the caller on hands they
# chose to pick: how strong a hand they pick on, so the blind is left
# out, as are picks the AI made for them.
_PICKING = """
    SELECT count(*)::int                AS picks,
           avg(t.trump)::float8         AS trump_per_pick,
           avg(t.queens)::float8        AS queens_per_pick
    FROM game_player gp
    JOIN game g ON g.game_id = gp.game_id,
    LATERAL (
        SELECT count(*) FILTER (
                   WHERE left(c.code, -1) IN ('Q', 'J') OR s.code = 'D'
               ) AS trump,
               count(*) FILTER (WHERE left(c.code, -1) = 'Q') AS queens
        FROM cardset_card cc
        JOIN card c ON c.card_id = cc.card_id
        JOIN suit s ON s.suit_id = c.suit_id
        WHERE cc.cardset_id = gp.starting_hand_id
    ) t
    WHERE g.time_closed IS NOT NULL
      AND gp.score IS NOT NULL
      AND gp.player_id = $1
      AND gp.is_picker
      AND NOT gp.is_substituted_pick
"""

# sort key -> (monotonic time cached, full ranking). The ranking holds every
# eligible player; only the top rows and the caller's own row are served.
_cache: dict[SortKey, tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    _cache.clear()


async def player_stats(pool: asyncpg.Pool, player_id: UUID) -> dict:
    row = await pool.fetchrow(
        _HAND_STATS + " AND gp.player_id = $1 GROUP BY gp.player_id", player_id
    )
    if row is None:
        return {
            "hands": 0,
            "total": 0,
            "sph": None,
            "win_pct": None,
            "pick_pct": None,
            "leaster_hands": 0,
            "abandoned_hands": 0,
            "ai_assisted_hands": 0,
            "forfeited_score": 0,
        }
    return {k: row[k] for k in row.keys() if k != "player_id"}


def _split(row: Optional[asyncpg.Record]) -> dict:
    if row is None:
        return {"hands": 0, "sph": None, "sph_margin": None, "win_pct": None}
    hands = row["hands"]
    margin = None
    if hands >= SPLIT_MARGIN_MIN_HANDS and row["sd"] is not None:
        margin = 1.96 * row["sd"] / math.sqrt(hands)
    return {
        "hands": hands,
        "sph": row["sph"],
        "sph_margin": margin,
        "win_pct": row["win_pct"],
    }


async def player_splits(pool: asyncpg.Pool, player_id: UUID) -> dict:
    """Score/hand and win rate against the AI alone and with other
    people at the table, with a 95% margin on score/hand."""
    rows = {r["vs_ai"]: r for r in await pool.fetch(_SPLITS, player_id)}
    return {"vs_ai": _split(rows.get(True)), "with_people": _split(rows.get(False))}


async def player_picking(pool: asyncpg.Pool, player_id: UUID) -> dict:
    row = await pool.fetchrow(_PICKING, player_id)
    assert row is not None  # an aggregate always returns a row
    return dict(row)


async def ranking(pool: asyncpg.Pool, sort: SortKey) -> list[dict]:
    """Every verified account with enough hands, best first by ``sort``
    (ties: more hands, then username). Cached for a minute per sort."""
    hit = _cache.get(sort)
    now = time.monotonic()
    if hit is not None and now - hit[0] < _CACHE_TTL:
        return hit[1]
    assert sort in SORT_KEYS  # interpolated below; never caller-controlled
    rows = await pool.fetch(
        f"""
        WITH stats AS ({_HAND_STATS} GROUP BY gp.player_id HAVING count(*) >= $1)
        SELECT row_number() OVER (
                   ORDER BY s.{sort} DESC, s.hands DESC, lower(a.username)
               )::int AS rank,
               a.username, s.*
        FROM stats s
        JOIN account a
          ON a.player_id = s.player_id AND a.email_verified_at IS NOT NULL
        ORDER BY rank
        """,
        LEADERBOARD_MIN_HANDS,
    )
    result = [dict(r) for r in rows]
    _cache[sort] = (now, result)
    return result


def row_of(rows: list[dict], player_id: Optional[UUID]) -> Optional[dict]:
    if player_id is None:
        return None
    return next((r for r in rows if r["player_id"] == player_id), None)
