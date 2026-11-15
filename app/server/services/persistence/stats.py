"""Per-player hand statistics and the public leaderboard.

Everything derives from game_player rows of finished hands. A hand counts
once it is closed and scored, which leaves out hands still in play and
passed-out doublers deals (closed, but never scored). The one exception is
pick rate: a pass-out was dealt and everyone declined to pick, so it counts
there as a hand not picked. Scores already include the doublers stake.
Hands the AI finished for a player (timeouts, disconnects) count as
theirs, so walking away can't hide a loss.

Nor can it bank a win: a hand is abandoned when the AI made more than
ABANDON_AI_ACTIONS of the player's decisions in it (excused moves aside --
a kick, a host close, a restart). An abandoned hand's loss counts in full,
but a positive score counts as 0. The rule is applied here, at query time,
so changing the threshold re-grades all history consistently.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Literal, NamedTuple, Optional
from uuid import UUID

import asyncpg

# Hands played and pick rate are shown on the board but deliberately not
# rankable: picking more often is a style, not a measure of skill.
SortKey = Literal["total", "sph", "win_pct"]
SORT_KEYS: tuple[SortKey, ...] = ("total", "sph", "win_pct")

# Which hands a board counts: all of them, one partner mode, or only
# hands played against the AI alone.
BoardKey = Literal["overall", "called_ace", "jd", "vs_ai"]

# Which closing times a board counts: every hand, or only hands closed in
# the current or the previous calendar month. Months run midnight to
# midnight on the 1st in LEADERBOARD_TZ, whatever the server's own zone.
PeriodKey = Literal["all_time", "month", "last_month"]
MONTHS_BACK: dict[PeriodKey, int] = {"month": 0, "last_month": 1}
LEADERBOARD_TZ = "America/Chicago"

# More AI decisions than this in a hand, not excused, abandon it.
ABANDON_AI_ACTIONS = 2

# SQL over a game_player row aliased gp.
AI_ASSISTED_SQL = "(gp.ai_actions > 0)"
ABANDONED_SQL = f"(gp.ai_actions - gp.ai_actions_excused > {ABANDON_AI_ACTIONS})"
COUNTED_SCORE_SQL = (
    f"(CASE WHEN {ABANDONED_SQL} AND gp.score > 0 THEN 0 ELSE gp.score END)"
)

LEADERBOARD_MIN_HANDS = 20
VS_AI_LEADERBOARD_MIN_HANDS = 40
LEADERBOARD_SIZE = 20
_CACHE_TTL = 60.0

# A closed hand that was played out and scored, as opposed to a passed-out
# doublers deal (closed, score NULL).
SCORED_SQL = "(gp.score IS NOT NULL)"
HANDS_SQL = f"count(*) FILTER (WHERE {SCORED_SQL})"

# Rows are every closed deal, pass-outs included, but only pick_pct reads
# the pass-outs: the counts filter on SCORED_SQL, and the score aggregates
# skip them because their counted score is NULL.
_HAND_STATS = f"""
    SELECT gp.player_id,
           ({HANDS_SQL})::int                               AS hands,
           sum({COUNTED_SCORE_SQL})::int                    AS total,
           avg({COUNTED_SCORE_SQL})::float8                 AS sph,
           avg(({COUNTED_SCORE_SQL} > 0)::int)::float8      AS win_pct,
           avg((gp.is_picker IS TRUE)::int)::float8         AS pick_pct,
           (count(*) FILTER (WHERE {SCORED_SQL} AND g.is_leaster))::int
                                                            AS leaster_hands,
           (count(*) FILTER (WHERE {SCORED_SQL} AND {ABANDONED_SQL}))::int
                                                            AS abandoned_hands,
           (count(*) FILTER (WHERE {SCORED_SQL} AND {AI_ASSISTED_SQL}))::int
                                                            AS ai_assisted_hands,
           sum(gp.score - {COUNTED_SCORE_SQL})::int         AS forfeited_score
    FROM game_player gp
    JOIN game g ON g.game_id = gp.game_id
    WHERE g.time_closed IS NOT NULL
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
#
# It comes in two equivalent forms, as gp's own seat is always a person's:
# "no other seat is a person's" (VS_AI_SQL) is "exactly one seat is"
# (VS_AI_GAMES_SQL). The per-row form suits one player's few thousand
# rows; over every player the planner misjudges its anti-join and
# rescans game per row (minutes at 100k hands), so the leaderboard
# finds the games once instead.
_PERSON_SEAT_SQL = """
    (o.player_id IS NOT NULL
     OR o.is_substituted_pick
     OR EXISTS (
         SELECT 1 FROM trick_card tc
         WHERE tc.game_player_id = o.game_player_id
           AND tc.is_substituted
     ))
"""
VS_AI_SQL = f"""
    NOT EXISTS (
        SELECT 1 FROM game_player o
        WHERE o.game_id = gp.game_id
          AND o.game_player_id <> gp.game_player_id
          AND {_PERSON_SEAT_SQL}
    )
"""
VS_AI_GAMES_SQL = f"""
    SELECT o.game_id FROM game_player o
    WHERE {_PERSON_SEAT_SQL}
    GROUP BY o.game_id
    HAVING count(*) = 1
"""

# Extra conditions on the hands each board counts, over gp and g: the
# per-row forms for one player's hands, and for ranking every player.
_BOARD_SQL: dict[BoardKey, str] = {
    "overall": "",
    "called_ace": " AND g.is_called_partner",
    "jd": " AND NOT g.is_called_partner",
    "vs_ai": f" AND {VS_AI_SQL}",
}
_RANKING_BOARD_SQL: dict[BoardKey, str] = {
    **_BOARD_SQL,
    "vs_ai": f" AND gp.game_id IN ({VS_AI_GAMES_SQL})",
}


def board_min_hands(board: BoardKey, monthly: bool = False) -> int:
    """Hands of ``board``'s kind to qualify for it: a monthly board asks
    half its lifetime count."""
    base = VS_AI_LEADERBOARD_MIN_HANDS if board == "vs_ai" else LEADERBOARD_MIN_HANDS
    return base // 2 if monthly else base


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


class Month(NamedTuple):
    """A leaderboard month: [start, end) as aware instants."""

    start: datetime
    end: datetime


# Postgres's own zone database, so the bounds don't depend on tzdata in
# the server's Python environment.
_MONTH_SQL = """
    WITH m AS (
        SELECT date_trunc('month', now() AT TIME ZONE $1)
                   - make_interval(months => $2) AS local_start
    )
    SELECT local_start AT TIME ZONE $1 AS start,
           (local_start + interval '1 month') AT TIME ZONE $1 AS end
    FROM m
"""


def _in_month_sql(start: str, end: str) -> str:
    """Hands closed in [start, end), two timestamptz parameters.

    time_closed is a naive timestamp stamped by now() in the session zone,
    so the bounds are compared in that zone too; read the other way round,
    the comparison would shift with the server's UTC offset.
    """
    local = "::timestamptz AT TIME ZONE current_setting('TimeZone')"
    return f" AND g.time_closed >= ({start}{local}) AND g.time_closed < ({end}{local})"


async def month_bounds(pool: asyncpg.Pool, period: PeriodKey) -> Optional[Month]:
    """The month a ``period`` covers; None for all time."""
    if period not in MONTHS_BACK:
        return None
    row = await pool.fetchrow(_MONTH_SQL, LEADERBOARD_TZ, MONTHS_BACK[period])
    assert row is not None
    return Month(row["start"], row["end"])


# (board, sort key, period start) -> (monotonic time cached, full ranking).
# The ranking holds every eligible player; only the top rows and the
# caller's own row are served. Keying on the start resets a month's board
# the moment it rolls over, not up to a TTL later.
_CacheKey = tuple[BoardKey, SortKey, Optional[datetime]]
_cache: dict[_CacheKey, tuple[float, list[dict]]] = {}


def clear_cache() -> None:
    _cache.clear()


async def player_stats(pool: asyncpg.Pool, player_id: UUID) -> dict:
    row = await pool.fetchrow(
        _HAND_STATS + " AND gp.player_id = $1 GROUP BY gp.player_id", player_id
    )
    # Only pass-outs so far is no hands yet, not a 0% pick rate.
    if row is None or row["hands"] == 0:
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


async def ranking(
    pool: asyncpg.Pool,
    sort: SortKey,
    board: BoardKey = "overall",
    month: Optional[Month] = None,
) -> list[dict]:
    """Every verified account with enough of ``board``'s hands, best first
    by ``sort`` (ties: more hands, then username); with ``month``, only
    hands closed in it count. Cached for a minute per board, sort and
    month."""
    key: _CacheKey = (board, sort, month.start if month else None)
    hit = _cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < _CACHE_TTL:
        return hit[1]
    # Both interpolated below; never caller-controlled.
    assert sort in SORT_KEYS and board in _RANKING_BOARD_SQL
    args: list = [board_min_hands(board, monthly=month is not None)]
    if month:
        args += [month.start, month.end]
    rows = await pool.fetch(
        f"""
        WITH stats AS (
            {_HAND_STATS}{_RANKING_BOARD_SQL[board]}{_in_month_sql("$2", "$3") if month else ""}
            GROUP BY gp.player_id HAVING {HANDS_SQL} >= $1
        )
        SELECT row_number() OVER (
                   ORDER BY s.{sort} DESC, s.hands DESC, lower(a.username)
               )::int AS rank,
               a.username, s.*
        FROM stats s
        JOIN account a
          ON a.player_id = s.player_id AND a.email_verified_at IS NOT NULL
        ORDER BY rank
        """,
        *args,
    )
    result = [dict(r) for r in rows]
    _cache[key] = (now, result)
    return result


async def board_hands(
    pool: asyncpg.Pool,
    player_id: UUID,
    board: BoardKey,
    month: Optional[Month] = None,
) -> int:
    """How many of ``board``'s hands the player has finished (in ``month``,
    if given): their count toward qualifying for it."""
    args: list = [player_id]
    if month:
        args += [month.start, month.end]
    row = await pool.fetchrow(
        f"""{_HAND_STATS}{_BOARD_SQL[board]}{_in_month_sql("$2", "$3") if month else ""}
            AND gp.player_id = $1 GROUP BY gp.player_id""",
        *args,
    )
    return row["hands"] if row is not None else 0


def row_of(rows: list[dict], player_id: Optional[UUID]) -> Optional[dict]:
    if player_id is None:
        return None
    return next((r for r in rows if r["player_id"] == player_id), None)
