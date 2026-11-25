"""Reading stored AI reviews back: one hand, a player's totals, the queue."""

from __future__ import annotations

import json
from datetime import UTC
from typing import Any, Dict, List, Optional
from uuid import UUID

import asyncpg

from server.services.persistence.stats import ROLE_SQL
from server.services.review_bands import EVEN_BELOW, FLAG_FROM

# Rows in each list of the review totals.
REVIEW_LIST_SIZE = 30

# A reviewed decision's standing against the AI's choice, as
# review_bands.standing gives it (test_review_api holds the two together).
# Every query that bands a decision is built from these.
_COSTLY_SQL = f"d.difference <= -{FLAG_FROM}"
_BETTER_SQL = f"d.difference >= {FLAG_FROM}"
STANDING_SQL = f"""
CASE WHEN d.is_forced THEN 'forced'
     WHEN d.difference IS NULL THEN 'agreed'
     WHEN {_COSTLY_SQL} THEN 'costly'
     WHEN {_BETTER_SQL} THEN 'better'
     WHEN abs(d.difference) < {EVEN_BELOW} THEN 'even'
     ELSE 'slight' END
"""


def decode(value: Any) -> Any:
    """A JSONB column as asyncpg hands it back (text unless a codec is set)."""
    return json.loads(value) if isinstance(value, str) else value


async def hand_review(
    pool: asyncpg.Pool, player_id: UUID, game_id: UUID
) -> Optional[Dict[str, Any]]:
    """The stored review of ``player_id``'s seat in ``game_id``: its queue
    row (with the model's label) and its decisions. None if the player was
    not in the hand or it was never queued."""
    row = await pool.fetchrow(
        """
        SELECT hr.*, gp.position, m.label AS ai_model
        FROM hand_review hr
        JOIN game_player gp ON gp.game_player_id = hr.game_player_id
        LEFT JOIN ai_model m ON m.ai_model_id = hr.ai_model_id
        WHERE gp.player_id = $1 AND gp.game_id = $2
        """,
        player_id,
        game_id,
    )
    if row is None:
        return None
    decisions = await pool.fetch(
        f"""
        SELECT d.*, {STANDING_SQL} AS standing
        FROM decision_review d
        WHERE d.game_player_id = $1
        ORDER BY d.index
        """,
        row["game_player_id"],
    )
    return {"row": row, "decisions": decisions}


async def hand_seats(pool: asyncpg.Pool, game_id: UUID) -> List[asyncpg.Record]:
    """Each seat of the hand: who sat there and in what role."""
    return await pool.fetch(
        f"""
        SELECT gp.position, gp.player_id, gp.name, {ROLE_SQL} AS role
        FROM game_player gp
        JOIN game g ON g.game_id = gp.game_id
        WHERE gp.game_id = $1
        ORDER BY gp.position
        """,
        game_id,
    )


# The kinds the review totals are broken down by: an under card is part of
# the partner call.
REVIEW_KINDS = ("pick", "call", "bury", "play")
_KIND_SQL = "CASE d.kind WHEN 'under' THEN 'call' ELSE d.kind END"

# The largest differences one way ({where}, largest first by {order}): the
# first $2 overall and the first $2 of each kind, so that a list narrowed
# to a kind is as long as a list of everything.
_DIFFERENCE_SQL = f"""
SELECT * FROM (
    SELECT gp.game_id, d.index, d.kind, d.trick_index, d.table_cards,
           d.action, d.model_action, d.policy, d.difference,
           {STANDING_SQL} AS standing, g.time_closed, t.name AS table_name,
           {_KIND_SQL} AS review_kind,
           row_number() OVER (ORDER BY {{order}}, g.time_closed DESC)
               AS place,
           row_number() OVER (PARTITION BY {_KIND_SQL}
                              ORDER BY {{order}}, g.time_closed DESC)
               AS place_in_kind
    FROM decision_review d
    JOIN game_player gp ON gp.game_player_id = d.game_player_id
    JOIN game g ON g.game_id = gp.game_id
    JOIN game_table t ON t.game_table_id = g.game_table_id
    WHERE gp.player_id = $1 AND {{where}}
) ranked
WHERE place <= $2 OR place_in_kind <= $2
ORDER BY place
"""


def _difference(row: asyncpg.Record) -> dict:
    return {
        "game_id": row["game_id"],
        "index": row["index"],
        "kind": row["kind"],
        "trick": row["trick_index"],
        "led": row["kind"] == "play" and not decode(row["table_cards"]),
        "action": row["action"],
        "ai_action": row["model_action"],
        "ai_probability": decode(row["policy"])[0][1],
        "difference": row["difference"],
        "standing": row["standing"],
        "time_closed": row["time_closed"].replace(tzinfo=UTC),
        "table_name": row["table_name"],
    }


async def _differences(
    pool: asyncpg.Pool, player_id: UUID, where: str, order: str
) -> dict:
    """The largest differences one way, overall and by kind of decision."""
    rows = await pool.fetch(
        _DIFFERENCE_SQL.format(where=where, order=order),
        player_id,
        REVIEW_LIST_SIZE,
    )
    lists: Dict[str, List[dict]] = {
        "all": [_difference(r) for r in rows if r["place"] <= REVIEW_LIST_SIZE]
    }
    for kind in REVIEW_KINDS:
        lists[kind] = [
            _difference(r)
            for r in rows
            if r["review_kind"] == kind and r["place_in_kind"] <= REVIEW_LIST_SIZE
        ]
    return lists


async def _picks(pool: asyncpg.Pool, player_id: UUID, picked: bool) -> dict:
    """Pick decisions where the player and the AI went different ways: how
    many there were, what those hands scored, and the most recent."""
    action = "PICK" if picked else "PASS"
    totals = await pool.fetchrow(
        """
        SELECT count(*)::int AS count, coalesce(sum(gp.score), 0)::int AS score
        FROM decision_review d
        JOIN game_player gp ON gp.game_player_id = d.game_player_id
        WHERE gp.player_id = $1 AND d.kind = 'pick'
          AND d.action <> d.model_action
          AND d.action = $2
        """,
        player_id,
        action,
    )
    rows = await pool.fetch(
        """
        SELECT gp.game_id, gp.position, gp.score, d.hand, d.policy,
               d.chosen_score, d.model_score, g.time_closed,
               t.name AS table_name
        FROM decision_review d
        JOIN game_player gp ON gp.game_player_id = d.game_player_id
        JOIN game g ON g.game_id = gp.game_id
        JOIN game_table t ON t.game_table_id = g.game_table_id
        WHERE gp.player_id = $1 AND d.kind = 'pick'
          AND d.action <> d.model_action
          AND d.action = $2
        ORDER BY g.time_closed DESC
        LIMIT $3
        """,
        player_id,
        action,
        REVIEW_LIST_SIZE,
    )
    out = []
    for row in rows:
        chances = {action: p for action, p in decode(row["policy"])}
        chosen, model = row["chosen_score"], row["model_score"]
        out.append(
            {
                "game_id": row["game_id"],
                "time_closed": row["time_closed"].replace(tzinfo=UTC),
                "table_name": row["table_name"],
                "seat": row["position"],
                "hand": row["hand"],
                "picked": picked,
                "ai_pick_probability": chances.get("PICK", 0.0),
                "pick_score": chosen if picked else model,
                "pass_score": model if picked else chosen,
                "score": row["score"],
            }
        )
    return {"count": totals["count"], "score": totals["score"], "picks": out}


async def account_review(pool: asyncpg.Pool, player_id: UUID) -> dict:
    """``player_id``'s reviewed hands, totalled."""
    totals = await pool.fetchrow(
        """
        SELECT count(*) FILTER (WHERE hr.status = 'reviewed')::int AS reviewed,
               count(*) FILTER (WHERE hr.status = 'queued')::int AS waiting,
               coalesce(sum(hr.decisions), 0)::int AS decisions,
               coalesce(sum(hr.agreed), 0)::int AS agreed,
               coalesce(sum(hr.even), 0)::int AS even,
               coalesce(sum(hr.slight), 0)::int AS slight,
               coalesce(sum(hr.costly), 0)::int AS costly,
               coalesce(sum(hr.better), 0)::int AS better,
               coalesce(sum(hr.gained - hr.cost), 0)::float AS net
        FROM hand_review hr
        JOIN game_player gp ON gp.game_player_id = hr.game_player_id
        WHERE gp.player_id = $1
        """,
        player_id,
    )
    kinds = await pool.fetch(
        f"""
        SELECT {_KIND_SQL} AS kind,
               count(*)::int AS decisions,
               count(*) FILTER (WHERE d.difference IS NULL)::int AS agreed,
               count(*) FILTER (WHERE {_COSTLY_SQL})::int AS costly,
               count(*) FILTER (WHERE {_BETTER_SQL})::int AS better,
               coalesce(sum(d.difference), 0)::float AS net
        FROM decision_review d
        JOIN game_player gp ON gp.game_player_id = d.game_player_id
        WHERE gp.player_id = $1 AND NOT d.is_forced
        GROUP BY 1
        """,
        player_id,
    )
    by_kind = {row["kind"]: row for row in kinds}
    pick = by_kind.get("pick")
    return {
        "hands_reviewed": totals["reviewed"],
        "hands_waiting": totals["waiting"],
        "decisions": totals["decisions"],
        "agreed": totals["agreed"],
        "even": totals["even"],
        "slight": totals["slight"],
        "costly": totals["costly"],
        "better": totals["better"],
        "net": round(totals["net"], 1),
        "by_kind": [
            {
                "kind": kind,
                "decisions": by_kind[kind]["decisions"],
                "agreed": by_kind[kind]["agreed"],
                "costly": by_kind[kind]["costly"],
                "better": by_kind[kind]["better"],
                "net": round(by_kind[kind]["net"], 1),
            }
            for kind in REVIEW_KINDS
            if kind in by_kind
        ],
        "cost_most": await _differences(pool, player_id, _COSTLY_SQL, "d.difference"),
        "gained_most": await _differences(
            pool, player_id, _BETTER_SQL, "d.difference DESC"
        ),
        "pick_decisions": pick["decisions"] if pick else 0,
        "pick_agreed": pick["agreed"] if pick else 0,
        "picked_ai_passes": await _picks(pool, player_id, picked=True),
        "passed_ai_picks": await _picks(pool, player_id, picked=False),
    }


async def queue_stats(pool: asyncpg.Pool, ai_model_label: str) -> dict:
    """The review queue as the admin page shows it."""
    row = await pool.fetchrow(
        """
        SELECT count(*)::int AS total,
               count(*) FILTER (WHERE status = 'queued')::int AS waiting,
               count(*) FILTER (WHERE status = 'reviewed')::int AS reviewed,
               count(*) FILTER (WHERE status = 'skipped')::int AS skipped,
               count(*) FILTER (WHERE status = 'failed')::int AS failed,
               min(time_queued) FILTER (WHERE status = 'queued') AS oldest_queued,
               max(time_reviewed) FILTER (WHERE status = 'reviewed')
                   AS last_reviewed,
               count(*) FILTER (WHERE status = 'reviewed'
                   AND time_reviewed > now() - interval '24 hours')::int
                   AS reviewed_24h,
               (avg(duration_ms) FILTER (WHERE status = 'reviewed'
                   AND time_reviewed > now() - interval '24 hours'))::int
                   AS average_ms_24h,
               max(duration_ms) FILTER (WHERE status = 'reviewed'
                   AND time_reviewed > now() - interval '24 hours')
                   AS slowest_ms_24h,
               count(*) FILTER (WHERE status = 'reviewed' AND ai_model_id <>
                   (SELECT ai_model_id FROM ai_model WHERE label = $1))::int
                   AS other_model_reviews
        FROM hand_review
        """,
        ai_model_label,
    )
    failures = await pool.fetch(
        """
        SELECT gp.game_id, gp.position AS seat, hr.failure AS reason
        FROM hand_review hr
        JOIN game_player gp ON gp.game_player_id = hr.game_player_id
        WHERE hr.status = 'failed'
        ORDER BY hr.time_reviewed DESC NULLS LAST
        LIMIT 20
        """
    )
    stats = dict(row)
    for key in ("oldest_queued", "last_reviewed"):
        if stats[key] is not None:
            stats[key] = stats[key].replace(tzinfo=UTC)
    stats["failures"] = [dict(f) for f in failures]
    return stats
