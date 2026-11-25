"""AI review storage: the queue of hands to review and what each review found.

A ``hand_review`` row is one registered player's seat in one finished hand.
It starts out queued, and the worker (runtime.review_worker) fills it in
and writes the seat's ``decision_review`` rows beside it. Every stored
figure names the ai_model that produced it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence
from uuid import UUID

import asyncpg

from server.services.hand_review import DecisionReview
from server.services.persistence.pool import Querier
from server.services.review_bands import summarize

# A hand is given up on after failing this many times for an unexpected
# reason. A record the engine cannot follow fails at once: it never will.
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class QueuedSeat:
    game_player_id: int
    game_id: UUID
    seat: int


async def enqueue_game(conn: Querier, game_id: UUID) -> None:
    """Queue the seats of a just-finished hand whose players have accounts."""
    await conn.execute(
        """
        INSERT INTO hand_review (game_player_id)
        SELECT gp.game_player_id
        FROM game_player gp
        JOIN account a ON a.player_id = gp.player_id
        WHERE gp.game_id = $1 AND gp.score IS NOT NULL
        ON CONFLICT DO NOTHING
        """,
        game_id,
    )


async def enqueue_player(conn: Querier, player_id: UUID) -> None:
    """Queue every finished hand ``player_id`` has played (on registering:
    the account keeps the hands its player row played as a guest)."""
    await conn.execute(
        """
        INSERT INTO hand_review (game_player_id)
        SELECT gp.game_player_id
        FROM game_player gp
        JOIN game g ON g.game_id = gp.game_id
        WHERE gp.player_id = $1
          AND g.time_closed IS NOT NULL AND gp.score IS NOT NULL
        ON CONFLICT DO NOTHING
        """,
        player_id,
    )


async def next_queued(conn: Querier) -> Optional[QueuedSeat]:
    """The newest hand waiting for review, if any."""
    row = await conn.fetchrow(
        """
        SELECT hr.game_player_id, gp.game_id, gp.position
        FROM hand_review hr
        JOIN game_player gp ON gp.game_player_id = hr.game_player_id
        WHERE hr.status = 'queued'
        ORDER BY hr.game_player_id DESC
        LIMIT 1
        """
    )
    if row is None:
        return None
    return QueuedSeat(row["game_player_id"], row["game_id"], row["position"])


async def store_review(
    pool: asyncpg.Pool,
    game_player_id: int,
    ai_model_id: int,
    decisions: Sequence[DecisionReview],
    duration_ms: int,
    evaluations: int,
) -> None:
    """Record a finished review, replacing any earlier one of the seat."""
    # Only the two sums are kept beside the decisions: they do not depend
    # on the bands. Everything banded is counted when the review is read.
    summary = summarize((d.index, d.forced, d.difference) for d in decisions)
    async with pool.acquire() as conn, conn.transaction():
        # Decisions first: they are tied to the model the old review named.
        await conn.execute(
            "DELETE FROM decision_review WHERE game_player_id = $1", game_player_id
        )
        await conn.execute(
            """
            UPDATE hand_review
            SET status = 'reviewed', failure = NULL, ai_model_id = $2,
                time_reviewed = now(), duration_ms = $3, evaluations = $4,
                cost = $5, gained = $6
            WHERE game_player_id = $1
            """,
            game_player_id,
            ai_model_id,
            duration_ms,
            evaluations,
            summary.cost,
            summary.gained,
        )
        await conn.executemany(
            """
            INSERT INTO decision_review (
                game_player_id, index, ai_model_id, kind, trick_index, hand,
                table_cards, action, model_action, policy, is_forced,
                expected_score, win_prob, oracle_score, chosen_score,
                model_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10::jsonb,
                    $11, $12, $13, $14, $15, $16)
            """,
            [
                (
                    game_player_id,
                    d.index,
                    ai_model_id,
                    d.kind,
                    d.trick,
                    d.hand,
                    json.dumps(d.table),
                    d.action,
                    d.model_action,
                    json.dumps([[a, round(p, 4)] for a, p in d.policy]),
                    d.forced,
                    d.state.score,
                    d.state.win_prob,
                    d.oracle_score,
                    d.chosen.score if d.difference is not None and d.chosen else None,
                    d.model.score if d.difference is not None and d.model else None,
                )
                for d in decisions
            ],
        )


async def mark_skipped(conn: Querier, game_player_id: int) -> None:
    """A hand that cannot be reviewed: the AI made the player's call or bury."""
    await conn.execute(
        """
        UPDATE hand_review
        SET status = 'skipped', time_reviewed = now()
        WHERE game_player_id = $1
        """,
        game_player_id,
    )


async def mark_failed(
    conn: Querier, game_player_id: int, reason: str, final: bool
) -> None:
    """Count a failed attempt. The hand goes back in the queue unless
    ``final`` or it has now failed MAX_ATTEMPTS times."""
    await conn.execute(
        """
        UPDATE hand_review
        SET attempts = attempts + 1, failure = $2, time_reviewed = now(),
            status = CASE WHEN $3 OR attempts + 1 >= $4
                          THEN 'failed' ELSE 'queued' END
        WHERE game_player_id = $1
        """,
        game_player_id,
        reason[:500],
        final,
        MAX_ATTEMPTS,
    )
