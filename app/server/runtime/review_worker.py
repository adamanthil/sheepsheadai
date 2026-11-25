"""The AI review worker: one background job that reviews queued hands.

It takes the newest queued hand, replays it against the model
(services.hand_review) and stores what it finds. Live play comes first, in
two ways: the worker does not start a hand while more than one table is in
play, and each forward pass it makes takes a single turn at the inference
gate, so a live move never waits behind more than one of them.

Nothing a player does triggers a review. Hands are queued when they end and
when a player registers (persistence.reviews), and this is the only place
they are worked through.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import anyio.to_thread
import asyncpg

from server.runtime.manager import tables
from server.services.ai_loader import load_agent
from server.services.hand_review import HandReview, ReplayError, review_hand
from server.services.persistence import reviews as reviews_db
from server.services.persistence.hand_records import load_hands, skipped_positions
from server.services.persistence.pool import DB_UNAVAILABLE
from server.services.persistence.reviews import QueuedSeat
from sheepshead.agent.ppo import PPOAgent

logger = logging.getLogger(__name__)

# The worker stands aside while more tables than this are in play.
MAX_TABLES_IN_PLAY = 1
# Seconds between looks at the queue while it is empty or the worker is
# standing aside.
IDLE_SECONDS = 5.0


def tables_in_play() -> int:
    """Tables with a hand in progress and at least one human connected."""
    count = 0
    for table in tables.tables.values():
        game = table.game
        if game is None or game.is_done():
            continue
        if any(c.seat is not None and c.sockets for c in table.clients.values()):
            count += 1
    return count


def is_paused() -> bool:
    return tables_in_play() > MAX_TABLES_IN_PLAY


async def review_queued_seat(
    pool: asyncpg.Pool, agent: PPOAgent, ai_model_id: int, job: QueuedSeat
) -> None:
    """Review one queued seat and record the outcome, whatever it is."""
    hands = await load_hands(pool, [job.game_id])
    if not hands:
        await reviews_db.mark_failed(
            pool, job.game_player_id, "hand not found", final=True
        )
        return
    hand = hands[0]
    if job.seat in hand.unflagged_ai:
        await reviews_db.mark_skipped(pool, job.game_player_id)
        return

    started = time.perf_counter()
    try:
        # In a worker thread: the engine is synchronous, and that is where
        # its forward passes can take their turns at the inference gate.
        result: HandReview = await anyio.to_thread.run_sync(
            lambda: review_hand(
                agent, hand.record, [job.seat], skip=skipped_positions(hand)
            )
        )
    except ReplayError as err:
        logger.warning(
            "hand %s seat %s does not replay: %s", job.game_id, job.seat, err
        )
        await reviews_db.mark_failed(
            pool, job.game_player_id, f"does not replay: {err}", final=True
        )
        return
    except DB_UNAVAILABLE:
        # Not this hand's fault, and there is nowhere to record it: leave
        # the hand queued for the worker's loop to come back to.
        raise
    except Exception as err:
        # Deliberately broad: whatever is wrong with this one hand, it must
        # not stop every hand behind it. The failure is logged in full and
        # counted, and the hand is given up on after MAX_ATTEMPTS.
        logger.exception("review of hand %s seat %s failed", job.game_id, job.seat)
        await reviews_db.mark_failed(
            pool, job.game_player_id, f"{type(err).__name__}: {err}", final=False
        )
        return

    await reviews_db.store_review(
        pool,
        job.game_player_id,
        ai_model_id,
        result.decisions,
        duration_ms=round((time.perf_counter() - started) * 1000),
        evaluations=result.evaluations,
    )


async def run_review_worker(
    pool: asyncpg.Pool, model_path: str, ai_model_id: int
) -> None:
    """Work through the queue for as long as the server runs."""
    # The reviewing agent is the deployed model taking its first choice: no
    # deployment sampler, and its own recurrent memory apart from any table.
    agent: Optional[PPOAgent] = None
    while True:
        try:
            job = None if is_paused() else await reviews_db.next_queued(pool)
            if job is not None:
                if agent is None:
                    agent = load_agent(model_path)
                await review_queued_seat(pool, agent, ai_model_id, job)
                continue
        except DB_UNAVAILABLE:
            # Postgres restarting or briefly out of reach: the hand stays
            # queued, and the worker tries again rather than stopping for
            # good.
            logger.warning("AI review paused: database unavailable")
        await asyncio.sleep(IDLE_SECONDS)
