"""The AI review queue and its worker, over hands played through the real
persistence hooks.

Opt-in like test_hand_history: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from server.runtime import review_worker
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Table
from server.services.hand_review import ReplayError
from server.services.persistence import accounts as accounts_db
from server.services.persistence import reviews as reviews_db
from server.services.persistence.reviews import MAX_ATTEMPTS
from server.tests.test_hand_history import _play_hand
from sheepshead.agent.ppo import PPOAgent
from sheepshead.game import ACTIONS, PARTNER_BY_CALLED_ACE, Game

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

needs_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


@pytest.fixture(scope="module")
def agent() -> PPOAgent:
    return PPOAgent(len(ACTIONS), critic_mode="oracle")


async def _register(pool, player_id: uuid.UUID) -> None:
    tag = uuid.uuid4().hex[:10]
    await accounts_db.create_account(
        pool, player_id, f"r_{tag}", f"{tag}@example.com", "not-a-real-hash"
    )


async def _review_rows(pool, player_id: uuid.UUID):
    return await pool.fetch(
        """
        SELECT hr.*, gp.position
        FROM hand_review hr
        JOIN game_player gp USING (game_player_id)
        WHERE gp.player_id = $1
        ORDER BY hr.game_player_id
        """,
        player_id,
    )


async def _queued(pool, player_id: uuid.UUID) -> reviews_db.QueuedSeat:
    row = await pool.fetchrow(
        """
        SELECT hr.game_player_id, gp.game_id, gp.position
        FROM hand_review hr
        JOIN game_player gp USING (game_player_id)
        WHERE gp.player_id = $1 AND hr.status = 'queued'
        """,
        player_id,
    )
    return reviews_db.QueuedSeat(row["game_player_id"], row["game_id"], row["position"])


async def _guest_then_registered(pool, seed: int = 0) -> uuid.UUID:
    """Play a hand as guests, then register seat 1's player."""
    _, players = await _play_hand(pool, seed, PARTNER_BY_CALLED_ACE)
    await _register(pool, players[1])
    await reviews_db.enqueue_player(pool, players[1])
    return players[1]


@needs_db
async def test_a_finished_hand_queues_only_registered_seats(db_app):
    _, pool = db_app
    # Seat 1's player has an account before the hand is dealt.
    registered = uuid.uuid4()
    from server.services.persistence.players import ensure_player

    await ensure_player(pool, registered, None)
    await _register(pool, registered)

    _, players = await _play_hand(
        pool, 0, PARTNER_BY_CALLED_ACE, seat_players={1: registered}
    )

    rows = await _review_rows(pool, registered)
    assert [(r["position"], r["status"]) for r in rows] == [(1, "queued")]
    for seat, guest in players.items():
        if seat != 1:
            assert await _review_rows(pool, guest) == []


@needs_db
async def test_registering_queues_the_hands_already_played(db_app):
    _, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    _, more = await _play_hand(
        pool, 1, PARTNER_BY_CALLED_ACE, seat_players={3: players[1]}
    )
    assert await _review_rows(pool, players[1]) == []

    await _register(pool, players[1])
    await reviews_db.enqueue_player(pool, players[1])
    # Queueing twice changes nothing.
    await reviews_db.enqueue_player(pool, players[1])

    rows = await _review_rows(pool, players[1])
    assert [(r["position"], r["status"]) for r in rows] == [
        (1, "queued"),
        (3, "queued"),
    ]
    assert await _review_rows(pool, more[1]) == []


@needs_db
async def test_the_worker_stores_a_review_under_its_model(db_app, agent):
    _, pool = db_app
    player = await _guest_then_registered(pool)
    job = await _queued(pool, player)
    model_id = await pool.fetchval("SELECT ai_model_id FROM ai_model LIMIT 1")

    await review_worker.review_queued_seat(pool, agent, model_id, job)

    (row,) = await _review_rows(pool, player)
    assert row["status"] == "reviewed"
    assert row["ai_model_id"] == model_id
    assert row["duration_ms"] is not None and row["evaluations"] > 0
    decisions = await pool.fetch(
        "SELECT * FROM decision_review WHERE game_player_id = $1 ORDER BY index",
        job.game_player_id,
    )
    assert decisions and all(d["ai_model_id"] == model_id for d in decisions)
    assert [d["index"] for d in decisions] == list(range(len(decisions)))
    # The summary is the decisions counted up.
    open_decisions = [d for d in decisions if not d["is_forced"]]
    assert row["decisions"] == len(open_decisions)
    assert row["decisions"] == (
        row["agreed"] + row["even"] + row["slight"] + row["costly"] + row["better"]
    )
    differences = [d["difference"] for d in decisions if d["difference"] is not None]
    assert row["cost"] == pytest.approx(-sum(d for d in differences if d < 0), abs=0.05)
    assert row["gained"] == pytest.approx(
        sum(d for d in differences if d > 0), abs=0.05
    )
    assert all(d["oracle_score"] is not None for d in decisions)

    # A second review replaces the first instead of adding to it.
    await review_worker.review_queued_seat(pool, agent, model_id, job)
    again = await pool.fetchval(
        "SELECT count(*) FROM decision_review WHERE game_player_id = $1",
        job.game_player_id,
    )
    assert again == len(decisions)


@needs_db
async def test_the_worker_takes_the_newest_hand_first(db_app):
    _, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    await _play_hand(pool, 1, PARTNER_BY_CALLED_ACE, seat_players={2: players[1]})
    await _register(pool, players[1])
    await reviews_db.enqueue_player(pool, players[1])

    job = await reviews_db.next_queued(pool)

    assert job is not None
    newest = await pool.fetchval("SELECT max(game_player_id) FROM hand_review")
    assert job.game_player_id == newest


@needs_db
async def test_a_hand_that_does_not_replay_fails_at_once(db_app, agent, monkeypatch):
    _, pool = db_app
    player = await _guest_then_registered(pool)
    job = await _queued(pool, player)

    def refuse(*args, **kwargs):
        raise ReplayError("PLAY QC: illegal for seat 1")

    monkeypatch.setattr(review_worker, "review_hand", refuse)
    await review_worker.review_queued_seat(pool, agent, 1, job)

    (row,) = await _review_rows(pool, player)
    assert row["status"] == "failed"
    assert row["failure"] == "does not replay: PLAY QC: illegal for seat 1"
    assert row["attempts"] == 1


@needs_db
async def test_an_unexpected_failure_is_retried_then_given_up(
    db_app, agent, monkeypatch
):
    _, pool = db_app
    player = await _guest_then_registered(pool)
    job = await _queued(pool, player)

    def explode(*args, **kwargs):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(review_worker, "review_hand", explode)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        await review_worker.review_queued_seat(pool, agent, 1, job)
        (row,) = await _review_rows(pool, player)
        assert row["attempts"] == attempt
        assert row["status"] == ("failed" if attempt == MAX_ATTEMPTS else "queued")
    assert row["failure"] == "RuntimeError: out of memory"


@needs_db
async def test_a_hand_with_an_ai_made_call_is_skipped(db_app, agent):
    _, pool = db_app
    player = await _guest_then_registered(pool)
    job = await _queued(pool, player)
    # The AI made one of this seat's decisions and no play or pick is
    # flagged as substituted: it was the call or the bury.
    await pool.execute(
        "UPDATE game_player SET ai_actions = 1 WHERE game_player_id = $1",
        job.game_player_id,
    )

    await review_worker.review_queued_seat(pool, agent, 1, job)

    (row,) = await _review_rows(pool, player)
    assert row["status"] == "skipped"


async def test_the_worker_rides_out_a_database_outage(monkeypatch):
    calls = []

    async def unreachable(pool):
        calls.append(len(calls))
        if len(calls) == 1:
            raise ConnectionResetError("connection lost")
        raise asyncio.CancelledError  # stop the loop on its second look

    async def no_wait(seconds):
        return None

    monkeypatch.setattr(reviews_db, "next_queued", unreachable)
    monkeypatch.setattr(review_worker.asyncio, "sleep", no_wait)

    # The first look fails; the worker comes back for a second.
    with pytest.raises(asyncio.CancelledError):
        await review_worker.run_review_worker(object(), "unused", 1)  # pyright: ignore[reportArgumentType]

    assert calls == [0, 1]


def _table(in_progress: bool, connected: bool) -> Table:
    table = Table(id=str(uuid.uuid4()), name="t")
    if in_progress:
        table.game = Game(seed=1)
    sockets = {object()} if connected else set()
    table.clients["h"] = ClientConn(
        client_id="h",
        display_name="h",
        seat=1,
        sockets=sockets,  # pyright: ignore[reportArgumentType]
    )
    return table


def test_the_worker_stands_aside_for_more_than_one_table_in_play(app):
    def seat(table: Table) -> None:
        tables.tables[table.id] = table

    assert review_worker.tables_in_play() == 0
    seat(_table(in_progress=True, connected=True))
    # Neither a table between hands nor one everybody has left counts.
    seat(_table(in_progress=False, connected=True))
    seat(_table(in_progress=True, connected=False))
    assert review_worker.tables_in_play() == 1
    assert not review_worker.is_paused()

    seat(_table(in_progress=True, connected=True))
    assert review_worker.tables_in_play() == 2
    assert review_worker.is_paused()
