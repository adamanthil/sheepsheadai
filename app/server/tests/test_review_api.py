"""The AI review endpoints, over hands played through the real hooks and
reviewed by the worker with an untrained model.

Opt-in like test_hand_history: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import os
import uuid

import httpx2
import pytest

from server.runtime import review_worker
from server.services.high_trump import high_trump, played_before
from server.services.persistence import review_reads
from server.services.persistence import reviews as reviews_db
from server.services.persistence.review_reads import STANDING_SQL
from server.services.persistence.sessions import create_session
from server.services.review_bands import is_notable, standing
from server.tests.test_hand_history import _play_hand
from sheepshead.agent.ppo import PPOAgent
from sheepshead.game import ACTIONS, PARTNER_BY_CALLED_ACE

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

needs_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


@pytest.fixture(scope="module")
def agent() -> PPOAgent:
    return PPOAgent(len(ACTIONS), critic_mode="oracle")


async def _verified(pool, player_id: uuid.UUID, admin: bool = False) -> dict:
    """Give ``player_id`` a confirmed account; returns its auth header."""
    tag = player_id.hex[:10]
    await pool.execute(
        "INSERT INTO account (player_id, username, email, password_hash, "
        "email_verified_at, time_created, last_updated, is_admin) "
        "VALUES ($1, $2, $3, 'x', now(), now(), now(), $4)",
        player_id,
        f"rv_{tag}",
        f"{tag}@example.com",
        admin,
    )
    await reviews_db.enqueue_player(pool, player_id)
    return {"Authorization": f"Bearer {await create_session(pool, player_id)}"}


async def _work_queue(pool, agent, player_id: uuid.UUID) -> None:
    """Review every queued hand of ``player_id``."""
    model_id = await pool.fetchval("SELECT ai_model_id FROM ai_model LIMIT 1")
    rows = await pool.fetch(
        """
        SELECT hr.game_player_id, gp.game_id, gp.position
        FROM hand_review hr JOIN game_player gp USING (game_player_id)
        WHERE gp.player_id = $1 AND hr.status = 'queued'
        """,
        player_id,
    )
    for row in rows:
        job = reviews_db.QueuedSeat(row["game_player_id"], row["game_id"], row[2])
        await review_worker.review_queued_seat(pool, agent, model_id, job)


def _review_kind(decision: dict) -> str:
    """The kind a decision is totalled under: an under card with the call."""
    return "call" if decision["kind"] == "under" else decision["kind"]


def _client(app) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://t"
    )


@needs_db
async def test_history_rows_say_where_the_review_stands(db_app, agent):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    auth = await _verified(pool, players[1])

    async with _client(app) as client:
        waiting = (await client.get("/api/account/hands", headers=auth)).json()
        await _work_queue(pool, agent, players[1])
        done = (await client.get("/api/account/hands", headers=auth)).json()

    assert waiting["hands"][0]["review"] == {
        "status": "queued",
        "cost": None,
        "gained": None,
        "net": None,
        "notable": False,
    }
    review = done["hands"][0]["review"]
    assert review["status"] == "reviewed"
    # The row and the hand's own page band the same decisions.
    game_id = done["hands"][0]["game_id"]
    async with _client(app) as client:
        page = (
            await client.get(f"/api/account/hands/{game_id}/review", headers=auth)
        ).json()
    summary = page["summary"]
    for figure in ("cost", "gained", "net", "notable"):
        assert review[figure] == summary[figure]
    assert review["notable"] == is_notable(
        summary["cost"], summary["gained"], summary["costly"], summary["better"]
    )
    assert review["net"] == pytest.approx(review["gained"] - review["cost"], abs=0.05)
    assert review["cost"] >= 0 and review["gained"] >= 0


@needs_db
async def test_a_hand_review_lists_every_decision_of_the_seat(db_app, agent):
    app, pool = db_app
    game, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE, ai_seats=(4, 5))
    auth = await _verified(pool, players[2])
    await _work_queue(pool, agent, players[2])
    game_id = await pool.fetchval(
        "SELECT game_id FROM game_player WHERE player_id = $1", players[2]
    )

    async with _client(app) as client:
        r = await client.get(f"/api/account/hands/{game_id}/review", headers=auth)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["seat"] == 2
    assert body["hand"]["game_id"] == str(game_id)
    assert body["ai_model"] == "test-model"
    assert (body["newer_game_id"], body["older_game_id"]) == (None, None)
    # The deal, for opening the pick analyzer on it.
    assert len(body["dealt_hand"]) == 6 and len(body["blind"]) == 2
    assert not set(body["dealt_hand"]) & set(body["blind"])
    assert body["partner_mode"] == PARTNER_BY_CALLED_ACE
    assert body["partner_card"] == body["called_card"]
    # Humans are named; the AI seats are not.
    assert [(s["seat"], s["is_ai"], s["is_you"]) for s in body["seats"]] == [
        (1, False, False),
        (2, False, True),
        (3, False, False),
        (4, True, False),
        (5, True, False),
    ]
    assert all((s["name"] is None) == s["is_ai"] for s in body["seats"])

    decisions = body["decisions"]
    assert [d["index"] for d in decisions] == list(range(len(decisions)))
    summary = body["summary"]
    assert summary["decisions"] == sum(d["standing"] != "forced" for d in decisions)
    # The summary is the decisions counted up as they are read.
    for band in ("agreed", "even", "slight", "costly", "better"):
        assert summary[band] == sum(d["standing"] == band for d in decisions)
    plays = [d for d in decisions if d["kind"] == "play"]
    assert len(plays) == 6 and [d["trick"] for d in plays] == list(range(6))
    # Nobody can tell who the partner is before a card is played, and
    # once the table knows it stays known.
    assert not any(d["partner_known"] for d in decisions if d["kind"] != "play")
    known = [d["partner_known"] for d in plays]
    assert known == sorted(known)
    for d in decisions:
        assert (d["difference"] is None) == (d["standing"] in ("agreed", "forced"))
        assert d["policy"][0][0] == d["ai_action"]
    # The last card: nothing held but it, so every trump still out is unseen.
    last = plays[-1]
    assert len(last["hand"]) == 1
    if last["high_trump"]["card"] is None:
        assert last["high_trump"] == {"card": None, "held": False, "unseen": 0}


@needs_db
async def test_a_hand_waiting_for_review_has_no_decisions_yet(db_app):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    auth = await _verified(pool, players[1])
    game_id = await pool.fetchval(
        "SELECT game_id FROM game_player WHERE player_id = $1", players[1]
    )

    async with _client(app) as client:
        r = await client.get(f"/api/account/hands/{game_id}/review", headers=auth)
        other = await client.get(
            f"/api/account/hands/{uuid.uuid4()}/review", headers=auth
        )

    body = r.json()
    assert body["hand"]["review"]["status"] == "queued"
    assert body["summary"] is None and body["decisions"] == []
    assert (other.status_code, other.json()["detail"]) == (404, "hand_not_found")


@needs_db
async def test_someone_elses_hand_is_not_found(db_app, agent):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    await _verified(pool, players[1])
    game_id = await pool.fetchval(
        "SELECT game_id FROM game_player WHERE player_id = $1", players[1]
    )
    _, strangers = await _play_hand(pool, 1, PARTNER_BY_CALLED_ACE)
    stranger = await _verified(pool, strangers[1])

    async with _client(app) as client:
        r = await client.get(f"/api/account/hands/{game_id}/review", headers=stranger)

    assert r.status_code == 404


@needs_db
async def test_review_totals_add_up_across_hands(db_app, agent):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    for seed in (1, 6):
        await _play_hand(
            pool, seed, PARTNER_BY_CALLED_ACE, seat_players={1: players[1]}
        )
    auth = await _verified(pool, players[1])
    await _work_queue(pool, agent, players[1])

    async with _client(app) as client:
        r = await client.get("/api/account/review", headers=auth)

    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["hands_reviewed"], body["hands_waiting"]) == (3, 0)
    assert body["decisions"] == (
        body["agreed"] + body["even"] + body["slight"] + body["costly"] + body["better"]
    )
    assert sum(k["decisions"] for k in body["by_kind"]) == body["decisions"]
    assert sum(k["net"] for k in body["by_kind"]) == pytest.approx(body["net"], abs=0.2)
    cost_most, gained_most = body["cost_most"], body["gained_most"]
    assert len(cost_most["all"]) == body["costly"]
    assert len(gained_most["all"]) == body["better"]
    assert all(d["standing"] == "costly" for d in cost_most["all"])
    assert all(d["standing"] == "better" for d in gained_most["all"])
    costs = [d["difference"] for d in cost_most["all"]]
    assert costs == sorted(costs) and all(c <= -1.0 for c in costs)
    gains = [d["difference"] for d in gained_most["all"]]
    assert gains == sorted(gains, reverse=True) and all(g >= 1.0 for g in gains)
    # Each kind's own list, and its count, cover the same decisions.
    for kind in body["by_kind"]:
        listed = cost_most[kind["kind"]]
        assert len(listed) == kind["costly"]
        assert all(_review_kind(d) == kind["kind"] for d in listed)
        assert len(gained_most[kind["kind"]]) == kind["better"]
    picked, passed = body["picked_ai_passes"], body["passed_ai_picks"]
    assert picked["count"] + passed["count"] == (
        body["pick_decisions"] - body["pick_agreed"]
    )
    assert (len(picked["picks"]), len(passed["picks"])) == (
        picked["count"],
        passed["count"],
    )
    assert picked["score"] == sum(p["score"] for p in picked["picks"])
    assert all(len(p["hand"]) == 6 for p in picked["picks"] + passed["picks"])


@needs_db
async def test_review_lists_are_capped_but_their_counts_are_not(
    db_app, agent, monkeypatch
):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    for seed in (1, 6, 7, 8):
        await _play_hand(
            pool, seed, PARTNER_BY_CALLED_ACE, seat_players={1: players[1]}
        )
    auth = await _verified(pool, players[1])
    await _work_queue(pool, agent, players[1])

    async with _client(app) as client:
        whole = (await client.get("/api/account/review", headers=auth)).json()
        monkeypatch.setattr(review_reads, "REVIEW_LIST_SIZE", 1)
        capped = (await client.get("/api/account/review", headers=auth)).json()

    # An untrained model disagrees with the script all over the place.
    assert whole["costly"] > 1 and whole["picked_ai_passes"]["count"] != 1
    for key in ("costly", "better", "by_kind", "pick_decisions", "pick_agreed"):
        assert capped[key] == whole[key]
    for way in ("cost_most", "gained_most"):
        for name, listed in whole[way].items():
            # The one kept is the largest, of everything and of each kind.
            assert capped[way][name] == listed[:1]
    for way in ("picked_ai_passes", "passed_ai_picks"):
        assert capped[way]["picks"] == whole[way]["picks"][:1]
        assert (capped[way]["count"], capped[way]["score"]) == (
            whole[way]["count"],
            whole[way]["score"],
        )


@needs_db
async def test_stored_decisions_are_banded_as_the_engine_bands_them(db_app):
    _, pool = db_app
    differences = [None, 0.0, 0.2, -0.2, 0.3, -0.3, 0.9, -0.9, 1.0, -1.0, 2.5, -2.5]
    cases = [(forced, d) for forced in (False, True) for d in differences]

    rows = await pool.fetch(
        f"""
        SELECT {STANDING_SQL} AS standing
        FROM unnest($1::boolean[], $2::real[])
             WITH ORDINALITY AS d(is_forced, difference, n)
        ORDER BY d.n
        """,
        [forced for forced, _ in cases],
        [d for _, d in cases],
    )

    assert [r["standing"] for r in rows] == [standing(d, forced) for forced, d in cases]


@needs_db
async def test_review_endpoints_need_a_confirmed_account(db_app):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    token = await create_session(pool, players[1])
    guest = {"Authorization": f"Bearer {token}"}

    async with _client(app) as client:
        totals = await client.get("/api/account/review", headers=guest)
        hand = await client.get(
            f"/api/account/hands/{uuid.uuid4()}/review", headers=guest
        )

    assert (totals.status_code, totals.json()["detail"]) == (403, "account_required")
    assert (hand.status_code, hand.json()["detail"]) == (403, "account_required")


@needs_db
async def test_the_queue_page_is_for_administrators_only(db_app, agent):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    member = await _verified(pool, players[1])
    admin = await _verified(pool, players[2], admin=True)
    await _work_queue(pool, agent, players[1])

    async with _client(app) as client:
        refused = await client.get("/api/admin/review-queue", headers=member)
        r = await client.get("/api/admin/review-queue", headers=admin)

    assert refused.status_code == 404
    assert r.status_code == 200, r.text
    body = r.json()
    # Other tests share the database, so only this test's rows are certain.
    assert body["reviewed"] >= 1 and body["waiting"] >= 1
    assert body["total"] >= body["reviewed"] + body["waiting"]
    assert body["reviewed_24h"] >= 1 and body["average_ms_24h"] is not None
    assert body["oldest_queued"] is not None and body["last_reviewed"] is not None
    assert body["ai_model"] == "test-model"
    assert (body["paused"], body["tables_in_play"]) == (False, 0)


def test_the_high_trump_is_the_top_trump_not_yet_played():
    hand = ["JC", "10D", "KC"]
    tricks = [
        [(1, "QC"), (2, "QS"), (3, "QH"), (4, "QD"), (5, "7C")],
        [(1, "AD"), (2, "9D"), (3, "8D"), (4, "KD"), (5, "8C")],
    ]

    before_any = high_trump(hand + ["AH", "8S", "QS"], played_before(tricks, None, []))
    assert (before_any.card, before_any.held) == ("QC", False)
    # 14 trump, three of them in this hand.
    assert before_any.unseen == 11

    # Third trick, with the J of diamonds already on the table.
    now = high_trump(hand, played_before(tricks, 2, [(1, "JD")]))
    assert (now.card, now.held) == ("JC", True)
    # Out and unseen: JS, JH and 7D.
    assert now.unseen == 3

    # A picker's own bury is gone too; the same trump is still out for
    # anyone else.
    as_picker = high_trump(["10D"], played_before(tricks, 2, [(1, "JD")]), ["JC", "JS"])
    assert (as_picker.card, as_picker.held, as_picker.unseen) == ("JH", False, 2)


def test_no_trump_left():
    every_trump = [
        "QC", "QS", "QH", "QD", "JC", "JS", "JH", "JD",
        "AD", "10D", "KD", "9D", "8D", "7D",
    ]  # fmt: skip
    gone = high_trump(["KC"], every_trump)
    assert (gone.card, gone.held, gone.unseen) == (None, False, 0)
