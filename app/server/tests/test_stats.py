"""Hand statistics and the leaderboard over seeded hand records.

Opt-in like test_api_flow: set TEST_DATABASE_URL to a migrated database.
The seeded rows are removed afterwards, so reruns against a long-lived
local database don't crowd the board.
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from server.services.persistence.sessions import create_session

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


class Seeder:
    def __init__(self, pool):
        self.pool = pool
        self.table_id = uuid.uuid4()
        self.players: list[uuid.UUID] = []
        self.cardsets: list[int] = []

    async def start(self):
        self.cardset = await self.pool.fetchval(
            "INSERT INTO cardset (cards_hash) VALUES ($1) RETURNING cardset_id",
            uuid.uuid4().hex,
        )
        await self.pool.execute(
            "INSERT INTO game_table (game_table_id, name, time_created) "
            "VALUES ($1, 'stats', now())",
            self.table_id,
        )

    async def player(
        self, *, username: str | None = None, verified: bool = True
    ) -> uuid.UUID:
        pid = uuid.uuid4()
        self.players.append(pid)
        await self.pool.execute(
            "INSERT INTO player (player_id, name, time_created, last_updated) "
            "VALUES ($1, 'Same Name', now(), now())",
            pid,
        )
        if username is not None:
            await self.pool.execute(
                "INSERT INTO account (player_id, username, email, password_hash, "
                "email_verified_at, time_created, last_updated) "
                "VALUES ($1, $2, $3, 'x', CASE WHEN $4 THEN now() END, now(), now())",
                pid,
                username,
                f"{username}@example.com",
                verified,
            )
        return pid

    async def hands(
        self,
        pid: uuid.UUID,
        scores: list[int | None],
        *,
        picker: list[bool] | None = None,
        leaster: list[bool] | None = None,
        ai_actions: list[int] | None = None,
        excused: list[int] | None = None,
        substituted: list[bool] | None = None,
        dealt: list[list[str]] | None = None,
        other_humans: list[int] | None = None,
        ai_seat: list[str | None] | None = None,
        closed: bool = True,
    ) -> None:
        n = len(scores)
        hand_ids = [self.cardset] * n
        if dealt is not None:
            hand_ids = [await self.hand(codes) for codes in dealt]
        games = [uuid.uuid4() for _ in range(n)]
        await self.pool.execute(
            """
            INSERT INTO game (game_id, game_table_id, is_double_on_the_bump,
                              is_called_partner, is_leaster, time_created,
                              time_closed, blind_id)
            SELECT g, $2, true, true, l, now(), CASE WHEN $3 THEN now() END, $4
            FROM unnest($1::uuid[], $5::bool[]) AS t(g, l)
            """,
            games,
            self.table_id,
            closed,
            self.cardset,
            leaster or [False] * n,
        )
        await self.pool.execute(
            """
            INSERT INTO game_player (game_id, player_id, name, position,
                                     starting_hand_id, is_picker, score,
                                     ai_actions, ai_actions_excused,
                                     is_substituted_pick)
            SELECT g, $2, 'p', 1, h, pk, sc, ai, ex, sub
            FROM unnest($1::uuid[], $3::bigint[], $4::bool[], $5::smallint[],
                        $6::smallint[], $7::smallint[], $8::bool[])
                 AS t(g, h, pk, sc, ai, ex, sub)
            """,
            games,
            pid,
            hand_ids,
            picker or [False] * n,
            scores,
            ai_actions or [0] * n,
            excused or [0] * n,
            substituted or [False] * n,
        )
        # Other people dealt into the hand; the remaining seats are left
        # empty, which the stats read the same as seats dealt to the AI.
        for game, count in zip(games, other_humans or [0] * n):
            for position in range(2, 2 + count):
                await self.pool.execute(
                    "INSERT INTO game_player (game_id, player_id, name, position, "
                    "starting_hand_id) VALUES ($1, $2, 'o', $3, $4)",
                    game,
                    await self.player(),
                    position,
                    self.cardset,
                )

        # One AI-dealt seat per hand, left to the AI ("ai") or taken over
        # by a person who then bid ("human_bid") or played a card
        # ("human_play") in it.
        for game, kind in zip(games, ai_seat or [None] * n):
            if kind is not None:
                await self.ai_row(game, kind)

    async def ai_row(self, game: uuid.UUID, kind: str) -> None:
        gp_id = await self.pool.fetchval(
            "INSERT INTO game_player (game_id, ai_player_id, name, position, "
            "starting_hand_id, is_substituted_pick) "
            "VALUES ($1, (SELECT min(ai_player_id) FROM ai_player), 'ai', 5, "
            "$2, $3) RETURNING game_player_id",
            game,
            self.cardset,
            kind == "human_bid",
        )
        trick_id = await self.pool.fetchval(
            "INSERT INTO trick (game_id, index, lead_player_id, points) "
            "VALUES ($1, 0, $2, 0) RETURNING trick_id",
            game,
            gp_id,
        )
        await self.pool.execute(
            "INSERT INTO trick_card (trick_id, card_id, game_player_id, index, "
            "is_substituted) "
            "VALUES ($1, (SELECT card_id FROM card WHERE code = 'QC'), $2, 0, $3)",
            trick_id,
            gp_id,
            kind == "human_play",
        )

    async def hand(self, codes: list[str]) -> int:
        cardset = await self.pool.fetchval(
            "INSERT INTO cardset (cards_hash) VALUES ($1) RETURNING cardset_id",
            uuid.uuid4().hex,
        )
        self.cardsets.append(cardset)
        await self.pool.execute(
            "INSERT INTO cardset_card (cardset_id, card_id) "
            "SELECT $1, card_id FROM card WHERE code = ANY($2)",
            cardset,
            codes,
        )
        return cardset

    async def cleanup(self):
        games = "SELECT game_id FROM game WHERE game_table_id = $1"
        await self.pool.execute(
            "DELETE FROM trick_card WHERE trick_id IN "
            f"(SELECT trick_id FROM trick WHERE game_id IN ({games}))",
            self.table_id,
        )
        await self.pool.execute(
            f"DELETE FROM trick WHERE game_id IN ({games})", self.table_id
        )
        await self.pool.execute(
            f"DELETE FROM game_player WHERE game_id IN ({games})", self.table_id
        )
        await self.pool.execute(
            "DELETE FROM game WHERE game_table_id = $1", self.table_id
        )
        await self.pool.execute(
            "DELETE FROM game_table WHERE game_table_id = $1", self.table_id
        )
        await self.pool.execute(
            "DELETE FROM player WHERE player_id = ANY($1)", self.players
        )
        await self.pool.execute(
            "DELETE FROM cardset_card WHERE cardset_id = ANY($1)", self.cardsets
        )
        await self.pool.execute(
            "DELETE FROM cardset WHERE cardset_id = ANY($1)", self.cardsets
        )


@pytest.fixture
async def seeded(db_app):
    app, pool = db_app
    seeder = Seeder(pool)
    await seeder.start()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            yield seeder, c
    finally:
        await seeder.cleanup()


def _tag() -> str:
    return uuid.uuid4().hex[:8]


async def test_personal_stats_count_only_finished_scored_hands(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"stat_{_tag()}")
    await seeder.hands(
        pid,
        [4, -2, 0, 2],
        picker=[True, True, False, False],
        leaster=[False, False, False, True],
    )
    await seeder.hands(pid, [None])  # passed-out doublers deal: closed, unscored
    await seeder.hands(pid, [6], closed=False)  # still being played
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    assert r.status_code == 200, r.text
    s = r.json()
    assert (s["hands"], s["total"], s["leaster_hands"]) == (4, 4, 1)
    assert s["sph"] == pytest.approx(1.0)
    assert s["win_pct"] == pytest.approx(0.5)
    assert s["pick_pct"] == pytest.approx(0.5)
    assert (s["rank"], s["qualifies_in"], s["min_hands"]) == (None, 46, 50)
    assert (s["abandoned_hands"], s["ai_assisted_hands"]) == (0, 0)
    assert s["completion_rate"] == 1.0
    assert (s["picks"], s["trump_per_pick"], s["queens_per_pick"]) == (2, 0, 0)


async def test_abandoned_hands_count_losses_but_not_wins(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"quit_{_tag()}")
    # AI decisions (not excused) per hand: 2 is within the threshold, 3 is
    # over it; excused moves never count.
    await seeder.hands(
        pid,
        [4, 6, -4, 2, 3],
        ai_actions=[2, 3, 3, 5, 0],
        excused=[0, 0, 0, 4, 0],
    )
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    s = r.json()
    # The +6 is zeroed; the abandoned -4 counts in full.
    assert (s["hands"], s["total"], s["forfeited_score"]) == (5, 5, 6)
    assert s["win_pct"] == pytest.approx(0.6)
    assert (s["abandoned_hands"], s["ai_assisted_hands"]) == (2, 4)
    assert s["completion_rate"] == pytest.approx(0.6)
    assert s["abandon_threshold"] == 2


async def test_splits_by_who_else_was_dealt_in(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"split_{_tag()}")
    # Vs the AI: two hands alone with the AI, one of them an abandoned win
    # (counted as 0). With people: one or four other humans dealt in.
    await seeder.hands(
        pid,
        [4, 6, -2, 2, -6],
        ai_actions=[0, 3, 0, 0, 0],
        other_humans=[0, 0, 1, 4, 1],
    )
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    s = r.json()
    assert s["vs_ai"] == {
        "hands": 2,
        "sph": pytest.approx(2.0),
        "sph_margin": None,  # too few hands for a margin
        "win_pct": pytest.approx(0.5),
    }
    assert s["with_people"]["hands"] == 3
    assert s["with_people"]["sph"] == pytest.approx(-2.0)
    assert s["with_people"]["win_pct"] == pytest.approx(1 / 3)
    # The splits add back up to the headline figures.
    assert s["vs_ai"]["hands"] + s["with_people"]["hands"] == s["hands"]


async def test_a_person_in_an_ai_seat_rules_out_vs_ai(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"takeover_{_tag()}")
    # All three hands had an AI-dealt seat; in two a person took it over
    # mid-hand and bid or played in it, which makes them hands with people.
    await seeder.hands(pid, [2, 4, -6], ai_seat=["ai", "human_bid", "human_play"])
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    s = r.json()
    assert (s["vs_ai"]["hands"], s["vs_ai"]["sph"]) == (1, pytest.approx(2.0))
    assert (s["with_people"]["hands"], s["with_people"]["sph"]) == (
        2,
        pytest.approx(-1.0),
    )


async def test_split_margin_is_a_95_percent_interval(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"margin_{_tag()}")
    scores = [2, -2] * 10  # mean 0, sample sd sqrt(80/19)
    await seeder.hands(pid, scores)
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    vs_ai = r.json()["vs_ai"]
    assert vs_ai["sph"] == pytest.approx(0.0)
    assert vs_ai["sph_margin"] == pytest.approx(1.96 * (80 / 19) ** 0.5 / 20**0.5)
    assert r.json()["with_people"] == {
        "hands": 0,
        "sph": None,
        "sph_margin": None,
        "win_pct": None,
    }


async def test_picking_counts_trump_and_queens_in_the_dealt_hand(seeded):
    seeder, client = seeded
    pid = await seeder.player(username=f"pick_{_tag()}")
    await seeder.hands(
        pid,
        [2, -4, 2, 1],
        picker=[True, True, True, False],
        substituted=[False, False, True, False],
        dealt=[
            # 5 trump (QC QS JD AD 7D), 2 queens.
            ["QC", "QS", "JD", "AD", "7D", "AC"],
            # 1 trump (JH), no queens.
            ["JH", "AC", "10S", "KH", "9C", "7S"],
            # The AI picked for them: left out.
            ["QC", "QS", "QH", "QD", "JC", "JS"],
            # Not a pick: left out.
            ["QC", "QS", "QH", "QD", "JC", "JS"],
        ],
    )
    token = await create_session(seeder.pool, pid)

    r = await client.get(
        "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
    )

    s = r.json()
    assert s["picks"] == 2
    assert s["trump_per_pick"] == pytest.approx(3.0)
    assert s["queens_per_pick"] == pytest.approx(1.0)


async def test_stats_need_a_verified_account(seeded):
    seeder, client = seeded
    guest = await seeder.player()
    unverified = await seeder.player(username=f"unv_{_tag()}", verified=False)
    for pid, detail in ((guest, "account_required"), (unverified, "email_unverified")):
        token = await create_session(seeder.pool, pid)
        r = await client.get(
            "/api/account/stats", headers={"Authorization": f"Bearer {token}"}
        )
        assert (r.status_code, r.json()["detail"]) == (403, detail)


async def test_leaderboard_serves_only_the_top_of_eligible_accounts(seeded):
    seeder, client = seeded
    tag = _tag()
    # 22 eligible players. Totals rise with i, but win rate falls with it, so
    # the total and win-rate boards run in opposite directions.
    eligible = []
    for i in range(22):
        pid = await seeder.player(username=f"lb{i:02d}_{tag}")
        # total = (i+1)(50-i) - i, increasing over 0..21; win rate (50-i)/50.
        await seeder.hands(pid, [i + 1] * (50 - i) + [-1] * i)
        eligible.append(pid)
    short = await seeder.player(username=f"short_{tag}")
    await seeder.hands(short, [100] * 49)
    hidden = await seeder.player(username=f"hidden_{tag}", verified=False)
    await seeder.hands(hidden, [100] * 60)
    guest = await seeder.player()
    await seeder.hands(guest, [100] * 60)

    r = await client.get("/api/leaderboard")
    board = r.json()
    ours = [row for row in board["rows"] if row["username"].endswith(tag)]
    assert board["sort"] == "total" and board["min_hands"] == 50
    assert len(board["rows"]) == 20
    # Highest total first, and nobody under 50 hands or unverified.
    assert [row["username"] for row in ours][:3] == [
        f"lb21_{tag}",
        f"lb20_{tag}",
        f"lb19_{tag}",
    ]
    names = {row["username"] for row in board["rows"]}
    assert f"short_{tag}" not in names and f"hidden_{tag}" not in names
    assert "Same Name" not in names

    # lb00 has the lowest total: off the page, but shown to its owner.
    token = await create_session(seeder.pool, eligible[0])
    r = await client.get(
        "/api/leaderboard", headers={"Authorization": f"Bearer {token}"}
    )
    you = r.json()["you"]
    assert you["username"] == f"lb00_{tag}" and you["is_you"] is True
    assert you["rank"] > 20
    assert not any(row["is_you"] for row in r.json()["rows"])

    # Re-sorting by win rate puts lb00 first: best-first on every column.
    r = await client.get(
        "/api/leaderboard",
        params={"sort": "win_pct"},
        headers={"Authorization": f"Bearer {token}"},
    )
    top = r.json()["rows"][0]
    assert (top["username"], top["is_you"]) == (f"lb00_{tag}", True)
    assert r.json()["you"] is None

    bad = await client.get("/api/leaderboard", params={"sort": "worst"})
    assert bad.status_code == 422


async def test_leaderboard_ranks_by_counted_scores(seeded):
    seeder, client = seeded
    tag = _tag()
    quitter = await seeder.player(username=f"aquit_{tag}")
    await seeder.hands(quitter, [5] * 50, ai_actions=[3] * 50)
    steady = await seeder.player(username=f"asteady_{tag}")
    await seeder.hands(steady, [1] * 50)

    r = await client.get("/api/leaderboard")

    ours = [row for row in r.json()["rows"] if row["username"].endswith(tag)]
    assert [(row["username"], row["total"]) for row in ours] == [
        (f"asteady_{tag}", 50),
        (f"aquit_{tag}", 0),
    ]
    assert ours[1]["win_pct"] == 0
