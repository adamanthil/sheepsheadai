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
        closed: bool = True,
    ) -> None:
        n = len(scores)
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
                                     starting_hand_id, is_picker, score)
            SELECT g, $2, 'p', 1, $3, pk, sc
            FROM unnest($1::uuid[], $4::bool[], $5::smallint[]) AS t(g, pk, sc)
            """,
            games,
            pid,
            self.cardset,
            picker or [False] * n,
            scores,
        )

    async def cleanup(self):
        await self.pool.execute(
            "DELETE FROM game_player WHERE player_id = ANY($1)", self.players
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
    assert s["pph"] == pytest.approx(1.0)
    assert s["win_pct"] == pytest.approx(0.5)
    assert s["pick_pct"] == pytest.approx(0.5)
    assert (s["rank"], s["qualifies_in"], s["min_hands"]) == (None, 46, 50)


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
