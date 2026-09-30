"""The account hand history, over hands played through the real hooks.

Opt-in like test_api_flow: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import os
import random
import uuid

import httpx
import pytest

from server.runtime.models import ClientConn, Occupant, Table
from server.services.persistence import history
from server.services.persistence.game_table import ensure_game_table
from server.services.persistence.games import fire_game_hooks
from server.services.persistence.hand_setup import persist_started_game
from server.services.persistence.players import ensure_player
from server.services.persistence.sessions import create_session
from server.services.persistence.snapshots import (
    capture_post_state,
    capture_pre_state,
)
from sheepshead.game import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


async def _play_hand(
    pool, seed: int, partner_mode: int, ai_seats: tuple[int, ...] = ()
) -> tuple[Game, dict[int, uuid.UUID]]:
    """Deal ``Game(seed)`` at a fresh table and play it out with uniformly
    random legal moves (seeded), firing the persistence hooks for every
    move. Returns the finished game and each human seat's player id."""
    table = Table(
        id=str(uuid.uuid4()),
        name=f"history {seed}",
        rules={"partnerMode": partner_mode},
    )
    players: dict[int, uuid.UUID] = {}
    for seat in range(1, 6):
        if seat in ai_seats:
            table.occupants[f"ai{seat}"] = Occupant(
                id=f"ai{seat}", display_name="AI", is_ai=True
            )
            table.seats[seat] = f"ai{seat}"
            continue
        pid = uuid.uuid4()
        await ensure_player(pool, pid, None)
        players[seat] = pid
        cid = f"h{seat}"
        table.clients[cid] = ClientConn(
            client_id=cid, display_name=cid, seat=seat, player_id=str(pid)
        )
        table.seats[seat] = cid
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    table.game = game
    table.status = "playing"
    await ensure_game_table(pool, table.id, table.name, False)
    await persist_started_game(pool, table, game)

    rng = random.Random(seed)
    while not game.is_done():
        seat = next(p.position for p in game.players if p.get_valid_action_ids())
        player = game.players[seat - 1]
        pre = capture_pre_state(game)
        assert player.act(rng.choice(sorted(player.get_valid_action_ids())))
        post = capture_post_state(game)
        await fire_game_hooks(table, pre, post, seat=seat, by_ai=seat in ai_seats)
    return game, players


def _engine_view(game: Game, seat: int) -> tuple[str, int, str]:
    """(role, card points taken, scope) for ``seat``, straight from the engine."""
    if game.is_leaster:
        return "leaster", game.points_taken[seat - 1], "own"
    picker_team = seat == game.picker or (
        not game.alone_called and seat == game.partner
    )
    points = (
        game.get_final_picker_points()
        if picker_team
        else game.get_final_defender_points()
    )
    if seat == game.picker:
        role = "picker"
    elif picker_team:
        role = "partner"
    else:
        role = "defender"
    return role, int(points), "team"


# Seeds whose random play reaches each case (found by search; play is
# deterministic given the seed).
CASES = [
    pytest.param(PARTNER_BY_CALLED_ACE, 0, id="called-ace"),
    pytest.param(PARTNER_BY_CALLED_ACE, 1, id="called-ace-alone"),
    pytest.param(PARTNER_BY_CALLED_ACE, 6, id="picker-team-takes-nothing"),
    pytest.param(PARTNER_BY_CALLED_ACE, 16, id="leaster"),
    pytest.param(PARTNER_BY_CALLED_ACE, 188, id="picker-is-partner"),
    pytest.param(PARTNER_BY_JD, 0, id="jd"),
    pytest.param(PARTNER_BY_JD, 1, id="jd-alone"),
    pytest.param(PARTNER_BY_JD, 6, id="jd-picker-holds-jd"),
    pytest.param(PARTNER_BY_JD, 7, id="jd-buried"),
    pytest.param(PARTNER_BY_JD, 21, id="jd-picker-team-takes-nothing"),
]


@pytest.mark.parametrize(("partner_mode", "seed"), CASES)
async def test_derived_points_match_the_engine(db_app, partner_mode, seed):
    _, pool = db_app
    game, players = await _play_hand(pool, seed, partner_mode)

    for seat, pid in players.items():
        page = await history.hand_history(pool, pid)
        (hand,) = page["hands"]
        role, points, scope = _engine_view(game, seat)
        assert (hand["role"], hand["points_taken"], hand["points_scope"]) == (
            role,
            points,
            scope,
        ), f"seat {seat}"
        assert hand["score"] == game.players[seat - 1].get_score()
        assert hand["opponents"] == {"humans": 4, "ai": 0}


async def test_seeds_still_reach_their_cases(db_app):
    # Guards the CASES table: a rules change that reroutes random play
    # would otherwise quietly stop covering the rare cases.
    _, pool = db_app
    game, _ = await _play_hand(pool, 6, PARTNER_BY_JD)
    assert game.partner == game.picker and "JD" not in game.bury
    game, _ = await _play_hand(pool, 7, PARTNER_BY_JD)
    assert game.partner == game.picker and "JD" in game.bury
    game, _ = await _play_hand(pool, 188, PARTNER_BY_CALLED_ACE)
    assert game.partner == game.picker and not game.alone_called
    game, _ = await _play_hand(pool, 21, PARTNER_BY_JD)
    assert not game.is_leaster and not game.get_final_picker_points()
    game, _ = await _play_hand(pool, 16, PARTNER_BY_CALLED_ACE)
    assert game.is_leaster
    game, _ = await _play_hand(pool, 1, PARTNER_BY_CALLED_ACE)
    assert game.alone_called


async def test_history_flags_counts_and_opponent_mix(db_app):
    _, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE, ai_seats=(4, 5))
    pid = players[1]
    await pool.execute(
        "UPDATE game_player SET ai_actions = 3, score = 2 WHERE player_id = $1", pid
    )

    (hand,) = (await history.hand_history(pool, pid))["hands"]

    assert hand["opponents"] == {"humans": 2, "ai": 2}
    assert (hand["ai_assisted"], hand["abandoned"]) == (True, True)
    assert (hand["score"], hand["counted_score"]) == (2, 0)
    assert hand["table_name"] == "history 0"
    assert hand["time_closed"].tzinfo is not None


async def test_history_pages_newest_first(db_app, monkeypatch):
    _, pool = db_app
    monkeypatch.setattr(history, "PAGE_SIZE", 2)
    pid = uuid.uuid4()
    await ensure_player(pool, pid, None)
    table_id = uuid.uuid4()
    await ensure_game_table(pool, str(table_id), "paging", False)
    blind = await pool.fetchval(
        "INSERT INTO cardset (cards_hash) VALUES ($1) RETURNING cardset_id",
        uuid.uuid4().hex,
    )
    # Five hands, two closing in the same second (the game_id breaks ties).
    for minute, score in ((1, 1), (2, 2), (3, 3), (3, 4), (5, 5)):
        gid = uuid.uuid4()
        await pool.execute(
            "INSERT INTO game (game_id, game_table_id, is_double_on_the_bump, "
            "is_called_partner, is_leaster, time_created, time_closed, blind_id) "
            "VALUES ($1, $2, true, true, false, now(), "
            "timestamp '2026-01-01' + make_interval(mins => $3), $4)",
            gid,
            table_id,
            minute,
            blind,
        )
        await pool.execute(
            "INSERT INTO game_player (game_id, player_id, name, position, "
            "starting_hand_id, is_picker, score) VALUES ($1, $2, 'p', 1, $3, false, $4)",
            gid,
            pid,
            blind,
            score,
        )

    seen: list[int] = []
    cursor = None
    pages = 0
    while True:
        page = await history.hand_history(pool, pid, cursor)
        pages += 1
        seen += [h["score"] for h in page["hands"]]
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert pages == 3
    assert seen[0] == 5 and seen[-1] == 1
    assert sorted(seen[1:3]) == [3, 4] and seen[3] == 2
    with pytest.raises(history.BadCursor):
        await history.hand_history(pool, pid, "not a cursor")


async def test_history_needs_a_verified_account(db_app):
    app, pool = db_app
    guest = uuid.uuid4()
    await ensure_player(pool, guest, None)
    token = await create_session(pool, guest)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get(
            "/api/account/hands", headers={"Authorization": f"Bearer {token}"}
        )
    assert (r.status_code, r.json()["detail"]) == (403, "account_required")


async def test_history_endpoint_serves_the_callers_hands(db_app):
    app, pool = db_app
    _, players = await _play_hand(pool, 0, PARTNER_BY_CALLED_ACE)
    pid = players[2]
    await pool.execute(
        "INSERT INTO account (player_id, username, email, password_hash, "
        "email_verified_at, time_created, last_updated) "
        "VALUES ($1, $2, $3, 'x', now(), now(), now())",
        pid,
        f"hist_{pid.hex[:8]}",
        f"{pid.hex[:8]}@example.com",
    )
    token = await create_session(pool, pid)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get(
            "/api/account/hands", headers={"Authorization": f"Bearer {token}"}
        )
        bad = await client.get(
            "/api/account/hands",
            params={"before": "%%%"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["hands"]) == 1 and body["next_cursor"] is None
    assert body["hands"][0]["opponents"] == {"humans": 4, "ai": 0}
    assert (bad.status_code, bad.json()["detail"]) == (400, "invalid_cursor")
