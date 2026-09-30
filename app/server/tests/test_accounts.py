"""Optional accounts end to end: register (upgrading a guest in place),
verify, sign in and out.

Opt-in like test_api_flow: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import httpx
import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


@pytest.fixture
def outbox(monkeypatch):
    """Capture account emails instead of logging them."""
    import server.api.account as account_module

    sent: list[dict] = []

    async def fake_send(to, subject, text, html):
        sent.append({"to": to, "subject": subject, "text": text})

    monkeypatch.setattr(account_module, "send_email", fake_send)
    return sent


@pytest.fixture
async def client(db_app):
    app, _ = db_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _fresh() -> tuple[str, str]:
    """A username and email no earlier run of this suite has used."""
    tag = uuid.uuid4().hex[:10]
    return f"user_{tag}", f"user.{tag}@example.com"


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _emailed_token(outbox: list[dict]) -> str:
    await asyncio.sleep(0)  # let the spawned send run
    return outbox[-1]["text"].split("#token=")[1].split()[0]


async def _guest(client) -> tuple[str, str]:
    """Join a table as a guest; returns (player_id, session token)."""
    created = await client.post("/api/tables", json={"name": "acct"})
    joined = await client.post(
        f"/api/tables/{created.json()['id']}/join",
        json={"display_name": "Guesty", "host_key": created.json()["host_key"]},
    )
    j = joined.json()
    await client.patch(
        f"/api/players/{j['player_id']}",
        json={"name": "Guesty"},
        headers=_bearer(j["session_token"]),
    )
    return j["player_id"], j["session_token"]


async def test_register_upgrades_the_guest_in_place_then_verifies(client, outbox):
    player_id, token = await _guest(client)
    username, email = _fresh()

    r = await client.post(
        "/api/account/register",
        json={"username": username, "email": email.upper(), "password": "hunter22!"},
        headers=_bearer(token),
    )

    assert r.status_code == 201, r.text
    body = r.json()
    # Same player, same session: the guest's history carries over.
    assert body["player_id"] == player_id
    assert body["session_token"] is None
    assert body["name"] == "Guesty"
    assert body["account"] == {
        "username": username,
        "email": email,
        "email_verified": False,
    }

    verify_token = await _emailed_token(outbox)
    assert outbox[-1]["to"] == email
    r = await client.post("/api/account/verify", json={"token": verify_token})
    assert r.status_code == 200
    assert r.json()["email_verified"] is True
    # Single use.
    r = await client.post("/api/account/verify", json={"token": verify_token})
    assert r.status_code == 400

    me = (await client.get("/api/account/me", headers=_bearer(token))).json()
    assert me["player_id"] == player_id
    assert me["account"]["email_verified"] is True


async def test_register_without_identity_mints_one(client, outbox):
    username, email = _fresh()
    r = await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
    )
    assert r.status_code == 201
    token = r.json()["session_token"]
    assert token
    me = (await client.get("/api/account/me", headers=_bearer(token))).json()
    assert me["account"]["username"] == username


async def test_register_conflicts(client, outbox):
    player_id, token = await _guest(client)
    username, email = _fresh()
    ok = await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(token),
    )
    assert ok.status_code == 201

    again = await client.post(
        "/api/account/register",
        json={"username": _fresh()[0], "email": _fresh()[1], "password": "hunter22!"},
        headers=_bearer(token),
    )
    assert again.json()["detail"] == "already_registered"

    same_name = await client.post(
        "/api/account/register",
        json={"username": username.upper(), "email": _fresh()[1], "password": "x" * 8},
    )
    assert (same_name.status_code, same_name.json()["detail"]) == (
        409,
        "username_taken",
    )

    same_email = await client.post(
        "/api/account/register",
        json={"username": _fresh()[0], "email": email, "password": "x" * 8},
    )
    assert (same_email.status_code, same_email.json()["detail"]) == (
        409,
        "email_taken",
    )

    avail = await client.get(
        "/api/account/username-available", params={"u": username.lower()}
    )
    assert avail.json() == {"available": False}
    avail = await client.get(
        "/api/account/username-available", params={"u": _fresh()[0]}
    )
    assert avail.json() == {"available": True}


async def test_login_by_email_or_username_on_another_device(client, outbox, db_app):
    _, pool = db_app
    player_id, token = await _guest(client)
    username, email = _fresh()
    await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(token),
    )

    for login in (email.upper(), username.lower()):
        r = await client.post(
            "/api/account/login", json={"login": login, "password": "hunter22!"}
        )
        assert r.status_code == 200, login
        body = r.json()
        assert body["player_id"] == player_id
        assert body["name"] == "Guesty"
        assert body["session_token"] not in (None, token)

    # httpx's ASGITransport reports the client as 127.0.0.1.
    last_ip = await pool.fetchval(
        "SELECT host(last_ip) FROM player WHERE player_id = $1", uuid.UUID(player_id)
    )
    assert last_ip == "127.0.0.1"

    wrong = await client.post(
        "/api/account/login", json={"login": username, "password": "wrong-pass"}
    )
    unknown = await client.post(
        "/api/account/login",
        json={"login": _fresh()[1], "password": "hunter22!"},
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "invalid_credentials"}


async def test_logout_ends_only_that_session(client, outbox):
    _, guest_token = await _guest(client)
    username, email = _fresh()
    await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(guest_token),
    )
    login = await client.post(
        "/api/account/login", json={"login": username, "password": "hunter22!"}
    )
    other_device = login.json()["session_token"]
    assert (
        await client.get("/api/account/me", headers=_bearer(other_device))
    ).status_code == 200

    r = await client.post("/api/account/logout", headers=_bearer(other_device))
    assert r.status_code == 200

    assert (
        await client.get("/api/account/me", headers=_bearer(other_device))
    ).status_code == 401
    assert (
        await client.get("/api/account/me", headers=_bearer(guest_token))
    ).status_code == 200


async def test_resend_verification(client, outbox):
    _, token = await _guest(client)
    none = await client.post("/api/account/resend-verification", headers=_bearer(token))
    assert none.json()["detail"] == "no_account"

    username, email = _fresh()
    await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(token),
    )
    first = await _emailed_token(outbox)
    r = await client.post("/api/account/resend-verification", headers=_bearer(token))
    assert r.status_code == 200
    second = await _emailed_token(outbox)
    assert second != first

    await client.post("/api/account/verify", json={"token": second})
    r = await client.post("/api/account/resend-verification", headers=_bearer(token))
    assert r.json()["detail"] == "already_verified"


async def test_password_reset_signs_out_everywhere(client, outbox):
    _, guest_token = await _guest(client)
    username, email = _fresh()
    await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(guest_token),
    )
    login = await client.post(
        "/api/account/login", json={"login": email, "password": "hunter22!"}
    )
    other_device = login.json()["session_token"]
    # Prime the auth cache so the reset has to evict, not just delete.
    assert (
        await client.get("/api/account/me", headers=_bearer(other_device))
    ).status_code == 200

    # Unknown addresses get the same answer and no email.
    sent_before = len(outbox)
    r = await client.post("/api/account/forgot", json={"email": _fresh()[1]})
    await asyncio.sleep(0)
    assert r.status_code == 202 and len(outbox) == sent_before

    r = await client.post("/api/account/forgot", json={"email": email.upper()})
    assert r.status_code == 202
    for _ in range(5):  # the lookup itself runs in the background
        await asyncio.sleep(0.02)
        if len(outbox) > sent_before:
            break
    reset_token = await _emailed_token(outbox)

    r = await client.post(
        "/api/account/reset", json={"token": reset_token, "password": "new-pass-99"}
    )
    assert r.status_code == 200, r.text
    fresh_token = r.json()["session_token"]
    # The reset proved the address.
    assert r.json()["account"]["email_verified"] is True

    for old in (guest_token, other_device):
        assert (
            await client.get("/api/account/me", headers=_bearer(old))
        ).status_code == 401
    assert (
        await client.get("/api/account/me", headers=_bearer(fresh_token))
    ).status_code == 200

    old_pw = await client.post(
        "/api/account/login", json={"login": email, "password": "hunter22!"}
    )
    new_pw = await client.post(
        "/api/account/login", json={"login": email, "password": "new-pass-99"}
    )
    assert (old_pw.status_code, new_pw.status_code) == (401, 200)

    again = await client.post(
        "/api/account/reset", json={"token": reset_token, "password": "another-1"}
    )
    assert again.status_code == 400


async def test_badge_shows_only_once_verified_and_without_rejoin(client, outbox):
    from server.runtime.manager import tables

    created = await client.post("/api/tables", json={"name": "badge"})
    table_id = created.json()["id"]
    joined = (
        await client.post(
            f"/api/tables/{table_id}/join",
            json={"display_name": "Bea", "host_key": created.json()["host_key"]},
        )
    ).json()
    token = joined["session_token"]
    seat = next(
        s
        for s, occ in joined["table"]["seatOccupants"].items()
        if occ == joined["client_id"]
    )
    assert joined["table"]["seatAccount"][seat] is None

    username, email = _fresh()
    await client.post(
        "/api/account/register",
        json={"username": username, "email": email, "password": "hunter22!"},
        headers=_bearer(token),
    )
    # Unverified: no badge, even on a fresh join.
    again = await client.post(
        f"/api/tables/{table_id}/join",
        json={"display_name": "Bea"},
        headers=_bearer(token),
    )
    assert again.json()["table"]["seatAccount"][seat] is None

    await client.post(
        "/api/account/verify", json={"token": await _emailed_token(outbox)}
    )
    table = tables.get_table(table_id).to_public_dict()
    assert table["seatAccount"][int(seat)] == username
    # The seat keeps showing the in-game name, not the username.
    assert table["seats"][int(seat)] == "Bea"
    # An AI seat never carries a badge.
    assert all(
        table["seatAccount"][s] is None for s, ai in table["seatIsAI"].items() if ai
    )
