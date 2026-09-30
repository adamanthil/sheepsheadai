"""Optional accounts: register, sign in and out, verify the email address.

Guests need none of this. Registering upgrades the caller's current guest
identity in place; signing in hands back a session for the account's
player, which the client adopts in place of whatever guest token it held.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import (
    PlayerIdentity,
    bearer_token,
    current_player,
    forget_player,
    forget_token,
    optional_player,
)
from server.api.ratelimit import (
    AUTH_EMAIL,
    AUTH_LOGIN,
    AUTH_LOOKUP,
    limiter,
    remote_address,
)
from server.api.schemas import (
    AccountMeResponse,
    AccountPublic,
    AccountSessionResponse,
    ChangePasswordRequest,
    EmailTokenRequest,
    ForgotPasswordRequest,
    LoginRequest,
    OkResponse,
    RegisterRequest,
    ResetPasswordRequest,
    UsernameAvailability,
    validate_username,
)
from server.config import get_settings
from server.realtime.broadcast import broadcast_table_update
from server.runtime.manager import tables
from server.runtime.tasks import spawn
from server.services.email import send_email
from server.services.passwords import hash_password, needs_rehash, verify_password
from server.services.persistence import accounts as accounts_db
from server.services.persistence import players as players_db
from server.services.persistence import sessions as sessions_db
from server.services.persistence.accounts import Account
from server.services.persistence.pool import get_db_pool

router = APIRouter()


def _public(account: Account) -> AccountPublic:
    return AccountPublic(
        username=account.username,
        email=account.email,
        email_verified=account.email_verified,
    )


def _client_address(request: Request) -> Optional[str]:
    return remote_address(request.client.host if request.client else None)


async def send_verification(account: Account) -> None:
    """Mint a verify token and email the link from a background task."""
    token = await accounts_db.create_email_token(
        get_db_pool(), account.player_id, "verify"
    )
    # The token rides in the fragment, which browsers never send to a
    # server, so it stays out of proxy and access logs.
    link = f"{get_settings().public_base_url}/account/verify#token={token}"
    spawn(
        send_email(
            account.email,
            "Confirm your Sheepshead account",
            f"Hi {account.username},\n\nConfirm your email address to unlock "
            f"your stats and a place on the leaderboard:\n\n{link}\n\n"
            "The link expires in 24 hours. If you didn't create this "
            "account, ignore this email.",
            f"<p>Hi {account.username},</p><p>Confirm your email address to "
            f"unlock your stats and a place on the leaderboard:</p>"
            f'<p><a href="{link}">Confirm my email</a></p>'
            "<p>The link expires in 24 hours. If you didn't create this "
            "account, ignore this email.</p>",
        ),
        name="email:verify",
    )


async def show_account_badge(account: Account) -> None:
    """Badge the newly verified player wherever they are seated now, so it
    appears without a rejoin."""
    for table in list(tables.tables.values()):
        conns = [
            c for c in table.clients.values() if c.player_id == str(account.player_id)
        ]
        if not conns:
            continue
        for conn in conns:
            conn.account_username = account.username
        await broadcast_table_update(table)


@router.get("/api/account/username-available", response_model=UsernameAvailability)
@limiter.limit(AUTH_LOOKUP)
async def username_available(request: Request, u: str):
    try:
        username = validate_username(u)
    except ValueError:
        return {"available": False}
    return {"available": await accounts_db.username_available(get_db_pool(), username)}


@router.post(
    "/api/account/register", response_model=AccountSessionResponse, status_code=201
)
@limiter.limit(AUTH_EMAIL)
async def register(request: Request, req: RegisterRequest):
    pool = get_db_pool()
    ip = _client_address(request)
    identity = await optional_player(request)
    # Upgrade the caller's guest identity in place, so the hands it has
    # already played count toward the account. A caller with no identity
    # yet gets one, exactly as a first table join would.
    player_id = identity.id if identity is not None else uuid.uuid4()
    await players_db.ensure_player(pool, player_id, ip)
    password_hash = await hash_password(req.password)
    try:
        account = await accounts_db.create_account(
            pool, player_id, req.username, req.email, password_hash
        )
    except accounts_db.AlreadyRegistered as e:
        raise HTTPException(status_code=409, detail="already_registered") from e
    except accounts_db.UsernameTaken as e:
        raise HTTPException(status_code=409, detail="username_taken") from e
    except accounts_db.EmailTaken as e:
        raise HTTPException(status_code=409, detail="email_taken") from e
    session_token = (
        await sessions_db.create_session(pool, player_id) if identity is None else None
    )
    await send_verification(account)
    player = await players_db.get_player(pool, player_id)
    return {
        "player_id": str(player_id),
        "name": player["name"] if player else None,
        "session_token": session_token,
        "account": _public(account),
    }


@router.post("/api/account/login", response_model=AccountSessionResponse)
@limiter.limit(AUTH_LOGIN)
async def login(request: Request, req: LoginRequest):
    pool = get_db_pool()
    account = await accounts_db.get_account_by_login(pool, req.login.strip())
    # verify_password burns a full hash check even for an unknown login, so
    # "no such account" and "wrong password" look the same from outside.
    ok = await verify_password(
        account.password_hash if account is not None else None, req.password
    )
    if account is None or not ok:
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if needs_rehash(account.password_hash):
        await accounts_db.set_password(
            pool, account.player_id, await hash_password(req.password)
        )
    await accounts_db.record_login(pool, account.player_id)
    ip = _client_address(request)
    if ip is not None:
        await players_db.touch_player_ip(pool, account.player_id, ip)
    session_token = await sessions_db.create_session(pool, account.player_id)
    player = await players_db.get_player(pool, account.player_id)
    return {
        "player_id": str(account.player_id),
        "name": player["name"] if player else None,
        "session_token": session_token,
        "account": _public(account),
    }


@router.post("/api/account/logout", response_model=OkResponse)
async def logout(request: Request, _: PlayerIdentity = Depends(current_player)):
    token = bearer_token(request)
    assert token is not None  # current_player rejected a missing token
    await sessions_db.delete_session(get_db_pool(), token)
    forget_token(token)
    return {"ok": True}


@router.get("/api/account/me", response_model=AccountMeResponse)
async def me(identity: PlayerIdentity = Depends(current_player)):
    pool = get_db_pool()
    player = await players_db.get_player(pool, identity.id)
    account = await accounts_db.get_account(pool, identity.id)
    return {
        "player_id": str(identity.id),
        "name": player["name"] if player else None,
        "account": _public(account) if account is not None else None,
    }


@router.post("/api/account/verify", response_model=AccountPublic)
async def verify_email(req: EmailTokenRequest):
    # No bearer needed: the link may be opened on a device that isn't
    # signed in. Holding the emailed token is the proof.
    pool = get_db_pool()
    player_id = await accounts_db.consume_email_token(pool, req.token, "verify")
    account = (
        await accounts_db.mark_verified(pool, player_id)
        if player_id is not None
        else None
    )
    if account is None:
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")
    await show_account_badge(account)
    return _public(account)


@router.post("/api/account/resend-verification", response_model=OkResponse)
@limiter.limit(AUTH_EMAIL)
async def resend_verification(
    request: Request, identity: PlayerIdentity = Depends(current_player)
):
    account = await accounts_db.get_account(get_db_pool(), identity.id)
    if account is None:
        raise HTTPException(status_code=404, detail="no_account")
    if account.email_verified:
        raise HTTPException(status_code=400, detail="already_verified")
    await send_verification(account)
    return {"ok": True}


async def _send_reset(email: str) -> None:
    """Email a reset link if ``email`` has an account; otherwise nothing."""
    pool = get_db_pool()
    account = await accounts_db.get_account_by_login(pool, email)
    if account is None:
        return
    token = await accounts_db.create_email_token(pool, account.player_id, "reset")
    link = f"{get_settings().public_base_url}/account/reset#token={token}"
    await send_email(
        account.email,
        "Reset your Sheepshead password",
        f"Hi {account.username},\n\nChoose a new password here:\n\n{link}\n\n"
        "The link expires in 1 hour and signs you out everywhere else. If "
        "you didn't ask for this, ignore this email.",
        f"<p>Hi {account.username},</p>"
        f'<p><a href="{link}">Choose a new password</a></p>'
        "<p>The link expires in 1 hour and signs you out everywhere else. "
        "If you didn't ask for this, ignore this email.</p>",
    )


@router.post("/api/account/forgot", response_model=OkResponse, status_code=202)
@limiter.limit(AUTH_EMAIL)
async def forgot_password(request: Request, req: ForgotPasswordRequest):
    # The lookup runs in the background too, so neither the answer nor the
    # response time says whether the address has an account.
    spawn(_send_reset(req.email), name="email:reset")
    return {"ok": True}


@router.post("/api/account/reset", response_model=AccountSessionResponse)
@limiter.limit(AUTH_LOGIN)
async def reset_password(request: Request, req: ResetPasswordRequest):
    pool = get_db_pool()
    player_id = await accounts_db.consume_email_token(pool, req.token, "reset")
    # Receiving the link proves the address, which also settles an
    # unverified account registered with someone else's email. (None: the
    # token outlived its account, dropped unverified by the purge.)
    account = (
        await accounts_db.mark_verified(pool, player_id)
        if player_id is not None
        else None
    )
    if player_id is None or account is None:
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")
    await accounts_db.set_password(pool, player_id, await hash_password(req.password))
    # Whoever knew the old password is signed out everywhere.
    await sessions_db.delete_player_sessions(pool, player_id)
    forget_player(player_id)
    await show_account_badge(account)
    session_token = await sessions_db.create_session(pool, player_id)
    player = await players_db.get_player(pool, player_id)
    return {
        "player_id": str(player_id),
        "name": player["name"] if player else None,
        "session_token": session_token,
        "account": _public(account),
    }


@router.post("/api/account/password", response_model=OkResponse)
@limiter.limit(AUTH_LOGIN)
async def change_password(
    request: Request,
    req: ChangePasswordRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    pool = get_db_pool()
    account = await accounts_db.get_account(pool, identity.id)
    if account is None:
        raise HTTPException(status_code=403, detail="account_required")
    if not await verify_password(account.password_hash, req.current_password):
        raise HTTPException(status_code=403, detail="invalid_credentials")
    await accounts_db.set_password(
        pool, identity.id, await hash_password(req.new_password)
    )
    # Every other device is signed out; this one stays signed in. The
    # current token re-resolves from the database on its next request.
    token = bearer_token(request)
    assert token is not None  # current_player rejected a missing token
    await sessions_db.delete_player_sessions_except(pool, identity.id, token)
    forget_player(identity.id)
    return {"ok": True}
