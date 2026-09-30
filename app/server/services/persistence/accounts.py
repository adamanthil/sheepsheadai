"""Optional accounts layered on anonymous players.

An account row hangs off an existing ``player`` row: registering upgrades
the caller's own guest identity in place, so hands already played count
toward it, and signing in on another device simply mints a new session for
that same player. The username is unique (case-insensitively) and separate
from ``player.name``, the non-unique in-game display name.

Emailed tokens (verification, password reset) are single-use and stored,
like session tokens, only as a SHA-256 hash.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID

import asyncpg

from server.services.persistence.sessions import hash_token

TokenPurpose = Literal["verify", "reset"]
VERIFY_TTL = "24 hours"
RESET_TTL = "1 hour"


class AlreadyRegistered(Exception):
    pass


class UsernameTaken(Exception):
    pass


class EmailTaken(Exception):
    pass


@dataclass(frozen=True)
class Account:
    player_id: UUID
    username: str
    email: str
    password_hash: str
    email_verified: bool


_COLUMNS = (
    "player_id, username, email, password_hash, "
    "email_verified_at IS NOT NULL AS email_verified"
)


def _account(row: Optional[asyncpg.Record]) -> Optional[Account]:
    if row is None:
        return None
    return Account(
        player_id=row["player_id"],
        username=row["username"],
        email=row["email"],
        password_hash=row["password_hash"],
        email_verified=row["email_verified"],
    )


async def create_account(
    pool: asyncpg.Pool,
    player_id: UUID,
    username: str,
    email: str,
    password_hash: str,
) -> Account:
    """Attach an account to ``player_id``; raises AlreadyRegistered,
    UsernameTaken, or EmailTaken."""
    try:
        row = await pool.fetchrow(
            f"""
            INSERT INTO account (player_id, username, email, password_hash,
                                 time_created, last_updated)
            VALUES ($1, $2, $3, $4, now(), now())
            RETURNING {_COLUMNS}
            """,
            player_id,
            username,
            email,
            password_hash,
        )
    except asyncpg.UniqueViolationError as e:
        if e.constraint_name == "account_pkey":
            raise AlreadyRegistered from e
        if e.constraint_name == "account_username_idx":
            raise UsernameTaken from e
        if e.constraint_name == "account_email_idx":
            raise EmailTaken from e
        raise
    account = _account(row)
    assert account is not None
    return account


async def get_account(pool: asyncpg.Pool, player_id: UUID) -> Optional[Account]:
    row = await pool.fetchrow(
        f"SELECT {_COLUMNS} FROM account WHERE player_id = $1", player_id
    )
    return _account(row)


async def get_account_by_login(pool: asyncpg.Pool, login: str) -> Optional[Account]:
    """Look an account up by email (anything containing "@") or username."""
    if "@" in login:
        row = await pool.fetchrow(
            f"SELECT {_COLUMNS} FROM account WHERE email = $1", login.lower()
        )
    else:
        row = await pool.fetchrow(
            f"SELECT {_COLUMNS} FROM account WHERE lower(username) = lower($1)",
            login,
        )
    return _account(row)


async def username_available(pool: asyncpg.Pool, username: str) -> bool:
    taken = await pool.fetchval(
        "SELECT EXISTS (SELECT 1 FROM account WHERE lower(username) = lower($1))",
        username,
    )
    return not taken


async def mark_verified(pool: asyncpg.Pool, player_id: UUID) -> Optional[Account]:
    row = await pool.fetchrow(
        f"""
        UPDATE account
        SET email_verified_at = coalesce(email_verified_at, now()),
            last_updated = now()
        WHERE player_id = $1
        RETURNING {_COLUMNS}
        """,
        player_id,
    )
    return _account(row)


async def set_password(pool: asyncpg.Pool, player_id: UUID, password_hash: str) -> None:
    await pool.execute(
        """
        UPDATE account SET password_hash = $2, last_updated = now()
        WHERE player_id = $1
        """,
        player_id,
        password_hash,
    )


async def record_login(pool: asyncpg.Pool, player_id: UUID) -> None:
    await pool.execute(
        "UPDATE account SET last_login = now() WHERE player_id = $1", player_id
    )


async def create_email_token(
    pool: asyncpg.Pool, player_id: UUID, purpose: TokenPurpose
) -> str:
    """Mint a single-use emailed token; returns the raw token (never stored)."""
    ttl = VERIFY_TTL if purpose == "verify" else RESET_TTL
    token = secrets.token_urlsafe(32)
    await pool.execute(
        f"""
        INSERT INTO email_token (token_hash, player_id, purpose, expires_at)
        VALUES ($1, $2, $3, now() + interval '{ttl}')
        """,
        hash_token(token),
        player_id,
        purpose,
    )
    return token


async def consume_email_token(
    pool: asyncpg.Pool, token: str, purpose: TokenPurpose
) -> Optional[UUID]:
    """Spend a live token of ``purpose``; returns its player, else None."""
    return await pool.fetchval(
        """
        UPDATE email_token SET used_at = now()
        WHERE token_hash = $1 AND purpose = $2
          AND used_at IS NULL AND expires_at > now()
        RETURNING player_id
        """,
        hash_token(token),
        purpose,
    )
