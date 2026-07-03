"""Anonymous session tokens (public-internet hardening).

The server mints an opaque bearer token whenever it creates a player
identity. Only the SHA-256 hash is stored; presenting the token is the sole
proof of identity. TTL is sliding: resolving a token refreshes it, throttled
to once an hour so routine traffic doesn't write on every request.
"""

from __future__ import annotations

import hashlib
import secrets
from typing import Optional
from uuid import UUID

import asyncpg

SESSION_TTL = "30 days"
# Refresh last_seen/expires_at at most this often.
BUMP_AFTER = "1 hour"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def create_session(pool: asyncpg.Pool, player_id: UUID) -> str:
    """Mint a session token for ``player_id`` and return it (never stored raw)."""
    token = secrets.token_urlsafe(32)
    await pool.execute(
        f"""
        INSERT INTO session (player_id, token_hash, time_created, last_seen, expires_at)
        VALUES ($1, $2, now(), now(), now() + interval '{SESSION_TTL}')
        """,
        player_id,
        hash_token(token),
    )
    return token


async def resolve_token(pool: asyncpg.Pool, token: str) -> Optional[UUID]:
    """Return the player_id for a live token, sliding its expiry; else None."""
    row = await pool.fetchrow(
        f"""
        UPDATE session
        SET last_seen = now(),
            expires_at = now() + interval '{SESSION_TTL}'
        WHERE token_hash = $1
          AND expires_at > now()
          AND last_seen < now() - interval '{BUMP_AFTER}'
        RETURNING player_id
        """,
        hash_token(token),
    )
    if row is not None:
        return row["player_id"]
    # Common case: seen recently, no write needed.
    row = await pool.fetchrow(
        "SELECT player_id FROM session WHERE token_hash = $1 AND expires_at > now()",
        hash_token(token),
    )
    return row["player_id"] if row is not None else None
