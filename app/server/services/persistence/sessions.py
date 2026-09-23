"""Anonymous session tokens (public-internet hardening).

The server mints an opaque bearer token whenever it creates a player
identity. Only the SHA-256 hash is stored; presenting the token is the sole
proof of identity. TTL is sliding: resolving a token refreshes it, throttled
to once an hour so routine traffic doesn't write on every request.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
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


# How long a new player row is spared by the purge. /join writes the
# player row and then its session as two statements, so a purge landing
# between them must not take the row out from under the session insert.
PURGE_GRACE = "1 hour"
PURGE_INTERVAL_SECONDS = 3600.0


async def purge_expired_identities(pool: asyncpg.Pool) -> tuple[int, int]:
    """Delete expired sessions, then players left with no session and no
    recorded hands (a player with hands keeps their row for the history).
    Returns (sessions deleted, players deleted)."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            sessions = await conn.execute(
                "DELETE FROM session WHERE expires_at <= now()"
            )
            players = await conn.execute(
                f"""
                DELETE FROM player p
                WHERE p.time_created < now() - interval '{PURGE_GRACE}'
                  AND NOT EXISTS (
                      SELECT 1 FROM session s WHERE s.player_id = p.player_id)
                  AND NOT EXISTS (
                      SELECT 1 FROM game_player gp WHERE gp.player_id = p.player_id)
                """
            )
    # asyncpg returns the command tag, e.g. "DELETE 3".
    return int(sessions.split()[-1]), int(players.split()[-1])


async def run_identity_purge(pool: asyncpg.Pool) -> None:
    """Purge expired identities hourly, forever; started by the app lifespan."""
    while True:
        try:
            sessions, players = await purge_expired_identities(pool)
            if sessions or players:
                logging.info(
                    "purged %d expired sessions and %d orphaned players",
                    sessions,
                    players,
                )
        except Exception:
            logging.exception("identity purge failed")
        await asyncio.sleep(PURGE_INTERVAL_SECONDS)
