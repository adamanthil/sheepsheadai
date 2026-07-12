"""Bearer-token authentication for the anonymous-identity model.

REST callers present ``Authorization: Bearer <token>``; the websocket path
passes the same token via subprotocol. Tokens resolve to a player_id through
the ``session`` table, fronted by a small in-process cache so steady-state
requests don't pay a DB round-trip. Single-process deployment makes the
process-local cache exact enough (invalidation lag is bounded by CACHE_TTL).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request

from server.services.persistence import sessions as sessions_db
from server.services.persistence.pool import get_db_pool
from server.services.persistence.sessions import hash_token

_CACHE_TTL = 60.0
_CACHE_MAX = 4096
# token_hash -> (player_id, cached_at monotonic)
_cache: dict[str, tuple[UUID, float]] = {}


@dataclass(frozen=True)
class PlayerIdentity:
    id: UUID


def clear_cache() -> None:
    _cache.clear()


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization")
    if not header or not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


async def resolve_player(token: str) -> Optional[PlayerIdentity]:
    """Resolve a raw token to an identity, using the cache. Shared with WS."""
    key = hash_token(token)
    now = time.monotonic()
    hit = _cache.get(key)
    if hit is not None and now - hit[1] < _CACHE_TTL:
        return PlayerIdentity(id=hit[0])
    player_id = await sessions_db.resolve_token(get_db_pool(), token)
    if player_id is None:
        _cache.pop(key, None)
        return None
    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = (player_id, now)
    return PlayerIdentity(id=player_id)


async def optional_player(request: Request) -> Optional[PlayerIdentity]:
    token = _bearer_token(request)
    if not token:
        return None
    return await resolve_player(token)


async def current_player(request: Request) -> PlayerIdentity:
    token = _bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="missing_token")
    identity = await resolve_player(token)
    if identity is None:
        raise HTTPException(status_code=401, detail="invalid_token")
    return identity
