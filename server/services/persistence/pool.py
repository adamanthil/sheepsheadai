"""asyncpg connection pool wired to the FastAPI lifespan.

The server fails fast at startup if ``DATABASE_URL`` is missing — persistence
is a hard requirement (Phase 3 §3.5). All accessors return non-optional
types and raise ``RuntimeError`` if called before the lifespan startup hook
has populated module state.
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

# Module-level state set at startup so background tasks (ai_loop, lifecycle)
# can access the pool and AI identity without holding a request/app reference.
_pool: Optional[asyncpg.Pool] = None
_ai_player_id: Optional[int] = None


def set_db_state(pool: asyncpg.Pool, ai_player_id: int) -> None:
    global _pool, _ai_player_id
    _pool = pool
    _ai_player_id = ai_player_id


def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Postgres pool is not initialised")
    return _pool


def get_ai_player_id() -> int:
    if _ai_player_id is None:
        raise RuntimeError("ai_player_id is not initialised")
    return _ai_player_id


async def open_pool(database_url: str) -> asyncpg.Pool:
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=10,
    )
    logger.info("Postgres pool initialised")
    return pool


async def close_pool() -> None:
    global _pool, _ai_player_id
    if _pool is not None:
        await _pool.close()
        _pool = None
        _ai_player_id = None
        logger.info("Postgres pool closed")
