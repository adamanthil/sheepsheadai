"""asyncpg connection pool wired to the FastAPI lifespan.

The server fails fast at startup if `DATABASE_URL` is missing — persistence
is a hard requirement (Phase 3 §3.5).
"""

from __future__ import annotations

import logging
from typing import Optional

import asyncpg
from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Module-level state set at startup so background tasks (ai_loop, lifecycle)
# can access the pool and AI identity without holding a request/app reference.
_pool: Optional[asyncpg.Pool] = None
_ai_player_id: Optional[int] = None


def set_db_state(pool: asyncpg.Pool, ai_player_id: int) -> None:
    global _pool, _ai_player_id
    _pool = pool
    _ai_player_id = ai_player_id


def get_db_pool() -> Optional[asyncpg.Pool]:
    return _pool


def get_ai_player_id() -> Optional[int]:
    return _ai_player_id


async def open_pool(app: FastAPI, database_url: str) -> asyncpg.Pool:
    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")
    pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=1,
        max_size=10,
    )
    app.state.db_pool = pool
    logger.info("Postgres pool initialised")
    return pool


async def close_pool(app: FastAPI) -> None:
    global _pool, _ai_player_id
    pool: asyncpg.Pool | None = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        app.state.db_pool = None
        _pool = None
        _ai_player_id = None
        logger.info("Postgres pool closed")


def get_pool(app: FastAPI) -> asyncpg.Pool:
    pool: asyncpg.Pool | None = getattr(app.state, "db_pool", None)
    if pool is None:
        raise RuntimeError("Postgres pool is not initialised")
    return pool
