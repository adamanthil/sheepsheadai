from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import actions as actions_router
from server.api import analyze as analyze_router
from server.api import games as games_router
from server.api import health as health_router
from server.api import players as players_router
from server.api import tables as tables_router
from server.config import get_settings
from server.realtime import websocket as websocket_router
from server.services.ai_loader import load_agent
from server.services.persistence.pool import close_pool, open_pool, set_db_state


# Dev-only CORS: local Next.js dev servers. Anchored so hostile origins that
# merely *contain* a local-looking suffix (e.g. http://evil.com:3000) never match.
DEV_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data)


async def _upsert_ai_model(conn, label: str) -> int:
    """Return the ai_model_id for ``label``, inserting if novel."""
    row = await conn.fetchrow(
        "INSERT INTO ai_model (label, time_created) VALUES ($1, now()) "
        "ON CONFLICT (label) DO NOTHING RETURNING ai_model_id",
        label,
    )
    if row is not None:
        return row["ai_model_id"]
    return await conn.fetchval(
        "SELECT ai_model_id FROM ai_model WHERE label = $1", label
    )


async def _upsert_ai_player(conn, ai_model_id: int) -> int:
    """Return the ai_player_id for (model, deterministic=true), inserting if novel."""
    row = await conn.fetchrow(
        "INSERT INTO ai_player (ai_model_id, is_deterministic) VALUES ($1, true) "
        "ON CONFLICT (ai_model_id, is_deterministic) DO NOTHING "
        "RETURNING ai_player_id",
        ai_model_id,
    )
    if row is not None:
        return row["ai_player_id"]
    return await conn.fetchval(
        "SELECT ai_player_id FROM ai_player "
        "WHERE ai_model_id = $1 AND is_deterministic = true",
        ai_model_id,
    )


def create_app() -> FastAPI:
    settings = get_settings()

    # Configure logging before anything else.
    if settings.log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    # Validate required config at startup so misconfiguration is caught immediately.
    if not settings.sheepshead_model_path:
        raise RuntimeError("SHEEPSHEAD_MODEL_PATH must be set")
    if not os.path.exists(settings.sheepshead_model_path):
        raise FileNotFoundError(
            f"SHEEPSHEAD_MODEL_PATH points to a missing file: {settings.sheepshead_model_path}"
        )
    if not settings.sheepshead_model_label:
        raise RuntimeError("SHEEPSHEAD_MODEL_LABEL must be set")
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be set")
    load_agent(settings.sheepshead_model_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = await open_pool(settings.database_url)

        async with pool.acquire() as conn:
            ai_model_id = await _upsert_ai_model(conn, settings.sheepshead_model_label)
            ai_player_id = await _upsert_ai_player(conn, ai_model_id)
        set_db_state(pool, ai_player_id)
        logging.getLogger(__name__).info(
            "AI model '%s' id=%s player_id=%s",
            settings.sheepshead_model_label,
            ai_model_id,
            ai_player_id,
        )

        try:
            yield
        finally:
            await close_pool()

    app = FastAPI(title="Sheepshead Realtime API", lifespan=lifespan)

    if settings.env == "production":
        origins = [
            o.strip() for o in settings.sheepshead_cors_origins.split(",") if o.strip()
        ]
        if not origins:
            raise RuntimeError(
                "SHEEPSHEAD_CORS_ORIGINS must be set in production (comma-separated list of allowed origins)"
            )
        cors_config = {
            "allow_origins": origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    else:
        cors_config = {
            "allow_origin_regex": DEV_CORS_ORIGIN_REGEX,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    app.add_middleware(CORSMiddleware, **cors_config)

    app.include_router(health_router.router)
    app.include_router(tables_router.router)
    app.include_router(games_router.router)
    app.include_router(actions_router.router)
    app.include_router(analyze_router.router)
    app.include_router(players_router.router)
    app.include_router(websocket_router.router)

    return app
