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
from server.api import players as players_router
from server.api import tables as tables_router
from server.config import get_settings
from server.realtime import websocket as websocket_router
from server.services.ai_loader import load_agent
from server.services.persistence.pool import close_pool, open_pool, set_db_state


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
        pool = await open_pool(app, settings.database_url)

        # Upsert ai_model and ai_player rows; cache IDs on app.state and module state.
        async with pool.acquire() as conn:
            ai_model_id = await conn.fetchval(
                """
                INSERT INTO ai_model (label, time_created)
                VALUES ($1, now())
                ON CONFLICT (label) DO UPDATE SET label = EXCLUDED.label
                RETURNING ai_model_id
                """,
                settings.sheepshead_model_label,
            )
            ai_player_id = await conn.fetchval(
                """
                INSERT INTO ai_player (ai_model_id, is_deterministic)
                VALUES ($1, true)
                ON CONFLICT (ai_model_id, is_deterministic)
                DO UPDATE SET ai_model_id = EXCLUDED.ai_model_id
                RETURNING ai_player_id
                """,
                ai_model_id,
            )
        app.state.ai_model_id = ai_model_id
        app.state.ai_player_id = ai_player_id
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
            await close_pool(app)

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
            "allow_origin_regex": r"http://[^/]+:3000",
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    app.add_middleware(CORSMiddleware, **cors_config)

    app.include_router(tables_router.router)
    app.include_router(games_router.router)
    app.include_router(actions_router.router)
    app.include_router(analyze_router.router)
    app.include_router(players_router.router)
    app.include_router(websocket_router.router)

    return app


app = create_app()
