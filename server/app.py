from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import get_settings
from server.services.ai_loader import load_agent
from server.api import actions as actions_router
from server.api import analyze as analyze_router
from server.api import games as games_router
from server.api import players as players_router
from server.api import tables as tables_router
from server.realtime import websocket as websocket_router


def create_app() -> FastAPI:
    settings = get_settings()

    # Validate model at startup so misconfiguration is caught immediately.
    if not settings.sheepshead_model_path:
        raise RuntimeError("SHEEPSHEAD_MODEL_PATH must be set")
    if not os.path.exists(settings.sheepshead_model_path):
        raise FileNotFoundError(
            f"SHEEPSHEAD_MODEL_PATH points to a missing file: {settings.sheepshead_model_path}"
        )
    load_agent(settings.sheepshead_model_path)

    logging.basicConfig(level=logging.INFO)

    app = FastAPI(title="Sheepshead Realtime API")

    if settings.env == "production":
        cors_config = {
            "allow_origins": [o.strip() for o in settings.sheepshead_cors_origins.split(",") if o.strip()] or ["https://yourdomain.com"],
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
