from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter

from server.api.schemas import HealthResponse
from server.config import get_settings
from server.services.persistence.pool import get_db_pool

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Liveness/readiness probe for uptime checks and container healthchecks.

    Always returns 200 while the process is serving; ``db`` reports
    best-effort connectivity so a degraded database shows up in monitoring
    without taking the probe (and thus the container) down with it.
    """
    db_ok = False
    try:
        pool = get_db_pool()
        await asyncio.wait_for(pool.fetchval("SELECT 1"), timeout=1.0)
        db_ok = True
    except Exception:
        logger.warning("health check: database unreachable", exc_info=True)
    return {
        "status": "ok",
        "db": db_ok,
        "model": get_settings().sheepshead_model_label,
    }
