from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Request

from server.api.ratelimit import ANALYZE, ANALYZE_PICK, limiter
from server.api.schemas import (
    AnalyzeModelResponse,
    AnalyzePickRequest,
    AnalyzePickResponse,
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
)

router = APIRouter()

# Each simulation runs a full game of torch inference; bound concurrency so a
# burst (this is a sync endpoint, so each request occupies a threadpool
# worker) cannot oversubscribe the CPU that live games depend on.
_sim_slots = threading.BoundedSemaphore(2)


def _run_simulation(name, fn):
    """Shared guard for the inference-bearing endpoints: a concurrency slot,
    ValueError -> 400, anything else -> logged 500."""
    if not _sim_slots.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="analyze_busy")
    try:
        return fn()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("%s failed: %s", name, e)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        _sim_slots.release()


@router.get("/api/analyze/model")
@limiter.limit(ANALYZE)
def analyze_model(request: Request) -> AnalyzeModelResponse:
    """Describe the deployed model: architecture, capabilities, and its
    card-embedding geometry. Cached per checkpoint."""
    try:
        from server.services.model_info import get_model_info

        return get_model_info()
    except Exception as e:
        logging.exception("analyze_model failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/api/analyze/simulate")
@limiter.limit(ANALYZE)
def analyze_simulate(
    request: Request, req: AnalyzeSimulateRequest
) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""
    from server.services.analyze import simulate_game

    return _run_simulation("analyze_simulate", lambda: simulate_game(req))


@router.post("/api/analyze/pick")
@limiter.limit(ANALYZE_PICK)
def analyze_pick(request: Request, req: AnalyzePickRequest) -> AnalyzePickResponse:
    """Analyze the pre-play decisions (pick/pass, call, bury) for a chosen
    seat, hand, and blind."""
    from server.services.pick_analysis import analyze_pick as run_pick

    return _run_simulation("analyze_pick", lambda: run_pick(req))
