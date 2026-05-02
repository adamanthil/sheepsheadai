from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from server.api.schemas import AnalyzeSimulateRequest, AnalyzeSimulateResponse

router = APIRouter()


@router.post("/api/analyze/simulate")
def analyze_simulate(req: AnalyzeSimulateRequest) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""
    try:
        from server.services.analyze import simulate_game

        return simulate_game(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("analyze_simulate failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
