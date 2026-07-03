from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from server.api.ratelimit import HOST_ACTIONS, limiter
from server.api.schemas import UpdatePlayerRequest
from server.services.persistence import players as players_db
from server.services.persistence.pool import get_db_pool

router = APIRouter()


@router.get("/api/players/{player_id}")
async def get_player(player_id: UUID):
    player = await players_db.get_player(get_db_pool(), player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player


@router.patch("/api/players/{player_id}")
@limiter.limit(HOST_ACTIONS)
async def update_player(request: Request, player_id: UUID, req: UpdatePlayerRequest):
    player = await players_db.set_player_name(get_db_pool(), player_id, req.name)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player
