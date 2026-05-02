from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from server.api.schemas import UpdatePlayerRequest
from server.services.persistence import players as players_db

router = APIRouter()


@router.get("/api/players/{player_id}")
async def get_player(player_id: UUID, request: Request):
    pool = request.app.state.db_pool
    player = await players_db.get_player(pool, player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player


@router.patch("/api/players/{player_id}")
async def update_player(player_id: UUID, req: UpdatePlayerRequest, request: Request):
    pool = request.app.state.db_pool
    player = await players_db.set_player_name(pool, player_id, req.name)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player
