from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player
from server.api.ratelimit import HOST_ACTIONS, limiter
from server.api.schemas import PlayerPublic, UpdatePlayerRequest
from server.services.persistence import players as players_db
from server.services.persistence.pool import get_db_pool

router = APIRouter()


def _require_owner(player_id: UUID, identity: PlayerIdentity) -> None:
    if player_id != identity.id:
        raise HTTPException(status_code=403, detail="not_your_player")


@router.get("/api/players/{player_id}", response_model=PlayerPublic)
async def get_player(
    player_id: UUID, identity: PlayerIdentity = Depends(current_player)
):
    _require_owner(player_id, identity)
    player = await players_db.get_player(get_db_pool(), player_id)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player


@router.patch("/api/players/{player_id}", response_model=PlayerPublic)
@limiter.limit(HOST_ACTIONS)
async def update_player(
    request: Request,
    player_id: UUID,
    req: UpdatePlayerRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    _require_owner(player_id, identity)
    player = await players_db.set_player_name(get_db_pool(), player_id, req.name)
    if player is None:
        raise HTTPException(status_code=404, detail="player_not_found")
    return player
