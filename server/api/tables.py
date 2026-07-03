from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player, optional_player
from server.api.ratelimit import CREATE_JOIN, HOST_ACTIONS, limiter
from server.api.schemas import (
    CloseTableRequest,
    CreateTableRequest,
    JoinTableRequest,
    SeatRequest,
    UpdateTableRulesRequest,
)
from server.realtime.broadcast import broadcast_table_event, broadcast_table_update
from server.realtime.chat import add_chat_message, broadcast_chat_append
from server.runtime.lifecycle import (
    close_table,
    is_draining,
    schedule_autoclose_if_no_humans,
)
from server.runtime.seating import (
    _allocate_ai_occupant,
    _is_ai_occupant,
    _lowest_non_human_seat,
    _pick_join_ai_seat,
    _replace_ai_with_human_and_reserve,
    _reserved_ai_ids,
)
from server.runtime.tables import (
    ClientConn,
    Table,
    TableLimitError,
    prune_table_state,
    tables,
)
from server.services.persistence import players as players_db
from server.services.persistence import sessions as sessions_db
from server.services.persistence.pool import get_db_pool

router = APIRouter()


def require_client(
    table: Table, client_id: Optional[str], identity: PlayerIdentity
) -> ClientConn:
    """Return the table connection for ``client_id`` after verifying it
    belongs to the authenticated player. client_id in a request body is a
    routing hint, never a credential — the bearer token is."""
    conn = table.clients.get(client_id or "")
    if conn is None:
        raise HTTPException(status_code=400, detail="client_not_joined")
    if conn.player_id != str(identity.id):
        raise HTTPException(status_code=403, detail="client_mismatch")
    return conn


def require_host(
    table: Table, client_id: Optional[str], identity: PlayerIdentity
) -> None:
    """Raise the appropriate HTTPException unless the authenticated caller
    is the host connection of ``table``."""
    require_client(table, client_id, identity)
    if not table.host_client_id or client_id != table.host_client_id:
        raise HTTPException(status_code=403, detail="not_host")


@router.get("/api/tables")
def list_tables():
    return tables.list_tables()


@router.post("/api/tables")
@limiter.limit(CREATE_JOIN)
async def create_table(request: Request, req: CreateTableRequest):
    if is_draining():
        raise HTTPException(status_code=503, detail="server_restarting")
    try:
        table = await tables.create_table(
            req.name, req.fillWithAI, req.rules.model_dump()
        )
    except TableLimitError:
        raise HTTPException(status_code=503, detail="table_limit_reached")
    # A table whose players never open a websocket would otherwise linger
    # forever: the other autoclose triggers live in the ws connect/disconnect
    # paths. Give it a generous window to acquire its first connection.
    schedule_autoclose_if_no_humans(table, delay_seconds=300.0)
    return table.to_public_dict()


@router.post("/api/tables/{table_id}/join")
@limiter.limit(CREATE_JOIN)
async def join_table(request: Request, table_id: str, req: JoinTableRequest):
    try:
        table = tables.get_table(table_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="table_not_found") from e

    if table.status == "finished":
        raise HTTPException(status_code=400, detail="table_finished")

    pool = get_db_pool()
    # Identity comes from the bearer token, never from the request body: a
    # valid token maps to an existing player; otherwise the server mints a
    # fresh player + session token and returns the token once.
    identity = await optional_player(request)
    session_token: Optional[str] = None
    if identity is not None:
        player_uuid = identity.id
        await players_db.ensure_player(pool, player_uuid)
    else:
        player_uuid = uuid.uuid4()
        await players_db.ensure_player(pool, player_uuid)
        session_token = await sessions_db.create_session(pool, player_uuid)

    client_id = str(uuid.uuid4())
    conn = ClientConn(
        client_id=client_id,
        display_name=req.display_name,
        seat=None,
        player_id=str(player_uuid),
    )
    async with table.state_lock:
        prune_table_state(table)
        table.clients[client_id] = conn
        if not table.host_client_id:
            table.host_client_id = client_id

        if table.status == "playing":
            ai_seat = _pick_join_ai_seat(table)
            if ai_seat is None:
                try:
                    del table.clients[client_id]
                except KeyError:
                    pass
                raise HTTPException(status_code=400, detail="no_ai_seat_available")
            await _replace_ai_with_human_and_reserve(table, ai_seat, client_id)
        else:
            seat_to_take: Optional[int] = None
            if table.host_client_id == client_id:
                if not table.seats.get(5) or _is_ai_occupant(
                    table, table.seats.get(5)
                ):
                    seat_to_take = 5
            if seat_to_take is None:
                seat_to_take = _lowest_non_human_seat(table)

            if seat_to_take is not None:
                prev_occ = table.seats.get(seat_to_take)
                if _is_ai_occupant(table, prev_occ):
                    await _replace_ai_with_human_and_reserve(
                        table, seat_to_take, client_id
                    )
                else:
                    table.seats[seat_to_take] = client_id
                    conn.seat = seat_to_take
                    msg_dict = await add_chat_message(
                        table,
                        "system",
                        f"{req.display_name} joined and took seat {seat_to_take}",
                    )
                    await broadcast_chat_append(table, msg_dict)
                    await broadcast_table_event(
                        table,
                        {
                            "type": "lobby_event",
                            "message": f"{req.display_name} joined and took seat {seat_to_take}",
                            "table": table.to_public_dict(),
                        },
                    )
                    await broadcast_table_update(table)
            else:
                msg_dict = await add_chat_message(
                    table, "system", f"{req.display_name} joined the table"
                )
                await broadcast_chat_append(table, msg_dict)
                await broadcast_table_event(
                    table,
                    {
                        "type": "lobby_event",
                        "message": f"{req.display_name} joined the table",
                        "table": table.to_public_dict(),
                    },
                )

    return {
        "client_id": client_id,
        "player_id": str(player_uuid),
        # Present only when a fresh identity was minted; the client must
        # store it and send it as Authorization: Bearer from then on.
        "session_token": session_token,
        "is_host": client_id == table.host_client_id,
        "table": table.to_public_dict(),
    }


@router.post("/api/tables/{table_id}/seat")
@limiter.limit(HOST_ACTIONS)
async def choose_seat(
    request: Request,
    table_id: str,
    req: SeatRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="table_not_found") from e

    if req.seat not in {1, 2, 3, 4, 5}:
        raise HTTPException(status_code=400, detail="invalid_seat")
    async with table.state_lock:
        # Check and write under the same lock so two clients cannot both pass
        # the occupancy check and land in one seat.
        current = table.seats.get(req.seat)
        if current and not (
            table.occupants.get(current).is_ai
            if table.occupants.get(current)
            else False
        ):
            raise HTTPException(status_code=409, detail="seat_taken")
        require_client(table, req.client_id, identity)

        for i in range(1, 6):
            if table.seats[i] == req.client_id:
                table.seats[i] = None
                break

        prev_occ = table.seats.get(req.seat)
        table.seats[req.seat] = req.client_id
        table.clients[req.client_id].seat = req.seat
        if _is_ai_occupant(table, prev_occ):
            if prev_occ in _reserved_ai_ids(table):
                placeholder = _allocate_ai_occupant()
                table.occupants[placeholder.id] = placeholder
                table.reserved_ai_by_human[req.client_id] = placeholder.id
            else:
                table.reserved_ai_by_human[req.client_id] = prev_occ  # type: ignore[assignment]

    display_name = table.clients[req.client_id].display_name
    msg_dict = await add_chat_message(
        table, "system", f"{display_name} took seat {req.seat}"
    )
    await broadcast_chat_append(table, msg_dict)
    await broadcast_table_update(table)

    return table.to_public_dict()


@router.patch("/api/tables/{table_id}/rules")
@limiter.limit(HOST_ACTIONS)
async def update_table_rules(
    request: Request,
    table_id: str,
    req: UpdateTableRulesRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if table.status != "open":
        raise HTTPException(status_code=400, detail="game_already_started")

    require_host(table, req.client_id, identity)

    if not table.rules:
        table.rules = {}
    table.rules.update(req.rules.model_dump(exclude_none=True))
    await broadcast_table_update(table)

    return {"status": "success", "rules": table.rules}


@router.post("/api/tables/{table_id}/close")
@limiter.limit(HOST_ACTIONS)
async def api_close_table(
    request: Request,
    table_id: str,
    req: CloseTableRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    require_host(table, req.client_id, identity)

    await close_table(table, reason="host_closed")
    return {"ok": True}
