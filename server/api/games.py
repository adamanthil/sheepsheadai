from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player
from server.api.ratelimit import GAME_ACTIONS, HOST_ACTIONS, limiter
from server.api.schemas import (
    ActionRequest,
    RedealRequest,
    StartGameRequest,
)
from server.api.tables import require_client, require_host
from server.config import get_settings
from server.realtime.broadcast import (
    broadcast_table_state,
    broadcast_table_update,
)
from server.realtime.chat import add_chat_message, broadcast_chat_append
from server.runtime.ai_loop import ai_observe_all, schedule_ai_turns
from server.runtime.tables import (
    Occupant,
    _try_int,
    get_actor_seat,
    get_valid_action_ids_for_seat,
    record_hand_result,
    tables,
)
from server.services.ai_loader import load_agent
from server.services.persistence.games import (
    capture_post_state,
    capture_pre_state,
    ensure_game_table,
    fire_game_hooks,
    persist_started_game,
)
from server.services.persistence.pool import get_db_pool
from sheepshead import ACTION_LOOKUP, CARD_FULL_NAMES, Game

router = APIRouter()


_AI_NAME_POOL = ("Dan", "Kyle", "John", "Trevor", "Tim", "Tom")


def _fill_empty_seats_with_ai(table) -> None:
    """Populate every empty seat with a fresh AI occupant."""
    for i in range(1, 6):
        if not table.seats[i]:
            occ_id = str(uuid.uuid4())
            display_name = _AI_NAME_POOL[(i - 1) % len(_AI_NAME_POOL)]
            table.occupants[occ_id] = Occupant(
                id=occ_id, display_name=display_name, is_ai=True
            )
            table.seats[i] = occ_id


@router.post("/api/tables/{table_id}/fill_ai")
@limiter.limit(HOST_ACTIONS)
async def fill_ai(
    request: Request,
    table_id: str,
    req: RedealRequest | None = None,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="table_not_found") from e

    require_host(table, req.client_id if req else None, identity)
    async with table.state_lock:
        _fill_empty_seats_with_ai(table)

    await broadcast_table_update(table)
    return table.to_public_dict()


@router.post("/api/tables/{table_id}/start")
@limiter.limit(HOST_ACTIONS)
async def start_game(
    request: Request,
    table_id: str,
    req: StartGameRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    require_host(table, req.client_id, identity)

    async with table.state_lock:
        # Status is re-checked under the lock so a double-submit cannot
        # build two Games for one table.
        if table.status != "open":
            raise HTTPException(status_code=400, detail="already_started")

        if table.fill_with_ai:
            _fill_empty_seats_with_ai(table)

        if not all(table.seats[i] for i in range(1, 6)):
            raise HTTPException(status_code=400, detail="not_enough_players")

        if not table.host_client_id or all(
            table.seats[i] != table.host_client_id for i in range(1, 6)
        ):
            raise HTTPException(status_code=400, detail="host_not_seated")

        rules = table.rules or {}
        partner_mode = _try_int(rules.get("partnerMode", 1), 1)
        double_on_the_bump = bool(rules.get("doubleOnTheBump", True))

        game = Game(
            double_on_the_bump=double_on_the_bump,
            partner_selection_mode=partner_mode,
        )
        table.game = game
        table.status = "playing"
        table.results_counted = False
        if not table.initial_seat_order:
            table.initial_seat_order = [str(table.seats[i] or "") for i in range(1, 6)]
            pub = table.to_public_dict()
            for i in range(1, 6):
                occ = table.seats[i]
                if occ:
                    table.initial_names[str(occ)] = pub["seats"][i]

        has_ai = any(
            (occ_id and (occ := table.occupants.get(occ_id)) and occ.is_ai)
            for occ_id in table.seats.values()
        )
        if has_ai:
            settings = get_settings()
            table.ai_agent = load_agent(settings.sheepshead_model_path)

    await broadcast_table_update(table)
    await broadcast_table_state(table)
    schedule_ai_turns(table, initial_delay=2.0)

    pool = get_db_pool()
    await ensure_game_table(pool, table.id, table.name)
    await persist_started_game(pool, table, game)

    return table.to_public_dict()


@router.post("/api/tables/{table_id}/redeal")
@limiter.limit(HOST_ACTIONS)
async def redeal(
    request: Request,
    table_id: str,
    req: RedealRequest | None = None,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    require_host(table, req.client_id if req else None, identity)

    async with table.state_lock:
        old = {i: table.seats[i] for i in range(1, 6)}
        new_map = {
            1: old[2],
            2: old[3],
            3: old[4],
            4: old[5],
            5: old[1],
        }
        for i in range(1, 6):
            table.seats[i] = new_map[i]
            occ = new_map[i]
            if occ and occ in table.clients:
                table.clients[occ].seat = i

        table.game = None
        table.status = "open"
        table.results_counted = False

    return table.to_public_dict()


@router.post("/api/tables/{table_id}/action")
@limiter.limit(GAME_ACTIONS)
async def post_action(
    request: Request,
    table_id: str,
    req: ActionRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if not table.game:
        raise HTTPException(status_code=400, detail="game_not_started")

    conn = require_client(table, req.client_id, identity)
    if not conn.seat:
        raise HTTPException(status_code=400, detail="client_not_joined")

    pre = None
    post = None
    async with table.game_lock:
        # Validate under the lock: turn order and the valid-action set can
        # change between an unlocked check and the apply (e.g. a concurrent
        # AI move or a double-submitted request).
        actor_seat = get_actor_seat(table)
        if actor_seat != conn.seat:
            raise HTTPException(status_code=400, detail="not_your_turn")

        valid = get_valid_action_ids_for_seat(table, conn.seat)
        if req.action_id not in valid:
            raise HTTPException(status_code=400, detail="invalid_action")

        player = table.game.players[conn.seat - 1]
        pre = capture_pre_state(table.game)
        ok = player.act(int(req.action_id))
        if not ok:
            raise HTTPException(status_code=400, detail="apply_failed")
        post = capture_post_state(table.game)

    if pre is not None and post is not None:
        await fire_game_hooks(table, pre, post)

    action_str = ACTION_LOOKUP.get(req.action_id, "")
    display_name = conn.display_name
    if action_str == "PICK":
        msg_dict = await add_chat_message(table, "system", f"{display_name} picked")
        await broadcast_chat_append(table, msg_dict)
    elif action_str == "PASS":
        msg_dict = await add_chat_message(table, "system", f"{display_name} passed")
        await broadcast_chat_append(table, msg_dict)
    elif action_str == "ALONE":
        msg_dict = await add_chat_message(table, "system", f"{display_name} goes alone")
        await broadcast_chat_append(table, msg_dict)
    elif action_str == "JD PARTNER":
        msg_dict = await add_chat_message(
            table, "system", f"{display_name} chose JD partner"
        )
        await broadcast_chat_append(table, msg_dict)
    elif action_str.startswith("CALL "):
        parts = action_str.split()
        called_card = parts[1] if len(parts) > 1 else ""
        under = "under" if len(parts) > 2 and parts[2] == "UNDER" else ""
        card_display = CARD_FULL_NAMES.get(called_card, called_card)
        call_msg = f"{display_name} calls {card_display}"
        if under:
            call_msg += " under"
        msg_dict = await add_chat_message(table, "system", call_msg)
        await broadcast_chat_append(table, msg_dict)

    await ai_observe_all(table, except_seat=conn.seat)
    await broadcast_table_state(table)
    schedule_ai_turns(table)

    if table.game and table.game.is_done():
        table.status = "finished"
        record_hand_result(table)
        await broadcast_table_state(table)

    return {"ok": True}
