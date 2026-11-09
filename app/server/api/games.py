from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from server.api.auth import PlayerIdentity, current_player
from server.api.ratelimit import GAME_ACTIONS, HOST_ACTIONS, limiter
from server.api.schemas import (
    ActionRequest,
    NextHandRequest,
    OkResponse,
    StartGameRequest,
    TablePublic,
)
from server.api.tables import require_client, require_host
from server.realtime.broadcast import (
    broadcast_table_state,
    broadcast_table_update,
)
from server.realtime.chat import emit_bid_chat_message, post_presence_notice
from server.runtime.ai_loop import schedule_ai_turns
from server.runtime.ai_move import observe_human_decision, observe_trick_end
from server.runtime.dealing import (
    hand_passed_out,
    new_game_for_table,
    redeal_passed_out_hand,
    refresh_table_agent,
)
from server.runtime.lifecycle import finish_hand, is_draining
from server.runtime.manager import tables
from server.runtime.models import ClientConn, Table
from server.runtime.rules import plays_doublers
from server.runtime.scores import assign_score_columns, rotate_score_columns
from server.runtime.seating import (
    home_seat,
    pick_join_ai_seat,
    schedule_ai_replacement_for_disconnected_human,
    seat_queued_joiners,
)
from server.runtime.timing import FIRST_AI_MOVE_DELAY_SECONDS
from server.runtime.turn_timer import cancel_turn_timer
from server.runtime.views import (
    get_actor_seat,
    get_valid_action_ids_for_seat,
)
from server.services.persistence.game_table import ensure_game_table
from server.services.persistence.games import fire_game_hooks
from server.services.persistence.hand_setup import persist_started_game
from server.services.persistence.pool import get_db_pool
from server.services.persistence.snapshots import capture_post_state, capture_pre_state
from sheepshead.agent.observation import observation_for
from sheepshead.game import ACTION_LOOKUP, Game

router = APIRouter()


@router.post("/api/tables/{table_id}/start", response_model=TablePublic)
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

    if is_draining():
        raise HTTPException(status_code=503, detail="server_restarting")
    require_host(table, req.client_id, identity)

    async with table.state_lock:
        # Status is re-checked under the lock so a double-submit cannot
        # build two Games for one table.
        if table.status != "open":
            raise HTTPException(status_code=400, detail="already_started")
        _require_full_table(table)
        # The first deal waits for the host to sit down.
        if not table.host_client_id or all(
            table.seats[i] != table.host_client_id for i in range(1, 6)
        ):
            raise HTTPException(status_code=400, detail="host_not_seated")
        game = _deal(table)

    await _announce_deal(table, game)
    return table.to_public_dict()


@router.post("/api/tables/{table_id}/next-hand", response_model=TablePublic)
@limiter.limit(HOST_ACTIONS)
async def next_hand(
    request: Request,
    table_id: str,
    req: NextHandRequest,
    identity: PlayerIdentity = Depends(current_player),
):
    """Pass the deal to the next player and deal them a hand.

    One step, so a refusal leaves the finished hand exactly as it was: the
    host is told why and can try again, and no one's seat has moved.

    A host who isn't seated is sat back down first: dealing on means they
    mean to play. The AI may hold their seat because their phone slept on
    the hand summary (and this request beat their reconnect), or the turn
    clock moved them out, or host passed to someone watching.
    """
    try:
        table = tables.get_table(table_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="table_not_found")

    if is_draining():
        raise HTTPException(status_code=503, detail="server_restarting")
    require_host(table, req.client_id, identity)

    async with table.state_lock:
        if table.settling:
            raise HTTPException(status_code=409, detail="table_closing")
        # Mid-hand this would throw away a live deal unrecorded, letting a
        # host erase a losing hand. Also what a double-submit sees.
        if table.hand_in_play:
            raise HTTPException(status_code=409, detail="hand_in_progress")
        # Only a hand finish_hand has recorded is followed by another.
        if table.status != "finished" or table.game is None:
            raise HTTPException(status_code=400, detail="no_hand_finished")
        host = _seat_the_host(table)
        joiners = seat_queued_joiners(table)
        _require_full_table(table)
        _pass_the_deal(table)
        game = _deal(table)

    # Seated by this request rather than a reconnect: if the socket
    # doesn't come back, the AI takes the seat again after the usual grace.
    if host is not None and not host.connected:
        schedule_ai_replacement_for_disconnected_human(table, host.client_id)
    for conn, _ in joiners:
        await post_presence_notice(table, conn, "took a seat", toast=False)
    await _announce_deal(table, game)
    return table.to_public_dict()


def _seat_the_host(table: Table) -> ClientConn | None:
    """Sit an unseated host back down and return them, or None if the host
    is already seated or every seat is held by another human (the hand is
    then dealt with the host watching).

    Their own seat if the AI holds it for them, else the first AI seat not
    held for someone else. Caller holds ``state_lock``.
    """
    host = table.clients.get(table.host_client_id or "")
    if host is None or host.seat is not None:
        return None
    seat = home_seat(table, host) or pick_join_ai_seat(table)
    if seat is None:
        return None
    ai = table.seats[seat]
    table.seats[seat] = host.client_id
    host.seat = seat
    host.home_occupant = None
    host.timeout_strikes = 0
    # The AI they displaced is the one that stands in for them next time.
    if ai is not None:
        table.reserved_ai_by_human[host.client_id] = ai
    return host


def _require_full_table(table: Table) -> None:
    """Refuse a deal unless every seat is filled."""
    if not all(table.seats[i] for i in range(1, 6)):
        raise HTTPException(status_code=400, detail="not_enough_players")


def _pass_the_deal(table: Table) -> None:
    """Rotate every occupant down a seat, seat 1's into seat 5.

    The dealer is always seat 5, so this passes the deal to the left.
    """
    old = {i: table.seats[i] for i in range(1, 6)}
    for i in range(1, 6):
        occ = old[i % 5 + 1]
        table.seats[i] = occ
        if occ and occ in table.clients:
            table.clients[occ].seat = i
    rotate_score_columns(table)


def _deal(table: Table) -> Game:
    """Deal a fresh hand at the table. Caller holds ``state_lock``."""
    game = new_game_for_table(table)
    table.game = game
    table.status = "playing"
    table.ever_dealt = True
    table.hand_finished_at = None
    table.results_counted = False
    # A fresh hand is played for the base stake; only a passed-out
    # doublers deal raises it, and only until that hand is played out.
    table.doublings = 0
    table.pass_out_count = 0
    assign_score_columns(table)

    refresh_table_agent(table)
    return game


async def _announce_deal(table: Table, game: Game) -> None:
    """Show everyone the new hand, set the AI moving, and record it."""
    await broadcast_table_update(table)
    await broadcast_table_state(table)
    schedule_ai_turns(table, initial_delay=FIRST_AI_MOVE_DELAY_SECONDS)

    pool = get_db_pool()
    await ensure_game_table(pool, table.id, table.name, plays_doublers(table.rules))
    await persist_started_game(pool, table, game)


@router.post("/api/tables/{table_id}/action", response_model=OkResponse)
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
    t_start = time.perf_counter()
    async with table.game_lock:
        t_locked = time.perf_counter()
        # Validate under the lock: turn order and the valid-action set can
        # change between an unlocked check and the apply (e.g. a concurrent
        # AI move or a double-submitted request).
        if table.settling:
            raise HTTPException(status_code=409, detail="table_closing")
        actor_seat = get_actor_seat(table)
        if actor_seat != conn.seat:
            raise HTTPException(status_code=400, detail="not_your_turn")

        valid = get_valid_action_ids_for_seat(table, conn.seat)
        if req.action_id not in valid:
            raise HTTPException(status_code=400, detail="invalid_action")

        player = table.game.players[conn.seat - 1]
        pre = capture_pre_state(table.game)
        decision_state = (
            observation_for(player, table.ai_agent) if table.ai_agent else None
        )
        ok = player.act(int(req.action_id))
        if not ok:
            raise HTTPException(status_code=400, detail="apply_failed")
        table.move_seq += 1
        post = capture_post_state(table.game)
        t_applied = time.perf_counter()
        # The AI's memory for this seat follows the human's play, in
        # training order: the decision, then any trick it completed.
        if decision_state is not None:
            await observe_human_decision(table, conn.seat, decision_state)
        await observe_trick_end(table)
        # A hand that just ended is recorded before anyone is shown it ended,
        # and a passed-out doublers deal is replaced before its leaster state
        # can go out; every other move is shown before it is persisted.
        show_first = not table.game.is_done() and not hand_passed_out(table)
    t_observed = time.perf_counter()
    # Moving in time clears the player's timeout strikes.
    conn.timeout_strikes = 0
    cancel_turn_timer(table)

    if show_first:
        await broadcast_table_state(table)
    t_shown = time.perf_counter()

    if pre is not None and post is not None:
        await fire_game_hooks(table, pre, post, seat=conn.seat, by_ai=False)

    action_str = ACTION_LOOKUP.get(req.action_id, "")
    await emit_bid_chat_message(table, action_str, conn.display_name)

    if not show_first:
        # A doublers table that just passed out swaps in a fresh deal here,
        # before anything is broadcast, so the momentary leaster state never
        # reaches a client.
        await redeal_passed_out_hand(table)

        if table.game and table.game.is_done():
            # Records the hand before broadcasting that it ended.
            await finish_hand(table)
        else:
            await broadcast_table_state(table)
    schedule_ai_turns(table)
    t_end = time.perf_counter()

    def ms(a: float, b: float) -> float:
        return round((b - a) * 1000, 1)

    # One line per human move, to see where the time before the player's
    # own move reaches the table goes (lock wait, inference, persistence).
    logging.info(
        "action timing table=%s seat=%s action=%r lock_wait=%s apply=%s "
        "observe=%s shown_at=%s after_shown=%s total=%s",
        table.id,
        conn.seat,
        action_str,
        ms(t_start, t_locked),
        ms(t_locked, t_applied),
        ms(t_applied, t_observed),
        ms(t_start, t_shown) if show_first else None,
        ms(t_shown, t_end),
        ms(t_start, t_end),
    )

    return {"ok": True}
