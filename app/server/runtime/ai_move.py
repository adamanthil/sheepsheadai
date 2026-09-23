"""One AI move at one seat, and the AI's memory updates after a move.

Shared by the AI turn loop (AI seats) and the turn timer (a human seat
whose clock ran out). A leaf of the runtime import graph: it imports the
data model and services, never the loop, the timer, or the broadcasters.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from server.runtime.models import Table
from server.services.ai_loader import inference_limit
from server.services.persistence.snapshots import (
    capture_post_state,
    capture_pre_state,
)
from sheepshead import ACTION_LOOKUP
from sheepshead.agent.observation import observation_for


@dataclass(frozen=True)
class AppliedMove:
    action_id: int
    action_str: str
    # Persistence snapshots from either side of the move (fire_game_hooks).
    pre: Dict[str, Any]
    post: Dict[str, Any]


async def ai_act_for_seat(table: Table, seat: int) -> Optional[AppliedMove]:
    """Let the table's AI choose and apply the move for ``seat``.

    The caller holds ``table.game_lock`` and has checked it is ``seat``'s
    turn. Torch inference runs in a worker thread while the lock is held:
    this table stays consistent, every other table (and the event loop
    itself) keeps moving. Returns None when the seat has no legal move.
    """
    game = table.game
    agent = table.ai_agent
    if game is None or agent is None:
        return None
    player = game.players[seat - 1]
    state = observation_for(player, agent)
    valid = player.get_valid_action_ids()
    if not valid:
        return None
    pre = capture_pre_state(game)
    async with inference_limit:
        action_id, _, _ = await asyncio.to_thread(
            agent.act,
            state,
            valid_actions=valid,
            player_id=seat,
            deterministic=True,
        )
    if not player.act(int(action_id)):
        raise RuntimeError(
            f"AI produced invalid action_id {action_id} for seat {seat}; valid set: {sorted(list(valid))}"
        )
    table.move_seq += 1
    return AppliedMove(
        action_id=int(action_id),
        action_str=ACTION_LOOKUP.get(int(action_id), ""),
        pre=pre,
        post=capture_post_state(game),
    )


def _observe_seats(agent, observations: list[tuple[dict, int]]) -> None:
    for state, seat in observations:
        agent.observe(state, player_id=seat)


async def ai_observe_all(table: Table, except_seat: Optional[int] = None) -> None:
    """Update the AI's recurrent memory for every AI seat.

    observe() is a torch forward pass; run the batch in a worker thread so
    the event loop (all tables, websockets, /health) never blocks on it.
    """
    if not table.ai_agent or not table.game:
        return
    observations: list[tuple[dict, int]] = []
    for seat, occupant in table.seats.items():
        if not occupant:
            continue
        occ = table.occupants.get(occupant)
        if not occ or not occ.is_ai:
            continue
        if seat == except_seat:
            continue
        player = table.game.players[seat - 1]
        observations.append((observation_for(player, table.ai_agent), seat))
    if not observations:
        return
    async with inference_limit:
        await asyncio.to_thread(_observe_seats, table.ai_agent, observations)
