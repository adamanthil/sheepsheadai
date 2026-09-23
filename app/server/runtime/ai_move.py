"""One AI move at one seat, and the AI's memory updates around a move.

The table AI keeps recurrent memory for all five seats, human or not, fed
on exactly the training schedule: each seat's own decisions (act() for AI
moves, observe_human_decision for human ones) and, after every trick but
the last, a last-trick observation for every seat (observe_trick_end).
Nothing else touches memory, so what the AI knows at a seat is what it
would know had it played that seat from the deal.

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
from sheepshead.agent.observation import (
    last_trick_observation_for,
    observation_for,
)


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
    post = capture_post_state(game)
    await observe_trick_end(table)
    return AppliedMove(
        action_id=int(action_id),
        action_str=ACTION_LOOKUP.get(int(action_id), ""),
        pre=pre,
        post=post,
    )


def _observe_seats(agent, observations: list[tuple[dict, int]]) -> None:
    for state, seat in observations:
        agent.observe(state, player_id=seat)


async def _observe(table: Table, observations: list[tuple[dict, int]]) -> None:
    # observe() is a torch forward pass; run it in a worker thread so the
    # event loop (all tables, websockets, /health) never blocks on it.
    async with inference_limit:
        await asyncio.to_thread(_observe_seats, table.ai_agent, observations)


async def observe_human_decision(table: Table, seat: int, state: dict) -> None:
    """Give the AI's memory for ``seat`` the update its own act() makes at a
    decision, for a decision a human made from ``state``.

    act() updates memory with the encoder pass alone (the chosen action
    never enters memory), so observe() on the decision state is exactly
    that update. Every seat's memory then follows the training schedule
    whoever plays it, and the AI can take the seat over mid-hand.
    The caller holds ``table.game_lock``.
    """
    if table.ai_agent is not None:
        await _observe(table, [(state, seat)])


async def observe_trick_end(table: Table) -> None:
    """After a move that completed a trick (not the last), every seat
    observes the finished trick -- the training schedule's only memory
    update besides each seat's own decisions. The caller holds
    ``table.game_lock``.
    """
    game, agent = table.game, table.ai_agent
    if game is None or agent is None:
        return
    if not game.was_trick_just_completed or game.is_done():
        return
    await _observe(
        table,
        [(last_trick_observation_for(p, agent), p.position) for p in game.players],
    )
