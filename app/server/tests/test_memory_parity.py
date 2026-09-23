"""A live table's AI memory matches training, seat for seat, bit for bit.

The table AI keeps recurrent memory for every seat, human or AI, so it can
take any seat over mid-hand as if it had played it from the deal. That only
holds if each seat's memory follows the training schedule exactly: its own
decisions, plus a last-trick observation for every seat after each trick
but the last. This plays one deal both ways -- a training-style replay with
the agent at every seat, and the server with humans at two seats making the
replay's moves -- and requires identical memory for all five seats.
"""

from __future__ import annotations

import copy
import types
import uuid

import httpx
import pytest
import torch
from fastapi import Request

from server.api.auth import PlayerIdentity, current_player
from server.runtime import ai_loop, turn_timer
from server.runtime.ai_loop import schedule_ai_turns
from server.runtime.dealing import new_game_for_table
from server.runtime.tables import ClientConn, Occupant, Table, get_actor_seat, tables
from sheepshead import ACTIONS
from sheepshead.agent.observation import last_trick_observation_for, observation_for
from sheepshead.agent.ppo import PPOAgent, device

HUMAN_SEATS = (2, 4)


def _training_replay(agent: PPOAgent, game) -> list[tuple[int, int]]:
    """Play ``game`` out as the trainer does; return (seat, action) moves."""
    moves = []
    while not game.is_done():
        player = next(p for p in game.players if p.get_valid_action_ids())
        action, _, _ = agent.act(
            observation_for(player, agent),
            player.get_valid_action_ids(),
            player.position,
            deterministic=True,
        )
        moves.append((player.position, action))
        player.act(action)
        if game.was_trick_just_completed and not game.is_done():
            for p in game.players:
                agent.observe(
                    last_trick_observation_for(p, agent), player_id=p.position
                )
    return moves


@pytest.mark.parametrize("arch", ["full", "perceiver-shared-v2"])
async def test_table_memory_matches_the_training_schedule(app, monkeypatch, arch):
    torch.manual_seed(0)
    reference = PPOAgent(len(ACTIONS), critic_mode="limited", arch=arch)
    served = copy.deepcopy(reference)

    table = Table(id="parity", name="parity")
    players: dict[int, uuid.UUID] = {}
    for seat in range(1, 6):
        if seat in HUMAN_SEATS:
            pid = uuid.uuid4()
            players[seat] = pid
            cid = f"h{seat}"
            table.clients[cid] = ClientConn(
                client_id=cid, display_name=cid, seat=seat, player_id=str(pid)
            )
            table.seats[seat] = cid
        else:
            table.occupants[f"ai{seat}"] = Occupant(
                id=f"ai{seat}", display_name=f"AI{seat}", is_ai=True
            )
            table.seats[seat] = f"ai{seat}"
    deal = new_game_for_table(table)
    table.game = copy.deepcopy(deal)
    table.status = "playing"
    table.ai_agent = served
    tables.tables[table.id] = table

    moves = _training_replay(reference, copy.deepcopy(deal))

    # Skip the AI loop's pacing pauses and never let a turn clock fire.
    async def no_pause(seconds: float) -> None:
        return None

    monkeypatch.setattr(
        ai_loop,
        "asyncio",
        types.SimpleNamespace(
            sleep=no_pause,
            create_task=ai_loop.asyncio.create_task,
            CancelledError=ai_loop.asyncio.CancelledError,
        ),
    )
    monkeypatch.setattr(turn_timer, "turn_timeout_seconds", lambda: 3600.0)

    def fake_player(request: Request) -> PlayerIdentity:
        return PlayerIdentity(id=uuid.UUID(request.headers["x-test-player"]))

    app.dependency_overrides[current_player] = fake_player
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            schedule_ai_turns(table)
            for seat, action in moves:
                if table.ai_task:
                    await table.ai_task  # the AIs play up to the next human
                if seat not in HUMAN_SEATS:
                    continue
                assert get_actor_seat(table) == seat
                r = await client.post(
                    "/api/tables/parity/action",
                    json={"client_id": f"h{seat}", "action_id": action},
                    headers={"x-test-player": str(players[seat])},
                )
                assert r.status_code == 200, r.text
            if table.ai_task:
                await table.ai_task
    finally:
        app.dependency_overrides.clear()
        turn_timer.cancel_turn_timer(table)

    assert table.game is not None and table.game.is_done()
    for seat in range(1, 6):
        assert torch.equal(
            served.get_recurrent_memory(seat, device=device),
            reference.get_recurrent_memory(seat, device=device),
        ), f"seat {seat} memory diverged from training"
