"""The current observation and the legacy contracts that adapt it."""

import numpy as np
import pytest
import torch

from sheepshead.agent.architectures.registry import ARCHITECTURES
from sheepshead.agent.observation import (
    LEGACY_PICK_SEAT_SURGERY,
    LEGACY_SEATLESS_RECALL,
    observation_for,
    oracle_observation_for,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.game import ACTIONS, Game

CURRENT_ARCHITECTURES = {"perceiver-recall-seated"}
PASS = ACTIONS.index("PASS") + 1
OPENING_LEADER_REL = [(1, 1), (2, 5), (3, 4), (4, 3), (5, 2)]


class Contracted:
    def __init__(self, legacy_observation):
        self.legacy_observation = legacy_observation


CURRENT = Contracted(None)
SEATLESS = Contracted(LEGACY_SEATLESS_RECALL)
PICK_SEAT = Contracted(LEGACY_PICK_SEAT_SURGERY)


def seated_at_pick(seat):
    game = Game(seed=11)
    for earlier in game.players[: seat - 1]:
        earlier.act(PASS)
    player = game.players[seat - 1]
    assert ACTIONS.index("PICK") + 1 in player.get_valid_action_ids()
    return player


def play_started_game():
    game = Game(seed=11)
    while not game.play_started:
        player = next(p for p in game.players if p.get_valid_action_ids())
        player.act(min(player.get_valid_action_ids()))
    return game


def test_only_deliberately_current_architectures_skip_the_legacy_contract():
    current = {
        name for name, spec in ARCHITECTURES.items() if spec.legacy_observation is None
    }
    assert current == CURRENT_ARCHITECTURES


@pytest.mark.parametrize("seat,expected", OPENING_LEADER_REL)
def test_state_reports_the_dealer_relative_seat_before_play(seat, expected):
    player = seated_at_pick(seat)
    assert int(player.get_state_dict()["leader_rel"]) == expected
    assert int(player.get_oracle_state_dict()["leader_rel"]) == expected


@pytest.mark.parametrize("seat,expected", OPENING_LEADER_REL)
def test_actor_contract_decides_the_pre_play_leader(seat, expected):
    player = seated_at_pick(seat)
    assert int(observation_for(player, CURRENT)["leader_rel"]) == expected
    assert int(observation_for(player, PICK_SEAT)["leader_rel"]) == expected
    assert int(observation_for(player, SEATLESS)["leader_rel"]) == 0


@pytest.mark.parametrize("seat,expected", OPENING_LEADER_REL)
def test_oracle_contract_decides_the_pre_play_leader(seat, expected):
    player = seated_at_pick(seat)
    assert int(oracle_observation_for(player, CURRENT)["leader_rel"]) == expected
    assert int(oracle_observation_for(player, PICK_SEAT)["leader_rel"]) == 0
    assert int(oracle_observation_for(player, SEATLESS)["leader_rel"]) == 0


@pytest.mark.parametrize("agent", [CURRENT, PICK_SEAT, SEATLESS])
def test_contracts_leave_the_observation_alone_once_play_starts(agent):
    game = play_started_game()
    player = next(p for p in game.players if p.get_valid_action_ids())
    state = player.get_state_dict()
    oracle = player.get_oracle_state_dict()
    assert int(state["leader_rel"]) > 0
    for key, value in observation_for(player, agent).items():
        assert np.array_equal(state[key], value), key
    for key, value in oracle_observation_for(player, agent).items():
        assert np.array_equal(oracle[key], value), key


def test_seat_reaches_the_seated_trunk_but_not_the_seatless_one():
    hand = seated_at_pick(1).hand

    def encoded(arch, seat):
        torch.manual_seed(0)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        player = seated_at_pick(seat)
        player.hand = list(hand)
        state = observation_for(player, agent)
        with torch.no_grad():
            return agent.encoder.encode_batch([state])["all_tokens"]

    assert not torch.equal(
        encoded("perceiver-recall-seated", 1), encoded("perceiver-recall-seated", 5)
    )
    assert torch.equal(encoded("perceiver-recall", 1), encoded("perceiver-recall", 5))
