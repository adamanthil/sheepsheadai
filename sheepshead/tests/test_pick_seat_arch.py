"""The perceiver-recall-pickseat surgery: inert at init, seat-aware after."""

import numpy as np
import pytest
import torch

from sheepshead.agent.architectures.pick_seat import PICK_CONTEXT_COUNT
from sheepshead.agent.observation import (
    hide_pre_play_leader,
    last_trick_observation_for,
    observation_for,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.game import ACTIONS, Game

ARCH = "perceiver-recall-pickseat"
PICK_ID = ACTIONS.index("PICK") + 1
PASS_ID = ACTIONS.index("PASS") + 1


def seated_at_pick(seat: int, mode: int = 0, seed: int = 3):
    game = Game(partner_selection_mode=mode, seed=seed)
    for s in range(1, seat):
        game.players[s - 1].act(PASS_ID)
    return game, game.players[seat - 1]


def migrated_pair():
    base = PPOAgent(len(ACTIONS), arch="perceiver-recall")
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    agent.encoder.load_state_dict(base.encoder.state_dict())
    agent.actor.load_state_dict(base.actor.state_dict(), strict=False)
    agent.critic.load_state_dict(base.critic.state_dict(), strict=False)
    return base, agent


@pytest.mark.parametrize("seat,expected", [(1, 1), (2, 5), (3, 4), (4, 3), (5, 2)])
def test_aware_agent_sees_dealer_relative_seat_before_play(seat, expected):
    aware = PPOAgent(len(ACTIONS), arch=ARCH)
    blind = PPOAgent(len(ACTIONS), arch="perceiver-recall")
    _, player = seated_at_pick(seat)
    assert int(observation_for(player, aware)["leader_rel"]) == expected
    assert int(observation_for(player, blind)["leader_rel"]) == 0


def test_stripping_restores_the_seat_blind_observation():
    aware = PPOAgent(len(ACTIONS), arch=ARCH)
    blind = PPOAgent(len(ACTIONS), arch="perceiver-recall")
    _, player = seated_at_pick(4)
    stripped = hide_pre_play_leader(observation_for(player, aware))
    for key, value in observation_for(player, blind).items():
        assert np.array_equal(stripped[key], value), key


def test_migrated_agent_matches_base_exactly_over_a_game():
    base, agent = migrated_pair()
    for mode in (0, 1):
        game = Game(partner_selection_mode=mode, seed=11 + mode)
        base.reset_recurrent_state()
        agent.reset_recurrent_state()
        base.stash_action_probs = True
        agent.stash_action_probs = True
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    action, _, value_b = base.act(
                        observation_for(player, base),
                        valid,
                        player.position,
                        deterministic=True,
                    )
                    _, _, value_a = agent.act(
                        observation_for(player, agent),
                        valid,
                        player.position,
                        deterministic=True,
                    )
                    assert value_a == value_b
                    assert np.array_equal(
                        np.asarray(base.last_action_probs),
                        np.asarray(agent.last_action_probs),
                    )
                    player.act(action)
                    if game.was_trick_just_completed and not game.is_done():
                        for seat in game.players:
                            for a in (base, agent):
                                a.observe(
                                    last_trick_observation_for(seat, a),
                                    player_id=seat.position,
                                )
                    valid = player.get_valid_action_ids()


def _pick_probs(agent, player):
    agent.reset_recurrent_state()
    agent.stash_action_probs = True
    agent.act(
        observation_for(player, agent),
        player.get_valid_action_ids(),
        player.position,
        deterministic=True,
    )
    return np.asarray(agent.last_action_probs)


def test_residual_moves_only_its_own_seat_and_mode():
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    with torch.no_grad():
        # Seat 5 -> leader_rel 2; partner mode 1 -> context 4.
        agent.actor.pick_seat_residual.offset.weight[4] = 2.0
    _, seat5 = seated_at_pick(5, mode=1)
    _, seat1 = seated_at_pick(1, mode=1)
    _, seat5_jd = seated_at_pick(5, mode=0)
    fresh = PPOAgent(len(ACTIONS), arch=ARCH)
    fresh.encoder.load_state_dict(agent.encoder.state_dict())
    fresh.actor.load_state_dict(
        {
            k: (torch.zeros_like(v) if k.endswith("offset.weight") else v)
            for k, v in agent.actor.state_dict().items()
        }
    )
    pick = PICK_ID - 1
    assert _pick_probs(agent, seat5)[pick] > _pick_probs(fresh, seat5)[pick]
    assert np.array_equal(_pick_probs(agent, seat1), _pick_probs(fresh, seat1))
    assert np.array_equal(_pick_probs(agent, seat5_jd), _pick_probs(fresh, seat5_jd))


def test_pick_context_is_zero_after_the_pick():
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    game, player = seated_at_pick(1)
    player.act(PICK_ID)
    out = agent.encoder.encode_batch([observation_for(player, agent)])
    assert out["pick_context"].tolist() == [0]
    _, picking = seated_at_pick(3, mode=1)
    out = agent.encoder.encode_batch([observation_for(picking, agent)])
    # Seat 3 -> leader_rel 4; partner mode 1.
    context = out["pick_context"].item()
    assert context == (4 - 1) * 2 + 1 + 1
    assert 0 < context < PICK_CONTEXT_COUNT


def test_pick_seat_mode_trains_only_the_residual_and_critics():
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    agent.set_trainable_heads("pick_seat")
    trainable = {n for n, p in agent.actor.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith("pick_seat_residual") for n in trainable)
    assert not any(p.requires_grad for p in agent.encoder.parameters())
    assert all(p.requires_grad for p in agent.critic.parameters())


def _aux_outputs(agent, player):
    agent.reset_recurrent_state()
    out = agent.encoder.encode_batch([observation_for(player, agent)])
    return agent.critic.aux_predictions(out)[:2]


def test_win_and_return_residuals_are_inert_at_init():
    base, agent = migrated_pair()
    _, player = seated_at_pick(4, mode=1)
    assert _aux_outputs(agent, player) == _aux_outputs(base, player)


def test_win_and_return_residuals_read_the_seat():
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    with torch.no_grad():
        # Seat 5 -> leader_rel 2; partner mode 1 -> context 4.
        agent.critic.pick_seat_win_residual.offset.weight[4] = 1.0
        agent.critic.pick_seat_return_residual.offset.weight[4] = 0.5
    game, seat5 = seated_at_pick(5, mode=1)
    state = observation_for(seat5, agent)
    out = agent.encoder.encode_batch([state])
    win, ret = agent.critic.aux_predictions(out)[:2]
    feat = agent.critic._aux_features_single(out)
    plain_win = torch.sigmoid(agent.critic.win_head(feat)).item()
    plain_ret = agent.critic.return_head(feat).item()
    assert win > plain_win
    assert ret == pytest.approx(plain_ret + 0.5, abs=1e-5)
    _, seat1 = seated_at_pick(1, mode=1)
    out = agent.encoder.encode_batch([observation_for(seat1, agent)])
    feat = agent.critic._aux_features_single(out)
    assert agent.critic.aux_predictions(out)[1] == pytest.approx(
        agent.critic.return_head(feat).item(), abs=1e-6
    )
