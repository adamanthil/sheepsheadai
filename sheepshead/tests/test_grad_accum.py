#!/usr/bin/env python3
"""Gradient-accumulation update path (Learning_System_Redesign batch-λ arm).

``update(grad_accum=True)`` must (a) equal the historical path bit-for-bit
when the whole buffer fits one minibatch — one backward, scale 1.0, step at
loop end is the same computation — and (b) apply exactly one optimizer step
per epoch when the buffer spans several minibatches."""

import pytest
import torch

from sheepshead import ACTIONS
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import play_episodes, seed_all

SEED = 20260722
N_EPISODES = 6

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _fresh_agent_with_buffer():
    seed_all(SEED)
    agent = PPOAgent(len(ACTIONS), arch="full", critic_mode="limited")
    play_episodes(agent, N_EPISODES, collect_oracle=False, seed0=SEED * 10)
    return agent


def test_single_minibatch_accum_is_bit_identical():
    agent_a = _fresh_agent_with_buffer()
    agent_b = _fresh_agent_with_buffer()

    seed_all(SEED + 1)
    agent_a.update(epochs=2, batch_size=256)  # buffer << 256: one minibatch
    seed_all(SEED + 1)
    agent_b.update(epochs=2, batch_size=256, grad_accum=True)

    params_a = dict(agent_a.encoder.named_parameters())
    for name, p_b in agent_b.encoder.named_parameters():
        assert torch.equal(params_a[name], p_b), name
    params_a = dict(agent_a.actor.named_parameters())
    for name, p_b in agent_b.actor.named_parameters():
        assert torch.equal(params_a[name], p_b), name
    params_a = dict(agent_a.critic.named_parameters())
    for name, p_b in agent_b.critic.named_parameters():
        assert torch.equal(params_a[name], p_b), name


def test_multi_minibatch_accum_steps_once_per_epoch():
    agent = _fresh_agent_with_buffer()
    n_segments = len(
        agent._segments_from_events([e["kind"] for e in agent.events])
    )
    assert n_segments >= 4  # batch_size=2 gives >= 2 minibatches per epoch

    seed_all(SEED + 2)
    stats = agent.update(epochs=3, batch_size=2, grad_accum=True)
    assert stats["timing"]["optimizer_steps"] == 3  # one per epoch


def test_multi_minibatch_per_step_mode_unchanged():
    agent = _fresh_agent_with_buffer()
    n_segments = len(
        agent._segments_from_events([e["kind"] for e in agent.events])
    )

    seed_all(SEED + 3)
    stats = agent.update(epochs=1, batch_size=2)
    expected_steps = -(-n_segments // 2)  # ceil: one step per minibatch
    assert stats["timing"]["optimizer_steps"] == expected_steps
