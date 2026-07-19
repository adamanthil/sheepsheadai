"""Shared helpers for the PPO update-path test suites.

``play_episodes`` fills an agent's event storage with seeded self-play
episodes (deals seeded deterministically, both partner modes, rotating
seats); ``prepare_minibatch_inputs`` mirrors update()'s preprocessing so
the minibatch builders can be tested on realistic events.
"""

import random
from types import SimpleNamespace

import numpy as np
import torch

from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import pfsp_runtime


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def play_episodes(agent: PPOAgent, n: int, collect_oracle: bool, seed0: int) -> None:
    opponents = [
        SimpleNamespace(agent=agent, metadata=SimpleNamespace(agent_id="self"))
        for _ in range(4)
    ]
    original_game = pfsp_runtime.Game
    deal_counter = {"n": 0}

    class _SeededGame(original_game):
        def __init__(self, *args, **kwargs):
            deal_counter["n"] += 1
            kwargs.setdefault("seed", seed0 + deal_counter["n"])
            super().__init__(*args, **kwargs)

    pfsp_runtime.Game = _SeededGame
    try:
        for episode in range(n):
            mode = PARTNER_BY_CALLED_ACE if episode % 2 == 0 else PARTNER_BY_JD
            _, events, _, _, _ = pfsp_runtime.play_population_game(
                training_agent=agent,
                opponents=opponents,
                partner_mode=mode,
                training_agent_position=(episode % 5) + 1,
                reward_mode="terminal",
                collect_oracle=collect_oracle,
            )
            agent.store_episode_events(events)
    finally:
        pfsp_runtime.Game = original_game


def prepare_minibatch_inputs(agent: PPOAgent):
    """Mirror update()'s preprocessing up to (not including) the epoch loop.

    Same code, same order, run on the same agent/events: GAE (limited-critic
    path), advantage normalization written back into self.events, then the
    static views and segment boundaries that feed _build_minibatch_tensors.
    """
    advantages, _returns = agent.compute_gae()
    if advantages.size:
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        for e in agent.events:
            if e.get("kind") == "action":
                e["advantage"] = float((e["advantage"] - adv_mean) / adv_std)
    states, masks_t, kinds = agent._prepare_training_views()
    segments = agent._segments_from_events(kinds)
    return states, masks_t, kinds, segments
