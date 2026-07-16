#!/usr/bin/env python3
"""Bit-exact characterization of PPOAgent.update().

Pins the post-update weights (sha256 over every state_dict tensor's bytes)
and the full returned stats dict (minus wall-clock timing) for seeded
self-play episodes, across the limited critic, the oracle critic, and the
perceiver-shared-v2 architecture. Any refactor of the update path
(minibatch construction, GAE, loss assembly, optimizer stepping) must keep
these outputs byte-identical; a legitimate behavioral change requires
regenerating the fixture:

    uv run python -m sheepshead.tests.test_ppo_update_characterization

Regeneration runs each config twice in-process and refuses to write a
fixture that does not reproduce itself.
"""

import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.training import pfsp_runtime

# Plays seeded episodes and runs real optimizer updates for three configs (~1min).
pytestmark = pytest.mark.slow

SEED = 20260716
N_EPISODES = 6
UPDATE_KWARGS = {"epochs": 2, "batch_size": 16}

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ppo_update_characterization.json"

CONFIGS = {
    "full-limited": {
        "arch": "full",
        "critic_mode": "limited",
        "collect_oracle": False,
    },
    "full-oracle": {
        "arch": "full",
        "critic_mode": "oracle",
        "collect_oracle": True,
    },
    "perceiver-shared-v2-oracle": {
        "arch": "perceiver-shared-v2",
        "critic_mode": "oracle",
        "collect_oracle": True,
    },
}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _play_episodes(agent: PPOAgent, n: int, collect_oracle: bool, seed0: int) -> None:
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


def _agent_networks(agent: PPOAgent):
    networks = [("encoder", agent.encoder), ("actor", agent.actor), ("critic", agent.critic)]
    oracle_critic = getattr(agent, "oracle_critic", None)
    if oracle_critic is not None:
        networks.append(("oracle_critic", oracle_critic))
    return networks


def _weights_sha256(agent: PPOAgent) -> str:
    digest = hashlib.sha256()
    for name, network in _agent_networks(agent):
        state = network.state_dict()
        for key in sorted(state):
            digest.update(f"{name}.{key}".encode())
            digest.update(state[key].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _normalize(value):
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.generic):
        return _normalize(value.item())
    if isinstance(value, torch.Tensor):
        return _normalize(value.detach().cpu().tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def _normalized_stats(stats: dict) -> dict:
    normalized = _normalize(stats)
    normalized.pop("timing", None)
    return normalized


def run_config(name: str) -> dict:
    config = CONFIGS[name]
    _seed_all(SEED)
    agent = PPOAgent(len(ACTIONS), arch=config["arch"], critic_mode=config["critic_mode"])
    _play_episodes(agent, N_EPISODES, config["collect_oracle"], seed0=SEED * 10)
    stats = agent.update(**UPDATE_KWARGS)
    return {"state_hash": _weights_sha256(agent), "stats": _normalized_stats(stats)}


def _load_fixture() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_update_is_bit_identical(name):
    expected = _load_fixture()[name]
    actual = run_config(name)
    # Round-trip through JSON so float representations match the fixture's.
    actual = json.loads(json.dumps(actual))
    assert actual["state_hash"] == expected["state_hash"]
    assert actual["stats"] == expected["stats"]


def _regenerate() -> None:
    fixture = {}
    for name in CONFIGS:
        first = run_config(name)
        second = run_config(name)
        if first != second:
            raise SystemExit(f"{name}: update() is not deterministic in-process")
        print(f"{name}: state_hash={first['state_hash']}")
        fixture[name] = first
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump(fixture, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    _regenerate()
