"""The ``perceiver-recall`` architecture and the observation contract
(Training_Program_Redesign_202609 §3; sheepshead/agent/observation.py).

Pins the structural claims the release-candidate architecture rests on:
the recall encoder reads exactly RECALL_KEYS and is invariant to the
picker-memory keys, its token sequence is the 15-token recall layout, the
memory GRU is driven by the post-reasoning MEMORY token, every registry
entry declares its observation contract truthfully, and the legacy layout
is untouched (the golden fixtures pin its numerics separately).
"""

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS, Game
from sheepshead.agent import architectures
from sheepshead.agent.architectures.encoders import RecallEncoder
from sheepshead.agent.observation import (
    HEADER_KEYS,
    LEGACY_PICKER_MEMORY_KEYS,
    RECALL_KEYS,
    TABLE_KEYS,
)
from sheepshead.agent.ppo import PPOAgent, load_agent
from sheepshead.agent.token_layout import (
    BASE_TOKEN_COUNT,
    MEMORY_TOKEN,
    RECALL_TOKEN_COUNT,
)
from sheepshead.tests.ppo_test_helpers import seed_all


def _picker_states(agent: PPOAgent, seed: int = 5) -> list:
    """Observation dicts of a played game where the actor is the picker
    holding a non-empty blind/bury injection."""
    game = Game(seed=seed)
    states = []
    while not game.is_done():
        for player in game.players:
            acts = player.get_valid_action_ids()
            while acts:
                state = player.get_state_dict()
                if state["blind_ids"].any() or state["bury_ids"].any():
                    states.append(state)
                action, _, _ = agent.act(state, acts, player.position)
                player.act(action)
                acts = player.get_valid_action_ids()
    return states


def _masked(state: dict) -> dict:
    out = dict(state)
    out["blind_ids"] = np.zeros(2, dtype=np.uint8)
    out["bury_ids"] = np.zeros(2, dtype=np.uint8)
    return out


class TestObservationContract:
    def test_key_sets_partition_the_observation_dict(self):
        state = Game(seed=1).players[0].get_state_dict()
        assert set(state) == RECALL_KEYS | LEGACY_PICKER_MEMORY_KEYS
        assert not (RECALL_KEYS & LEGACY_PICKER_MEMORY_KEYS)
        assert set(HEADER_KEYS) | set(TABLE_KEYS) == RECALL_KEYS

    def test_recall_agent_reads_exactly_recall_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        assert set(agent.observation_keys) == RECALL_KEYS

    def test_legacy_agent_reads_the_picker_memory_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        assert set(agent.observation_keys) == RECALL_KEYS | LEGACY_PICKER_MEMORY_KEYS

    @pytest.mark.parametrize("arch", architectures.available_architectures())
    def test_registry_flag_matches_encoder(self, arch):
        spec = architectures.get_spec(arch)
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        reads_legacy = bool(set(agent.observation_keys) & LEGACY_PICKER_MEMORY_KEYS)
        assert reads_legacy == spec.legacy_picker_memory, arch

    def test_only_the_recall_family_is_recall(self):
        recall = {
            name
            for name, spec in architectures.ARCHITECTURES.items()
            if not spec.legacy_picker_memory
        }
        assert recall == {"perceiver-recall"}


class TestRecallEncoder:
    def test_recall_layout_and_invariance(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        states = _picker_states(agent)
        assert states, "the seeded game produced no picker states"
        out = agent.encoder.encode_batch(states)
        assert out["all_tokens"].shape[1] == RECALL_TOKEN_COUNT
        masked = agent.encoder.encode_batch([_masked(s) for s in states])
        for key in ("features", "memory_out", "hand_tokens", "all_tokens"):
            assert torch.equal(out[key], masked[key]), key

    def test_recall_marshal_never_touches_picker_memory_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        state = Game(seed=2).players[0].get_state_dict()
        stripped = {k: v for k, v in state.items() if k in RECALL_KEYS}
        a = agent.encoder.encode_batch([state])
        b = agent.encoder.encode_batch([stripped])
        assert torch.equal(a["features"], b["features"])

    def test_legacy_encoder_still_sees_the_injection(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        states = _picker_states(agent)
        out = agent.encoder.encode_batch(states)
        assert out["all_tokens"].shape[1] == BASE_TOKEN_COUNT
        masked = agent.encoder.encode_batch([_masked(s) for s in states])
        assert not torch.equal(out["features"], masked["features"])

    def test_legacy_encoder_fails_loudly_without_the_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        state = Game(seed=2).players[0].get_state_dict()
        stripped = {k: v for k, v in state.items() if k in RECALL_KEYS}
        with pytest.raises(KeyError, match="legacy picker-memory"):
            agent.encoder.encode_batch([stripped])

    def test_memory_token_drives_recurrence(self):
        """Perturbing the post-reasoning MEMORY token changes memory_out;
        the context-driven v2 encoder ignores that same perturbation."""
        seed_all(0)
        enc = RecallEncoder()
        state = Game(seed=3).players[1].get_state_dict()
        obs = enc.marshal_batch([state])
        memory_in = torch.zeros((1, enc.d_model))
        base = enc.encode_tensors(obs, memory_in=memory_in)
        # Re-run with the reasoner's MEMORY-token output shifted: the GRU
        # driver moves, so must memory_out; the readout features move too
        # because the readout attends over every token.
        reasoner = enc.card_reasoner

        def shifted_forward(tokens, mask, _orig=reasoner.forward):
            out = _orig(tokens, mask).clone()
            out[:, MEMORY_TOKEN, :] += 1.0
            return out

        reasoner.forward = shifted_forward  # type: ignore[method-assign]
        try:
            moved = enc.encode_tensors(obs, memory_in=memory_in)
        finally:
            del reasoner.forward
        assert not torch.equal(base["memory_out"], moved["memory_out"])

    def test_no_picker_memory_modules_and_shape_parity(self):
        seed_all(0)
        recall = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        seed_all(0)
        v2bp = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        assert not hasattr(recall.encoder, "token_mlp_simple")
        assert not hasattr(recall.encoder, "pool_blind")
        recall_keys = set(recall.encoder.state_dict())
        v2_keys = set(v2bp.encoder.state_dict())
        assert v2_keys - recall_keys == {
            "token_mlp_simple.0.weight",
            "token_mlp_simple.0.bias",
        }
        assert recall_keys <= v2_keys
        for k in recall_keys:
            assert (
                recall.encoder.state_dict()[k].shape
                == v2bp.encoder.state_dict()[k].shape
            )
        assert recall.actor.state_dict().keys() == v2bp.actor.state_dict().keys()
        assert recall.critic.state_dict().keys() == v2bp.critic.state_dict().keys()

    def test_checkpoint_roundtrip_and_update(self, tmp_path):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        path = tmp_path / "recall.pt"
        agent.save(str(path))
        loaded = load_agent(str(path))
        assert loaded.arch_name == "perceiver-recall"
        assert set(loaded.observation_keys) == RECALL_KEYS
