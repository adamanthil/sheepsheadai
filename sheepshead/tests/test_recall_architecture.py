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

from sheepshead.agent.architectures.encoders import RecallEncoder
from sheepshead.agent.architectures.onehot import (
    ONEHOT_RECALL_STATE_DIM,
    ONEHOT_STATE_DIM,
    build_onehot_state,
)
from sheepshead.agent.architectures.registry import (
    ARCHITECTURES,
    available_architectures,
    get_spec,
)
from sheepshead.agent.convention_wrapper import ConventionWrapper
from sheepshead.agent.observation import (
    HEADER_KEYS,
    LEGACY_PICKER_MEMORY_KEYS,
    RECALL_KEYS,
    TABLE_KEYS,
    needs_picker_memory,
    observation_for,
)
from sheepshead.agent.ppo import (
    DETERMINISTIC_AUX_BASE_COEFFS,
    PPOAgent,
    load_agent,
)
from sheepshead.agent.token_layout import (
    BASE_TOKEN_COUNT,
    MEMORY_TOKEN,
    RECALL_TOKEN_COUNT,
)
from sheepshead.game import ACTIONS, Game
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.train_ppo import build_arg_parser

# The recall-contract ladder (Architecture_Ablation_202610), bottom to top.
RECALL_LADDER = (
    "onehot-recall",
    "pooled-recall-uninformed-noaux",
    "pooled-recall-noaux",
    "full-recall-noaux",
    "full-recall",
    "perceiver-recall-nobilinear",
    "perceiver-recall",
)


def _picker_states(agent: PPOAgent, seed: int = 5) -> list:
    """Observation dicts (with the picker-memory interface merged in) of a
    played game where the actor is the picker holding a non-empty blind/bury."""
    game = Game(seed=seed)
    states = []
    while not game.is_done():
        for player in game.players:
            acts = player.get_valid_action_ids()
            while acts:
                state = observation_for(player, agent)
                memory = player.get_picker_memory()
                if memory["blind_ids"].any() or memory["bury_ids"].any():
                    states.append({**state, **memory})
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
    def test_key_sets_are_the_two_interfaces(self):
        player = Game(seed=1).players[0]
        assert set(player.get_state_dict()) == RECALL_KEYS
        assert set(player.get_picker_memory()) == LEGACY_PICKER_MEMORY_KEYS
        assert not (RECALL_KEYS & LEGACY_PICKER_MEMORY_KEYS)
        assert set(HEADER_KEYS) | set(TABLE_KEYS) == RECALL_KEYS

    def test_observation_for_merges_only_for_legacy_agents(self):
        seed_all(0)
        recall = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        legacy = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        player = Game(seed=1).players[0]
        assert recall.needs_picker_memory is False
        assert legacy.needs_picker_memory is True
        assert set(observation_for(player, recall)) == RECALL_KEYS
        assert (
            set(observation_for(player, legacy))
            == RECALL_KEYS | LEGACY_PICKER_MEMORY_KEYS
        )
        # Scripted agents and anything without the attribute get the clean
        # observation; a wrapper delegates to what it wraps.
        assert needs_picker_memory(ScriptedAgent()) is False
        assert set(observation_for(player, ScriptedAgent())) == RECALL_KEYS
        assert needs_picker_memory(ConventionWrapper(legacy)) is True
        assert needs_picker_memory(ConventionWrapper(recall)) is False

    def test_needs_picker_memory_never_masks_a_broken_property(self):
        class Broken:
            @property
            def needs_picker_memory(self):
                raise AttributeError("arch attribute missing")

        class Plain:
            pass

        with pytest.raises(AttributeError, match="arch attribute missing"):
            needs_picker_memory(Broken())
        assert needs_picker_memory(Plain()) is False

    def test_onehot_encoder_fails_loudly_on_the_clean_observation(self):
        state = Game(seed=1).players[0].get_state_dict()
        with pytest.raises(KeyError, match="observation_for"):
            build_onehot_state(state)

    def test_recall_agent_reads_exactly_recall_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        assert set(agent.observation_keys) == RECALL_KEYS

    def test_legacy_agent_reads_the_picker_memory_keys(self):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2-bp")
        assert set(agent.observation_keys) == RECALL_KEYS | LEGACY_PICKER_MEMORY_KEYS

    @pytest.mark.parametrize("arch", available_architectures())
    def test_registry_flag_matches_encoder(self, arch):
        spec = get_spec(arch)
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        reads_legacy = bool(set(agent.observation_keys) & LEGACY_PICKER_MEMORY_KEYS)
        assert reads_legacy == spec.legacy_picker_memory, arch

    def test_only_the_recall_family_is_recall(self):
        recall = {
            name
            for name, spec in ARCHITECTURES.items()
            if not spec.legacy_picker_memory
        }
        assert recall == set(RECALL_LADDER)


class TestRecallLadder:
    @pytest.mark.parametrize("arch", RECALL_LADDER)
    def test_reads_exactly_recall_keys_and_ignores_picker_memory(self, arch):
        seed_all(0)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        assert set(agent.observation_keys) == RECALL_KEYS
        states = _picker_states(agent)
        assert states, "the seeded game produced no picker states"
        out = agent.encoder.encode_batch(states)
        masked = agent.encoder.encode_batch([_masked(s) for s in states])
        for key in ("features", "memory_out"):
            assert torch.equal(out[key], masked[key]), key
        clean = Game(seed=2).players[0].get_state_dict()
        agent.encoder.encode_batch([clean])  # no KeyError without the keys

    def test_onehot_recall_vector_is_legacy_minus_the_memory_sections(self):
        seed_all(0)
        legacy_agent = PPOAgent(len(ACTIONS), arch="onehot-ff")
        state = _picker_states(legacy_agent)[0]
        legacy = build_onehot_state(state)
        recall = build_onehot_state(state, observe_picker_memory=False)
        assert recall.shape == (ONEHOT_RECALL_STATE_DIM,)
        assert ONEHOT_STATE_DIM - ONEHOT_RECALL_STATE_DIM == 2 * 34
        # Sections before the blind/bury block (hand, trick cards, two flag
        # rows) and after it keep their values and order.
        cut = 34 + 5 * 34 + 5 + 5
        assert legacy[cut : cut + 68].any(), "picker state carries no memory"
        assert np.array_equal(recall[:cut], legacy[:cut])
        assert np.array_equal(recall[cut:], legacy[cut + 68 :])

    def test_uninformed_rung_differs_only_in_card_init(self):
        seed_all(0)
        uninformed = PPOAgent(len(ACTIONS), arch="pooled-recall-uninformed-noaux")
        seed_all(0)
        informed = PPOAgent(len(ACTIONS), arch="pooled-recall-noaux")
        a, b = uninformed.encoder.state_dict(), informed.encoder.state_dict()
        assert a.keys() == b.keys()
        assert not torch.equal(a["card.weight"], b["card.weight"])

    def test_nobilinear_rung_is_recall_without_the_bilinear_term(self):
        seed_all(0)
        plain = PPOAgent(len(ACTIONS), arch="perceiver-recall-nobilinear")
        seed_all(0)
        recall = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        assert plain.encoder.state_dict().keys() == recall.encoder.state_dict().keys()
        extra = set(recall.actor.state_dict()) - set(plain.actor.state_dict())
        assert extra == {"pointer_U.weight", "pointer_V.weight"}
        assert set(plain.actor.state_dict()) <= set(recall.actor.state_dict())


class TestDeterministicAuxScale:
    def test_default_is_the_baked_2_5x(self):
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        for name, base in DETERMINISTIC_AUX_BASE_COEFFS.items():
            assert getattr(agent, name) == pytest.approx(2.5 * base), name

    def test_scale_one_restores_the_original_coefficients(self):
        agent = PPOAgent(len(ACTIONS), arch="perceiver-recall")
        agent.set_deterministic_aux_scale(1.0)
        assert agent.secret_loss_coeff == pytest.approx(0.1)
        assert agent.points_loss_coeff == pytest.approx(0.2)
        assert agent.seen_trump_mask_loss_coeff == pytest.approx(0.2)
        assert agent.unseen_trump_higher_than_hand_loss_coeff == pytest.approx(0.1)
        assert agent.win_loss_coeff == pytest.approx(0.05)
        assert agent.return_loss_coeff == pytest.approx(0.1)

    def test_trainer_flag_defaults_to_the_baked_scale(self):
        base = ["--phase", "bootstrap", "--run-name", "x", "--until", "1"]
        assert build_arg_parser().parse_args(base).aux_det_scale == 2.5
        args = build_arg_parser().parse_args(base + ["--aux-det-scale", "1.0"])
        assert args.aux_det_scale == 1.0


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
        player = Game(seed=2).players[0]
        clean = observation_for(player, agent)
        assert set(clean) == RECALL_KEYS
        with_memory = {**clean, **player.get_picker_memory()}
        a = agent.encoder.encode_batch([clean])
        b = agent.encoder.encode_batch([with_memory])
        assert torch.equal(a["features"], b["features"])
        assert (
            set(agent.encoder.marshal_batch([clean])) & LEGACY_PICKER_MEMORY_KEYS
            == set()
        )

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
        clean = Game(seed=2).players[0].get_state_dict()
        with pytest.raises(KeyError, match="legacy picker-memory"):
            agent.encoder.encode_batch([clean])

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
