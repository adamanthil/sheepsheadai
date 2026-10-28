#!/usr/bin/env python3
"""Structural golden and consistency guards for the architecture registry.

The committed fixture (sheepshead/tests/fixtures/arch_structure.json, written
by ``capture_arch_goldens --write-structure``) pins each architecture's
state_dict key names, parameter counts and aux-head flags, which hold on
every platform. Weight and forward-output bit identity is platform-dependent
and is checked commit-against-commit by ``sheepshead.devtools.check_goldens``
(CI runs it on every push), not here.
"""

import os
import tempfile

import pytest
import torch

from sheepshead.agent.architectures.registry import (
    ARCHITECTURES,
    available_architectures,
)
from sheepshead.agent.observation import observation_for
from sheepshead.agent.ppo import PPOAgent
from sheepshead.analysis.capture_arch_goldens import (
    build_agent,
    load_structure,
    structure_record,
)
from sheepshead.game import ACTIONS, Game

_STRUCTURE = load_structure()
_REWRITE = "uv run python -m sheepshead.analysis.capture_arch_goldens --write-structure"


class TestRegistryConsistency:
    """Welds the redundant identity fields together: dict key == spec.name,
    spec.has_aux_heads == built critic.has_aux_heads == aux-module presence
    in the critic state_dict, and the encoder attribute contract."""

    @pytest.mark.parametrize("key", list(ARCHITECTURES.keys()))
    def test_every_entry(self, key):
        spec = ARCHITECTURES[key]
        assert key == spec.name
        agent = build_agent(key)
        assert agent.critic.has_aux_heads == spec.has_aux_heads
        has_aux_keys = any(
            k.startswith(("win_head", "trump_aux", "points_head"))
            for k in agent.critic.state_dict()
        )
        assert has_aux_keys == spec.has_aux_heads
        for attr in ("d_model", "d_card_dim", "d_token_dim", "card"):
            assert hasattr(agent.encoder, attr), attr
        assert callable(agent.encoder.param_groups)


class TestStructuralGoldens:
    """Every registered architecture's structure matches the committed
    fixture, and the fixture covers exactly the registry."""

    def test_fixture_covers_registry(self):
        assert sorted(_STRUCTURE) == sorted(available_architectures()), (
            f"registry and arch_structure.json disagree; rewrite with: {_REWRITE}"
        )

    @pytest.mark.parametrize("arch", available_architectures())
    def test_structure_matches(self, arch):
        golden = _STRUCTURE.get(arch)
        assert golden is not None, f"{arch} not in the fixture; run: {_REWRITE}"
        current = structure_record(build_agent(arch))
        for net, keys in golden["keys"].items():
            gk, ck = set(keys), set(current["keys"][net])
            assert gk == ck, (
                f"{arch}/{net}: state_dict keys drifted "
                f"(missing={sorted(gk - ck)} added={sorted(ck - gk)})"
            )
        assert current["meta"] == golden["meta"]


class TestLegacyValueTrunkShim:
    """Characterizes the pre-value_trunk checkpoint compatibility shim in
    PPOAgent.load (critic strict=False + value_trunk -> critic_adapter
    aliasing) so refactors of the load path cannot silently change it."""

    def test_legacy_checkpoint_aliases_value_trunk(self):
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "legacy.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            ckpt["critic_state_dict"] = {
                k: v
                for k, v in ckpt["critic_state_dict"].items()
                if not k.startswith("value_trunk")
            }
            torch.save(ckpt, path)

            loaded = PPOAgent(len(ACTIONS))
            loaded.load(path)
            assert loaded.critic.value_trunk is loaded.critic.critic_adapter

            # The value path must route through the trained adapter exactly.
            game = Game(seed=126)
            enc_out = loaded.encoder.encode_batch(
                [observation_for(game.players[0], loaded)]
            )
            with torch.no_grad():
                value = loaded.critic(enc_out)
                expected = loaded.critic.value_head(
                    loaded.critic.critic_adapter(enc_out["features"])
                )
            assert torch.equal(value, expected)

    def test_modern_checkpoint_keeps_trunk_distinct(self):
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "modern.pt")
            agent.save(path)
            loaded = PPOAgent(len(ACTIONS))
            loaded.load(path)
            assert loaded.critic.value_trunk is not loaded.critic.critic_adapter


class TestCheckpointPickleSafety:
    """Checkpoints must contain only state_dicts and plain metadata — no
    pickled class paths — so classes can move between modules freely."""

    def test_fresh_save_loads_weights_only(self):
        agent = PPOAgent(len(ACTIONS), arch="no-aux")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            assert ckpt["arch"] == "no-aux"

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "models/pfsp-swish-30m.pt",
            )
        ),
        reason="legacy checkpoint not present",
    )
    def test_legacy_30m_checkpoint_loads_weights_only(self):
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        ckpt = torch.load(
            os.path.join(root, "models/pfsp-swish-30m.pt"),
            map_location="cpu",
            weights_only=True,
        )
        assert "encoder_state_dict" in ckpt


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
