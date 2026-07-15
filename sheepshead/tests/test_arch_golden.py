#!/usr/bin/env python3
"""Golden-fixture and consistency guards for the architecture registry.

Fixtures live in sheepshead/tests/fixtures/arch_golden/ and are captured by
sheepshead/analysis/capture_arch_goldens.py at a known-good commit. Structural checks
(state_dict key names) run everywhere; weight/numerical bit-identity checks
run only on the environment recorded in the fixture manifest (orthogonal
init and kernels are LAPACK/BLAS dependent).
"""

import os
import tempfile

import pytest
import torch

from sheepshead.agent import architectures
from sheepshead.analysis.capture_arch_goldens import (
    FIXTURE_DIR,
    build_agent,
    check_arch,
    load_manifest,
    manifest_path,
    runtime_matches_manifest,
    _key_sha,
)
from sheepshead import ACTIONS, Game
from sheepshead.agent.ppo import PPOAgent

_HAVE_FIXTURES = os.path.exists(manifest_path())


def _golden(arch: str) -> dict:
    return torch.load(os.path.join(FIXTURE_DIR, f"{arch}.pt"), weights_only=True)


class TestRegistryConsistency:
    """Welds the redundant identity fields together: dict key == spec.name,
    spec.has_aux_heads == built critic.has_aux_heads == aux-module presence
    in the critic state_dict, and the encoder attribute contract."""

    @pytest.mark.parametrize("key", list(architectures.ARCHITECTURES.keys()))
    def test_every_entry(self, key):
        spec = architectures.ARCHITECTURES[key]
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

        if _HAVE_FIXTURES:
            golden = _golden(key)
            for net_name, net in (
                ("encoder", agent.encoder),
                ("actor", agent.actor),
                ("critic", agent.critic),
            ):
                assert _key_sha(net) == golden["key_sha"][net_name], (
                    f"{key}/{net_name}: state_dict key names drifted "
                    "from the golden fixture"
                )


@pytest.mark.skipif(not _HAVE_FIXTURES, reason="no golden fixtures captured")
class TestNumericalGoldens:
    """Full bit-identity check (weights + forward outputs) against the
    fixtures; only meaningful on the environment that captured them."""

    @classmethod
    def setup_class(cls):
        manifest = load_manifest()
        if not runtime_matches_manifest(manifest):
            pytest.skip(
                "runtime differs from fixture manifest "
                f"(torch {manifest['torch']}, {manifest['platform']}); "
                "re-capture with sheepshead/analysis/capture_arch_goldens.py to gate "
                "refactors on this machine"
            )

    @pytest.mark.parametrize("arch", architectures.available_architectures())
    def test_all_archs_bit_identical(self, arch):
        torch.set_num_threads(1)
        problems = check_arch(arch)
        assert problems == [], f"{arch}: {problems}"


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
            enc_out = loaded.encoder.encode_batch([game.players[0].get_state_dict()])
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
                "final_pfsp_swish_ppo.pt",
            )
        ),
        reason="legacy checkpoint not present",
    )
    def test_legacy_30m_checkpoint_loads_weights_only(self):
        root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        ckpt = torch.load(
            os.path.join(root, "final_pfsp_swish_ppo.pt"),
            map_location="cpu",
            weights_only=True,
        )
        assert "encoder_state_dict" in ckpt


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
