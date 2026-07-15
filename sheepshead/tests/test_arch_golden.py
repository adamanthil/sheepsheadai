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
import unittest

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


class TestRegistryConsistency(unittest.TestCase):
    """Welds the redundant identity fields together: dict key == spec.name,
    spec.has_aux_heads == built critic.has_aux_heads == aux-module presence
    in the critic state_dict, and the encoder attribute contract."""

    def test_every_entry(self):
        for key, spec in architectures.ARCHITECTURES.items():
            with self.subTest(arch=key):
                self.assertEqual(key, spec.name)
                agent = build_agent(key)
                self.assertEqual(agent.critic.has_aux_heads, spec.has_aux_heads)
                has_aux_keys = any(
                    k.startswith(("win_head", "trump_aux", "points_head"))
                    for k in agent.critic.state_dict()
                )
                self.assertEqual(has_aux_keys, spec.has_aux_heads)
                for attr in ("d_model", "d_card_dim", "d_token_dim", "card"):
                    self.assertTrue(hasattr(agent.encoder, attr), attr)
                self.assertTrue(callable(agent.encoder.param_groups))

                if _HAVE_FIXTURES:
                    golden = _golden(key)
                    for net_name, net in (
                        ("encoder", agent.encoder),
                        ("actor", agent.actor),
                        ("critic", agent.critic),
                    ):
                        self.assertEqual(
                            _key_sha(net),
                            golden["key_sha"][net_name],
                            f"{key}/{net_name}: state_dict key names drifted "
                            "from the golden fixture",
                        )


@unittest.skipUnless(_HAVE_FIXTURES, "no golden fixtures captured")
class TestNumericalGoldens(unittest.TestCase):
    """Full bit-identity check (weights + forward outputs) against the
    fixtures; only meaningful on the environment that captured them."""

    @classmethod
    def setUpClass(cls):
        manifest = load_manifest()
        if not runtime_matches_manifest(manifest):
            raise unittest.SkipTest(
                "runtime differs from fixture manifest "
                f"(torch {manifest['torch']}, {manifest['platform']}); "
                "re-capture with sheepshead/analysis/capture_arch_goldens.py to gate "
                "refactors on this machine"
            )

    def test_all_archs_bit_identical(self):
        torch.set_num_threads(1)
        for arch in architectures.available_architectures():
            with self.subTest(arch=arch):
                problems = check_arch(arch)
                self.assertEqual(problems, [], f"{arch}: {problems}")


class TestLegacyValueTrunkShim(unittest.TestCase):
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
            self.assertIs(loaded.critic.value_trunk, loaded.critic.critic_adapter)

            # The value path must route through the trained adapter exactly.
            game = Game(seed=126)
            enc_out = loaded.encoder.encode_batch([game.players[0].get_state_dict()])
            with torch.no_grad():
                value = loaded.critic(enc_out)
                expected = loaded.critic.value_head(
                    loaded.critic.critic_adapter(enc_out["features"])
                )
            self.assertTrue(torch.equal(value, expected))

    def test_modern_checkpoint_keeps_trunk_distinct(self):
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "modern.pt")
            agent.save(path)
            loaded = PPOAgent(len(ACTIONS))
            loaded.load(path)
            self.assertIsNot(loaded.critic.value_trunk, loaded.critic.critic_adapter)


class TestCheckpointPickleSafety(unittest.TestCase):
    """Checkpoints must contain only state_dicts and plain metadata — no
    pickled class paths — so classes can move between modules freely."""

    def test_fresh_save_loads_weights_only(self):
        agent = PPOAgent(len(ACTIONS), arch="no-aux")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(ckpt["arch"], "no-aux")

    @unittest.skipUnless(
        os.path.exists(
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "final_pfsp_swish_ppo.pt",
            )
        ),
        "legacy checkpoint not present",
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
        self.assertIn("encoder_state_dict", ckpt)


if __name__ == "__main__":
    unittest.main()
