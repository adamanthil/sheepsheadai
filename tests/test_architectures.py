#!/usr/bin/env python3
"""Tests for the architecture registry (architectures.py).

Covers: every registered arch builds/plays/updates; the default "full" arch
is construction-identical to building the networks directly (the registry
adds no RNG activity — the guard behind bit-identical seeded training);
checkpoint arch metadata (record / mismatch / legacy default / load_agent);
the no-aux critic; the no-transformer encoder; and the one-hot state vector.
"""

import os
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import architectures
import pfsp_runtime
import ppo
from architectures import (
    ONEHOT_STATE_DIM,
    OneHotFeedForwardEncoder,
    PooledMemoryEncoder,
    build_onehot_state,
)
from encoder import CardEmbeddingConfig, CardReasoningEncoder
from ppo import MultiHeadRecurrentActorNetwork, PPOAgent, RecurrentCriticNetwork
from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

# Frozen sha256 over the sorted encoder/actor/critic state_dict KEY NAMES of
# the full architecture (recorded pre-refactor at commit 825b37c). Weights are
# platform/torch-version sensitive; key names are not — any drift here means
# the default architecture's module structure changed.
FULL_ARCH_KEYS_SHA256 = (
    "39c54bccfdce10abcfe65abdff0cbcd8503f9263751f15194c440f9c78eb2743"
)


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _play_episodes(agent: PPOAgent, n: int, seed0: int = 900_000) -> None:
    """Self-play n seeded episodes and store their events on the agent."""
    opponents = [
        SimpleNamespace(agent=agent, metadata=SimpleNamespace(agent_id="self"))
        for _ in range(4)
    ]
    orig_game = pfsp_runtime.Game
    counter = {"n": 0}

    class _SeededGame(orig_game):
        def __init__(self, *a, **kw):
            counter["n"] += 1
            kw.setdefault("seed", seed0 + counter["n"])
            super().__init__(*a, **kw)

    pfsp_runtime.Game = _SeededGame
    try:
        for ep in range(n):
            mode = PARTNER_BY_CALLED_ACE if ep % 2 == 0 else PARTNER_BY_JD
            _, events, _, _, _ = pfsp_runtime.play_population_game(
                training_agent=agent,
                opponents=opponents,
                partner_mode=mode,
                training_agent_position=(ep % 5) + 1,
                reward_mode="terminal",
            )
            agent.store_episode_events(events)
    finally:
        pfsp_runtime.Game = orig_game


class TestRegistry(unittest.TestCase):
    def test_available_architectures(self):
        names = architectures.available_architectures()
        for expected in (
            "full",
            "full-uninformed",
            "no-aux",
            "no-transformer",
            "no-transformer-uninformed",
            "onehot-ff",
        ):
            self.assertIn(expected, names)

    def test_unknown_arch_raises_with_names(self):
        with self.assertRaises(KeyError) as ctx:
            architectures.get_spec("bogus")
        self.assertIn("full", str(ctx.exception))

    def test_all_archs_build_play_update(self):
        for arch in architectures.available_architectures():
            with self.subTest(arch=arch):
                _seed_all(11)
                agent = PPOAgent(len(ACTIONS), arch=arch)
                self.assertEqual(agent.arch_name, arch)
                _play_episodes(agent, 4)
                stats = agent.update(epochs=1, batch_size=16)
                self.assertGreater(stats["timing"]["optimizer_steps"], 0)
                self.assertTrue(np.isfinite(stats["approx_kl"]))


class TestFullArchUnchanged(unittest.TestCase):
    """The registry path must construct exactly what direct construction
    builds — same classes, same order, no extra RNG draws in between."""

    def test_registry_equals_direct_construction(self):
        _seed_all(42)
        via_registry = PPOAgent(len(ACTIONS))

        _seed_all(42)
        encoder = CardReasoningEncoder(card_config=CardEmbeddingConfig()).to(ppo.device)
        # Reuse the agent's own mapping builder (pure, no RNG)
        mappings = via_registry._build_action_index_mappings()
        actor = MultiHeadRecurrentActorNetwork(
            len(ACTIONS),
            via_registry.action_groups,
            d_card=encoder.d_card_dim,
            d_token=encoder.d_token_dim,
            map_cid_to_play_action_index=mappings[0],
            map_cid_to_bury_action_index=mappings[1],
            map_cid_to_under_action_index=mappings[2],
            call_action_global_indices=mappings[3],
            call_card_ids=mappings[4],
            play_under_action_index=mappings[5],
        ).to(ppo.device)
        critic = RecurrentCriticNetwork(d_card=encoder.d_card_dim).to(ppo.device)

        for reg_net, direct_net in (
            (via_registry.encoder, encoder),
            (via_registry.actor, actor),
            (via_registry.critic, critic),
        ):
            reg_sd = reg_net.state_dict()
            dir_sd = direct_net.state_dict()
            self.assertEqual(sorted(reg_sd), sorted(dir_sd))
            for k in reg_sd:
                self.assertTrue(
                    torch.equal(reg_sd[k], dir_sd[k]), f"weights differ: {k}"
                )

    def test_full_arch_state_dict_keys_frozen(self):
        import hashlib

        agent = PPOAgent(len(ACTIONS))
        h = hashlib.sha256()
        for net in (agent.encoder, agent.actor, agent.critic):
            for k in sorted(net.state_dict().keys()):
                h.update(k.encode())
        self.assertEqual(h.hexdigest(), FULL_ARCH_KEYS_SHA256)


class TestCheckpointArchMetadata(unittest.TestCase):
    def _roundtrip(self, arch: str):
        _seed_all(5)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            self.assertEqual(ckpt["arch"], arch)
            loaded = ppo.load_agent(path)
            self.assertEqual(loaded.arch_name, arch)
            for k, v in agent.encoder.state_dict().items():
                self.assertTrue(torch.equal(v, loaded.encoder.state_dict()[k]))

    def test_roundtrip_every_arch(self):
        for arch in architectures.available_architectures():
            with self.subTest(arch=arch):
                self._roundtrip(arch)

    def test_arch_mismatch_fails_loudly(self):
        agent = PPOAgent(len(ACTIONS), arch="no-aux")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            full_agent = PPOAgent(len(ACTIONS))
            with self.assertRaises(ValueError) as ctx:
                full_agent.load(path)
            self.assertIn("no-aux", str(ctx.exception))

    def test_legacy_checkpoint_defaults_to_full(self):
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            del ckpt["arch"]  # simulate a pre-registry checkpoint
            torch.save(ckpt, path)
            # Loads fine into a full agent (and via load_agent)...
            PPOAgent(len(ACTIONS)).load(path)
            self.assertEqual(ppo.load_agent(path).arch_name, "full")
            # ...but not into a different arch.
            with self.assertRaises(ValueError):
                PPOAgent(len(ACTIONS), arch="no-transformer").load(path)


class TestNoAuxCritic(unittest.TestCase):
    def test_aux_modules_absent_and_accessors_raise(self):
        critic = RecurrentCriticNetwork(use_aux_heads=False)
        self.assertFalse(critic.has_aux_heads)
        for attr in (
            "critic_adapter",
            "win_head",
            "return_head",
            "secret_partner_head",
            "points_head",
            "trump_aux",
            "seen_trump_query",
            "unseen_trump_higher_than_hand_head",
        ):
            self.assertFalse(hasattr(critic, attr), attr)
        feat = torch.zeros(1, 256)
        with self.assertRaises(RuntimeError):
            critic.aux_predictions({"features": feat})
        with self.assertRaises(RuntimeError):
            critic.seen_trump_mask_logits(feat, None)
        with self.assertRaises(RuntimeError):
            critic.unseen_trump_higher_than_hand_logits(feat)

    def test_aux_critic_unchanged_when_enabled(self):
        torch.manual_seed(3)
        with_flag = RecurrentCriticNetwork(d_card=16, use_aux_heads=True)
        torch.manual_seed(3)
        default = RecurrentCriticNetwork(d_card=16)
        for k, v in default.state_dict().items():
            self.assertTrue(torch.equal(v, with_flag.state_dict()[k]))

    def test_no_aux_agent_fewer_params_same_value_path(self):
        _seed_all(9)
        full = PPOAgent(len(ACTIONS))
        _seed_all(9)
        lean = PPOAgent(len(ACTIONS), arch="no-aux")
        n_full = sum(p.numel() for p in full.critic.parameters())
        n_lean = sum(p.numel() for p in lean.critic.parameters())
        self.assertLess(n_lean, n_full)
        # Same value-trunk shape: the value path is architecturally identical.
        self.assertEqual(
            {k for k in lean.critic.state_dict()},
            {
                k
                for k in full.critic.state_dict()
                if k.startswith(("value_trunk", "value_head"))
            },
        )


class TestNoTransformer(unittest.TestCase):
    def test_reasoner_is_parameterless_noop(self):
        enc = PooledMemoryEncoder()
        self.assertEqual(sum(p.numel() for p in enc.card_reasoner.parameters()), 0)
        tokens = torch.randn(2, 19, enc.d_token_dim)
        mask = torch.ones(2, 19, dtype=torch.bool)
        out = enc.card_reasoner(tokens, mask)
        self.assertTrue(torch.equal(out, tokens))

    def test_encode_batch_contract(self):
        enc = PooledMemoryEncoder()
        game = Game(seed=123)
        out = enc.encode_batch([game.players[0].get_state_dict()])
        self.assertEqual(tuple(out["features"].shape), (1, 256))
        self.assertEqual(tuple(out["hand_tokens"].shape), (1, 8, enc.d_token_dim))

    def test_memory_feeds_features(self):
        # The whole point of the pooled-memory design: without attention the
        # base encoder's memory is write-only, so this variant must route the
        # recurrent state into the features the heads consume.
        enc = PooledMemoryEncoder()
        game = Game(seed=124)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        self.assertFalse(torch.equal(out1["features"], out2["features"]))
        # Heads consume the recurrent state itself (pre-transformer LSTM shape).
        self.assertTrue(torch.equal(out1["features"], out1["memory_out"]))

    def test_base_encoder_memory_is_write_only_without_attention(self):
        # Documents why PooledMemoryEncoder exists: plain n_reasoning_layers=0
        # leaves features independent of memory.
        enc = CardReasoningEncoder(
            card_config=CardEmbeddingConfig(), n_reasoning_layers=0
        )
        game = Game(seed=125)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=torch.randn(1, 256))
        self.assertTrue(torch.equal(out1["features"], out2["features"]))

    def test_uninformed_init_differs(self):
        torch.manual_seed(4)
        informed = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        torch.manual_seed(4)
        uninformed = CardReasoningEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        )
        self.assertFalse(torch.equal(informed.card.weight, uninformed.card.weight))


class TestTokenRead(unittest.TestCase):
    def test_encoder_adds_tokens_without_changing_anything_else(self):
        # Same seed -> same params (TokenReadEncoder adds none) -> the
        # standard outputs must be byte-identical to the base encoder.
        torch.manual_seed(7)
        base = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        torch.manual_seed(7)
        enc = architectures.TokenReadEncoder(card_config=CardEmbeddingConfig())
        game = Game(seed=126)
        s = game.players[0].get_state_dict()
        out_b, out_t = base.encode_batch([s]), enc.encode_batch([s])
        self.assertTrue(torch.equal(out_b["features"], out_t["features"]))
        self.assertTrue(torch.equal(out_b["memory_out"], out_t["memory_out"]))
        self.assertNotIn("all_tokens", out_b)
        self.assertEqual(tuple(out_t["all_tokens"].shape), (1, 19, enc.d_token_dim))
        self.assertEqual(tuple(out_t["all_mask"].shape), (1, 19))
        # Context + memory tokens always valid (readout needs >= 1 key).
        self.assertTrue(bool(out_t["all_mask"][:, :2].all()))

    def test_encode_sequences_carries_tokens(self):
        enc = architectures.TokenReadEncoder(card_config=CardEmbeddingConfig())
        game = Game(seed=127)
        seqs = [
            [game.players[0].get_state_dict(), game.players[0].get_state_dict()],
            [game.players[1].get_state_dict()],
        ]
        out = enc.encode_sequences(seqs)
        self.assertEqual(tuple(out["all_tokens"].shape), (2, 2, 19, enc.d_token_dim))
        self.assertEqual(tuple(out["all_mask"].shape), (2, 2, 19))
        self.assertEqual(out["all_mask"].dtype, torch.bool)

    def test_readout_reaches_the_logits(self):
        # Perturbing a token the pools would summarize must change the
        # logits through the readout path alone.
        _seed_all(8)
        agent = PPOAgent(len(ACTIONS), arch="full-tokenread")
        game = Game(seed=128)
        s = game.players[0].get_state_dict()
        enc_out = agent.encoder.encode_batch([s])
        mask = torch.ones(1, len(ACTIONS), dtype=torch.bool)
        hand_ids = torch.as_tensor(s["hand_ids"], dtype=torch.long).view(1, -1)
        with torch.no_grad():
            _, logits1 = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
            enc_out["all_tokens"] = enc_out["all_tokens"].clone()
            # Perturb the post-reasoning MEMORY token (index 1, always
            # valid; trick/blind/bury tokens are masked pre-play). The base
            # architecture discards this token entirely, so a logit change
            # proves the readout's unmediated path.
            enc_out["all_tokens"][:, 1, :] += 1.0
            _, logits2 = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
        self.assertFalse(torch.equal(logits1, logits2))

    def test_actor_requires_tokens(self):
        _seed_all(9)
        agent = PPOAgent(len(ACTIONS), arch="full-tokenread")
        with self.assertRaises(RuntimeError):
            agent.actor._adapt_features(torch.randn(1, 256))

    def test_base_actor_ignores_token_kwargs(self):
        # The threaded-through kwargs must be inert for the default arch.
        _seed_all(10)
        agent = PPOAgent(len(ACTIONS), arch="full")
        game = Game(seed=129)
        s = game.players[0].get_state_dict()
        enc_out = agent.encoder.encode_batch([s])
        self.assertNotIn("all_tokens", enc_out)
        mask = torch.ones(1, len(ACTIONS), dtype=torch.bool)
        hand_ids = torch.as_tensor(s["hand_ids"], dtype=torch.long).view(1, -1)
        with torch.no_grad():
            probs, _ = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
        self.assertTrue(torch.isfinite(probs).all())


class TestPerceiver(unittest.TestCase):
    def test_pools_and_trunk_gone(self):
        enc = architectures.PerceiverEncoder()
        for name in (
            "pool_hand",
            "pool_trick",
            "pool_blind",
            "pool_bury",
            "feature_proj",
        ):
            self.assertFalse(hasattr(enc, name), name)
        base = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        n_base = sum(p.numel() for p in base.parameters())
        n_perc = sum(p.numel() for p in enc.parameters())
        self.assertLess(n_perc, n_base - 150_000)

    def test_memory_token_drives_recurrence(self):
        enc = architectures.PerceiverEncoder()
        game = Game(seed=130)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        # Recurrence is live and features carry the recurrent state.
        self.assertFalse(torch.equal(out1["memory_out"], out2["memory_out"]))
        self.assertTrue(torch.equal(out1["features"], out1["memory_out"]))
        self.assertEqual(tuple(out1["all_tokens"].shape), (1, 19, enc.d_token_dim))

    def test_actor_ignores_features_reads_tokens(self):
        _seed_all(12)
        agent = PPOAgent(len(ACTIONS), arch="perceiver")
        game = Game(seed=131)
        s = game.players[0].get_state_dict()
        enc_out = agent.encoder.encode_batch([s])
        mask = torch.ones(1, len(ACTIONS), dtype=torch.bool)
        hand_ids = torch.as_tensor(s["hand_ids"], dtype=torch.long).view(1, -1)
        with torch.no_grad():
            _, logits1 = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
            enc_out2 = dict(enc_out)
            enc_out2["features"] = torch.randn_like(enc_out["features"])
            _, logits2 = agent.actor.forward_with_logits(
                enc_out2, mask, hand_ids, agent.encoder.card
            )
            enc_out3 = dict(enc_out)
            enc_out3["all_tokens"] = enc_out["all_tokens"].clone()
            enc_out3["all_tokens"][:, 1, :] += 1.0
            _, logits3 = agent.actor.forward_with_logits(
                enc_out3, mask, hand_ids, agent.encoder.card
            )
        self.assertTrue(torch.equal(logits1, logits2))
        self.assertFalse(torch.equal(logits1, logits3))

    def test_critic_reads_tokens_and_sequences(self):
        _seed_all(13)
        agent = PPOAgent(len(ACTIONS), arch="perceiver")
        game = Game(seed=132)
        s = game.players[0].get_state_dict()
        enc_out = agent.encoder.encode_batch([s])
        with torch.no_grad():
            v1 = agent.critic(enc_out)
            enc_out2 = dict(enc_out)
            enc_out2["all_tokens"] = enc_out["all_tokens"].clone()
            enc_out2["all_tokens"][:, 1, :] += 1.0
            v2 = agent.critic(enc_out2)
        self.assertEqual(tuple(v1.shape), (1, 1))
        self.assertFalse(torch.equal(v1, v2))
        seq_out = agent.encoder.encode_sequences([[s, s], [s]])
        vals = agent.critic.sequence_values(seq_out)
        self.assertEqual(tuple(vals.shape), (2, 2))
        with self.assertRaises(RuntimeError):
            agent.critic.aux_predictions(enc_out)

    def test_base_critic_sequence_values_matches_inline(self):
        # The seam must be exactly the old two lines for existing archs.
        _seed_all(14)
        agent = PPOAgent(len(ACTIONS), arch="full")
        game = Game(seed=133)
        s = game.players[0].get_state_dict()
        seq_out = agent.encoder.encode_sequences([[s, s]])
        with torch.no_grad():
            via_seam = agent.critic.sequence_values(seq_out)
            inline = agent.critic.value_head(
                agent.critic.value_trunk(seq_out["features"])
            ).squeeze(-1)
        self.assertTrue(torch.equal(via_seam, inline))


class TestSizeVariants(unittest.TestCase):
    def test_dmodel_shapes_propagate(self):
        for d_model in (128, 512):
            enc = CardReasoningEncoder(
                card_config=CardEmbeddingConfig(), d_model=d_model
            )
            self.assertEqual(enc.d_model, d_model)
            game = Game(seed=200 + d_model)
            out = enc.encode_batch([game.players[0].get_state_dict()])
            self.assertEqual(tuple(out["features"].shape), (1, d_model))
            self.assertEqual(tuple(out["memory_out"].shape), (1, d_model))

    def test_default_dmodel_matches_historical_pool_widths(self):
        enc = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        self.assertEqual(enc.d_model, 256)
        # 256 -> 64/64/32/32: the historical constants exactly.
        self.assertEqual(enc.pool_hand.proj.out_features, 64)
        self.assertEqual(enc.pool_blind.proj.out_features, 32)

    def test_agent_state_size_follows_encoder(self):
        _seed_all(6)
        agent = PPOAgent(len(ACTIONS), arch="full-dmodel128")
        self.assertEqual(agent.state_size, 128)
        mem = agent.get_recurrent_memory(None)
        self.assertEqual(tuple(mem.shape), (128,))


class TestOneHotState(unittest.TestCase):
    def test_dim_and_determinism(self):
        game = Game(seed=77)
        state = game.players[0].get_state_dict()
        v1 = build_onehot_state(state)
        v2 = build_onehot_state(state)
        self.assertEqual(v1.shape, (ONEHOT_STATE_DIM,))
        self.assertTrue(np.array_equal(v1, v2))

    def test_hand_multi_hot_matches_hand_ids(self):
        game = Game(seed=78)
        state = game.players[2].get_state_dict()
        vec = build_onehot_state(state)
        hand = vec[:34]
        expected = set(int(c) for c in state["hand_ids"] if int(c) > 0)
        self.assertEqual({i for i in range(34) if hand[i] == 1.0}, expected)
        self.assertEqual(hand.sum(), len(expected))

    def test_empty_state_is_zero_safe(self):
        vec = build_onehot_state({})
        # header one-hots for rel-seat value 0 are set; everything else zero
        self.assertEqual(vec.shape, (ONEHOT_STATE_DIM,))
        self.assertTrue(np.isfinite(vec).all())

    def test_encoder_contract_no_hand_tokens(self):
        enc = OneHotFeedForwardEncoder()
        game = Game(seed=79)
        batch = [p.get_state_dict() for p in game.players]
        out = enc.encode_batch(batch)
        self.assertEqual(tuple(out["features"].shape), (5, 256))
        self.assertEqual(tuple(out["memory_out"].shape), (5, 256))
        self.assertNotIn("hand_tokens", out)
        seq_out = enc.encode_sequences([batch, batch[:2]])
        self.assertEqual(tuple(seq_out["features"].shape), (2, 5, 256))
        self.assertNotIn("hand_tokens", seq_out)

    def test_memory_actually_recurs(self):
        enc = OneHotFeedForwardEncoder()
        game = Game(seed=80)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        self.assertFalse(torch.equal(out1["features"], out2["features"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
