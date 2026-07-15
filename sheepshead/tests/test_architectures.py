#!/usr/bin/env python3
"""Tests for the architecture registry (the architectures package).

Covers: every registered arch builds/plays/updates; the default "full" arch
is construction-identical to building the networks directly (the registry
adds no RNG activity — the guard behind bit-identical seeded training);
checkpoint arch metadata (record / mismatch / legacy default / load_agent);
the no-aux critic; the no-transformer encoder; and the one-hot state vector.
"""

import os
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sheepshead.agent import architectures
from sheepshead.training import pfsp_runtime
from sheepshead.agent import ppo
from sheepshead.agent.architectures import (
    ONEHOT_STATE_DIM,
    OneHotFeedForwardEncoder,
    PooledMemoryEncoder,
    build_onehot_state,
)
from sheepshead.agent.encoder import CardEmbeddingConfig, CardReasoningEncoder
from sheepshead.agent.ppo import (
    MultiHeadRecurrentActorNetwork,
    PPOAgent,
    RecurrentCriticNetwork,
)
from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

# Builds, plays, and updates every registered architecture (~30s+).
pytestmark = pytest.mark.slow

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


class TestRegistry:
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
            assert expected in names

    def test_unknown_arch_raises_with_names(self):
        with pytest.raises(KeyError) as ctx:
            architectures.get_spec("bogus")
        assert "full" in str(ctx.value)

    @pytest.mark.parametrize("arch", architectures.available_architectures())
    def test_all_archs_build_play_update(self, arch):
        _seed_all(11)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        assert agent.arch_name == arch
        _play_episodes(agent, 4)
        stats = agent.update(epochs=1, batch_size=16)
        assert stats["timing"]["optimizer_steps"] > 0
        assert np.isfinite(stats["approx_kl"])


class TestFullArchUnchanged:
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
            assert sorted(reg_sd) == sorted(dir_sd)
            for k in reg_sd:
                assert torch.equal(reg_sd[k], dir_sd[k]), f"weights differ: {k}"

    def test_full_arch_state_dict_keys_frozen(self):
        import hashlib

        agent = PPOAgent(len(ACTIONS))
        h = hashlib.sha256()
        for net in (agent.encoder, agent.actor, agent.critic):
            for k in sorted(net.state_dict().keys()):
                h.update(k.encode())
        assert h.hexdigest() == FULL_ARCH_KEYS_SHA256


class TestCheckpointArchMetadata:
    def _roundtrip(self, arch: str):
        _seed_all(5)
        agent = PPOAgent(len(ACTIONS), arch=arch)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            assert ckpt["arch"] == arch
            loaded = ppo.load_agent(path)
            assert loaded.arch_name == arch
            for k, v in agent.encoder.state_dict().items():
                assert torch.equal(v, loaded.encoder.state_dict()[k])

    @pytest.mark.parametrize("arch", architectures.available_architectures())
    def test_roundtrip_every_arch(self, arch):
        self._roundtrip(arch)

    def test_arch_mismatch_fails_loudly(self):
        agent = PPOAgent(len(ACTIONS), arch="no-aux")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            agent.save(path)
            full_agent = PPOAgent(len(ACTIONS))
            with pytest.raises(ValueError) as ctx:
                full_agent.load(path)
            assert "no-aux" in str(ctx.value)

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
            assert ppo.load_agent(path).arch_name == "full"
            # ...but not into a different arch.
            with pytest.raises(ValueError):
                PPOAgent(len(ACTIONS), arch="no-transformer").load(path)


class TestLegacyValueTrunkShim:
    """The pre-value_trunk critic shim, including the aliased re-save
    signature (a shimmed agent saved again, e.g. by league seeding): the
    aliased module used to serialize under both prefixes, leaving shallow
    ``value_trunk.*`` copies that half-loaded into a fresh deep trunk."""

    def _save_legacy(self, agent: PPOAgent, path: str) -> None:
        """Persist ``agent`` as a pre-value_trunk checkpoint."""
        agent.save(path)
        ckpt = torch.load(path, map_location="cpu")
        ckpt["critic_state_dict"] = {
            k: v
            for k, v in ckpt["critic_state_dict"].items()
            if not k.startswith("value_trunk")
        }
        torch.save(ckpt, path)

    def test_legacy_load_resave_roundtrip(self):
        _seed_all(11)
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            legacy = os.path.join(d, "legacy.pt")
            self._save_legacy(agent, legacy)
            first = ppo.load_agent(legacy)
            assert first.critic.value_trunk is first.critic.critic_adapter

            # Re-save the shimmed agent (the league seeding path): the save
            # must strip the aliased value_trunk.* duplicates...
            resaved = os.path.join(d, "resaved.pt")
            first.save(resaved)
            ckpt = torch.load(resaved, map_location="cpu")
            assert not any(
                k.startswith("value_trunk") for k in ckpt["critic_state_dict"]
            )
            # ...and the reload must shim again, with the trained weights.
            second = ppo.load_agent(resaved)
            assert second.critic.value_trunk is second.critic.critic_adapter
            want = first.critic.critic_adapter.state_dict()
            got = second.critic.value_trunk.state_dict()
            for k, v in want.items():
                assert torch.equal(v, got[k]), k

    def test_preexisting_aliased_resave_loads(self):
        # The on-disk signature this shim extension fixes: value_trunk.*
        # present as a shallow tensor-equal copy of critic_adapter.* (written
        # by a save() that predates the strip).
        _seed_all(12)
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "aliased.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            cs = {
                k: v
                for k, v in ckpt["critic_state_dict"].items()
                if not k.startswith("value_trunk")
            }
            for k, v in list(cs.items()):
                if k.startswith("critic_adapter."):
                    cs["value_trunk." + k[len("critic_adapter.") :]] = v.clone()
            ckpt["critic_state_dict"] = cs
            torch.save(ckpt, path)

            loaded = ppo.load_agent(path)
            assert loaded.critic.value_trunk is loaded.critic.critic_adapter
            want = agent.critic.critic_adapter.state_dict()
            got = loaded.critic.value_trunk.state_dict()
            for k, v in want.items():
                assert torch.equal(v, got[k]), k

    def test_truncated_deep_trunk_does_not_shim(self):
        # A real deep-trunk checkpoint missing only its deep layer is NOT the
        # legacy signature (value_trunk.0/1 differ from critic_adapter.*):
        # the shim must not silently alias it.
        _seed_all(13)
        agent = PPOAgent(len(ACTIONS))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "truncated.pt")
            agent.save(path)
            ckpt = torch.load(path, map_location="cpu")
            for k in ("value_trunk.3.weight", "value_trunk.3.bias"):
                del ckpt["critic_state_dict"][k]
            torch.save(ckpt, path)
            loaded = ppo.load_agent(path)
            assert loaded.critic.value_trunk is not loaded.critic.critic_adapter


class TestNoAuxCritic:
    def test_aux_modules_absent_and_accessors_raise(self):
        critic = RecurrentCriticNetwork(use_aux_heads=False)
        assert not critic.has_aux_heads
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
            assert not hasattr(critic, attr), attr
        feat = torch.zeros(1, 256)
        with pytest.raises(RuntimeError):
            critic.aux_predictions({"features": feat})
        with pytest.raises(RuntimeError):
            critic.seen_trump_mask_logits(feat, None)
        with pytest.raises(RuntimeError):
            critic.unseen_trump_higher_than_hand_logits(feat)

    def test_aux_critic_unchanged_when_enabled(self):
        torch.manual_seed(3)
        with_flag = RecurrentCriticNetwork(d_card=16, use_aux_heads=True)
        torch.manual_seed(3)
        default = RecurrentCriticNetwork(d_card=16)
        for k, v in default.state_dict().items():
            assert torch.equal(v, with_flag.state_dict()[k])

    def test_no_aux_agent_fewer_params_same_value_path(self):
        _seed_all(9)
        full = PPOAgent(len(ACTIONS))
        _seed_all(9)
        lean = PPOAgent(len(ACTIONS), arch="no-aux")
        n_full = sum(p.numel() for p in full.critic.parameters())
        n_lean = sum(p.numel() for p in lean.critic.parameters())
        assert n_lean < n_full
        # Same value-trunk shape: the value path is architecturally identical.
        assert {k for k in lean.critic.state_dict()} == {
            k
            for k in full.critic.state_dict()
            if k.startswith(("value_trunk", "value_head"))
        }


class TestNoTransformer:
    def test_reasoner_is_parameterless_noop(self):
        enc = PooledMemoryEncoder()
        assert sum(p.numel() for p in enc.card_reasoner.parameters()) == 0
        tokens = torch.randn(2, 19, enc.d_token_dim)
        mask = torch.ones(2, 19, dtype=torch.bool)
        out = enc.card_reasoner(tokens, mask)
        assert torch.equal(out, tokens)

    def test_encode_batch_contract(self):
        enc = PooledMemoryEncoder()
        game = Game(seed=123)
        out = enc.encode_batch([game.players[0].get_state_dict()])
        assert tuple(out["features"].shape) == (1, 256)
        assert tuple(out["hand_tokens"].shape) == (1, 8, enc.d_token_dim)

    def test_memory_feeds_features(self):
        # The whole point of the pooled-memory design: without attention the
        # base encoder's memory is write-only, so this variant must route the
        # recurrent state into the features the heads consume.
        enc = PooledMemoryEncoder()
        game = Game(seed=124)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        assert not torch.equal(out1["features"], out2["features"])
        # Heads consume the recurrent state itself (pre-transformer LSTM shape).
        assert torch.equal(out1["features"], out1["memory_out"])

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
        assert torch.equal(out1["features"], out2["features"])

    def test_uninformed_init_differs(self):
        torch.manual_seed(4)
        informed = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        torch.manual_seed(4)
        uninformed = CardReasoningEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        )
        assert not torch.equal(informed.card.weight, uninformed.card.weight)


class TestTokenRead:
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
        assert torch.equal(out_b["features"], out_t["features"])
        assert torch.equal(out_b["memory_out"], out_t["memory_out"])
        assert "all_tokens" not in out_b
        assert tuple(out_t["all_tokens"].shape) == (1, 19, enc.d_token_dim)
        assert tuple(out_t["all_mask"].shape) == (1, 19)
        # Context + memory tokens always valid (readout needs >= 1 key).
        assert bool(out_t["all_mask"][:, :2].all())

    def test_encode_sequences_carries_tokens(self):
        enc = architectures.TokenReadEncoder(card_config=CardEmbeddingConfig())
        game = Game(seed=127)
        seqs = [
            [game.players[0].get_state_dict(), game.players[0].get_state_dict()],
            [game.players[1].get_state_dict()],
        ]
        out = enc.encode_sequences(seqs)
        assert tuple(out["all_tokens"].shape) == (2, 2, 19, enc.d_token_dim)
        assert tuple(out["all_mask"].shape) == (2, 2, 19)
        assert out["all_mask"].dtype == torch.bool

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
        assert not torch.equal(logits1, logits2)

    def test_actor_requires_tokens(self):
        _seed_all(9)
        agent = PPOAgent(len(ACTIONS), arch="full-tokenread")
        with pytest.raises(RuntimeError):
            agent.actor._adapt_features(torch.randn(1, 256))

    def test_base_actor_ignores_token_kwargs(self):
        # The threaded-through kwargs must be inert for the default arch.
        _seed_all(10)
        agent = PPOAgent(len(ACTIONS), arch="full")
        game = Game(seed=129)
        s = game.players[0].get_state_dict()
        enc_out = agent.encoder.encode_batch([s])
        assert "all_tokens" not in enc_out
        mask = torch.ones(1, len(ACTIONS), dtype=torch.bool)
        hand_ids = torch.as_tensor(s["hand_ids"], dtype=torch.long).view(1, -1)
        with torch.no_grad():
            probs, _ = agent.actor.forward_with_logits(
                enc_out, mask, hand_ids, agent.encoder.card
            )
        assert torch.isfinite(probs).all()


class TestPerceiver:
    def test_pools_and_trunk_gone(self):
        enc = architectures.PerceiverEncoder()
        for name in (
            "pool_hand",
            "pool_trick",
            "pool_blind",
            "pool_bury",
            "feature_proj",
        ):
            assert not hasattr(enc, name), name
        base = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        n_base = sum(p.numel() for p in base.parameters())
        n_perc = sum(p.numel() for p in enc.parameters())
        assert n_perc < n_base - 150_000

    def test_memory_token_drives_recurrence(self):
        enc = architectures.PerceiverEncoder()
        game = Game(seed=130)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        # Recurrence is live and features carry the recurrent state.
        assert not torch.equal(out1["memory_out"], out2["memory_out"])
        assert torch.equal(out1["features"], out1["memory_out"])
        assert tuple(out1["all_tokens"].shape) == (1, 19, enc.d_token_dim)

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
        assert torch.equal(logits1, logits2)
        assert not torch.equal(logits1, logits3)

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
        assert tuple(v1.shape) == (1, 1)
        assert not torch.equal(v1, v2)
        seq_out = agent.encoder.encode_sequences([[s, s], [s]])
        vals = agent.critic.sequence_values(seq_out)
        assert tuple(vals.shape) == (2, 2)
        with pytest.raises(RuntimeError):
            agent.critic.aux_predictions(enc_out)

    def test_size_variants_propagate(self):
        _seed_all(15)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-dmodel128")
        assert agent.state_size == 128
        enc = architectures.PerceiverEncoder(d_token=128)
        assert enc.d_token_dim == 128
        game = Game(seed=134)
        out = enc.encode_batch([game.players[0].get_state_dict()])
        assert tuple(out["all_tokens"].shape) == (1, 19, 128)

    def test_attention_shape_variants(self):
        _seed_all(16)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-readq8")
        assert tuple(agent.actor.readout_query.shape) == (8, 64)
        assert tuple(agent.critic.readout_query.shape) == (8, 64)
        # End-to-end: a non-default readout shape must act.
        game = Game(seed=140)
        p = game.players[0]
        valid = p.get_valid_action_ids()
        a, _, _ = agent.act(p.get_state_dict(), list(valid), p.position)
        assert a in valid

        agent2 = PPOAgent(len(ACTIONS), arch="perceiver-readheads2")
        assert agent2.actor.readout_mha.num_heads == 2
        assert agent2.critic.readout_mha.num_heads == 2

        agent3 = PPOAgent(len(ACTIONS), arch="perceiver-rheads8")
        assert agent3.encoder.card_reasoner.attn_layers[0].num_heads == 8

        # Helper defaults still reproduce the base perceiver readout shape,
        # so the pre-existing size variants are untouched by the refactor.
        agent4 = PPOAgent(len(ACTIONS), arch="perceiver-layers6")
        assert tuple(agent4.actor.readout_query.shape) == (4, 64)
        assert agent4.actor.readout_mha.num_heads == 4

    def test_perceiver_aux_critic(self):
        _seed_all(17)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-aux")
        assert agent.critic.has_aux_heads
        for mod in ("win_head", "points_head", "trump_aux", "readout_query"):
            assert hasattr(agent.critic, mod), mod
        # Encoder is still pool-free.
        assert not hasattr(agent.encoder, "pool_hand")
        game = Game(seed=141)
        enc_out = agent.encoder.encode_batch([game.players[0].get_state_dict()])
        win_prob, _exp_ret, secret_prob, points = agent.critic.aux_predictions(enc_out)
        assert 0.0 <= win_prob <= 1.0
        assert 0.0 <= secret_prob <= 1.0
        assert len(points) == 5
        # Sequence aux features come from the token readout, (B, T, d_model).
        s = game.players[0].get_state_dict()
        seq_out = agent.encoder.encode_sequences([[s, s]])
        aux_bt = agent.critic.aux_sequence_features(seq_out)
        assert tuple(aux_bt.shape) == (1, 2, 256)

    def test_decomposition_hybrids(self):
        from sheepshead.agent.ppo import (
            MultiHeadRecurrentActorNetwork,
            PerceiverActorNetwork,
            PerceiverCriticNetwork,
            RecurrentCriticNetwork,
        )

        _seed_all(19)
        # readout-actor: perceiver actor + pooled no-aux critic, pools alive.
        ra = PPOAgent(len(ACTIONS), arch="readout-actor")
        assert isinstance(ra.actor, PerceiverActorNetwork)
        assert isinstance(ra.critic, RecurrentCriticNetwork)
        assert not ra.critic.has_aux_heads
        assert hasattr(ra.encoder, "pool_hand")
        # readout-critic: standard pointer actor + perceiver critic.
        rc = PPOAgent(len(ACTIONS), arch="readout-critic")
        assert not isinstance(rc.actor, PerceiverActorNetwork)
        assert isinstance(rc.actor, MultiHeadRecurrentActorNetwork)
        assert isinstance(rc.critic, PerceiverCriticNetwork)
        assert hasattr(rc.encoder, "pool_hand")

    def test_shared_readout_features_and_aux(self):
        _seed_all(21)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared")
        enc = agent.encoder
        # Pools/fusion gone; shared readout present; aux critic wired.
        assert not hasattr(enc, "pool_hand")
        assert not hasattr(enc, "feature_proj")
        assert hasattr(enc, "readout_query")
        assert agent.critic.has_aux_heads
        game = Game(seed=144)
        s = game.players[0].get_state_dict()
        out = enc.encode_batch([s])
        assert tuple(out["features"].shape) == (1, 256)
        # Features flow from the readout (not the memory state) and carry
        # gradient to the readout query.
        assert not torch.equal(out["features"], out["memory_out"])
        out["features"].sum().backward()
        assert enc.readout_query.grad is not None
        # Memory driver is the context token (full's convention).
        with torch.no_grad():
            out2 = enc.encode_batch([s])
            expect = enc.memory_gru(out2["all_tokens"][:, 0, :], torch.zeros(1, 256))
        assert torch.allclose(out2["memory_out"], expect, atol=1e-6)
        # param_groups cover every encoder parameter exactly once.
        grouped = [p for g in enc.param_groups(3e-4) for p in g["params"]]
        assert sorted(p.data_ptr() for p in grouped) == sorted(
            p.data_ptr() for p in enc.parameters()
        )

    def test_shared_readout_v2_corrections(self):
        _seed_all(22)
        agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2")
        enc = agent.encoder
        # 16 queries, normed readout, context-token driver, aux critic.
        assert enc.readout_n_queries == 16
        assert tuple(enc.readout_query.shape) == (16, enc.d_token_dim)
        assert isinstance(enc.readout_proj, torch.nn.Sequential)
        assert isinstance(enc.readout_proj[1], torch.nn.LayerNorm)
        assert not enc.memory_token_driver
        assert agent.critic.has_aux_heads
        game = Game(seed=145)
        s = game.players[0].get_state_dict()
        out = enc.encode_batch([s])
        assert tuple(out["features"].shape) == (1, 256)
        # LayerNorm pins the feature scale to full's convention (norm =
        # sqrt(d_model) with default affine init).
        assert float(out["features"].norm()) == pytest.approx(256**0.5, abs=0.5)
        # Memory driver is the context token (index 0), as in v1/full
        # (operator decision 2026-07-09: keep the game-start prior).
        with torch.no_grad():
            out2 = enc.encode_batch([s])
            expect = enc.memory_gru(out2["all_tokens"][:, 0, :], torch.zeros(1, 256))
        assert torch.allclose(out2["memory_out"], expect, atol=1e-6)
        # Gradient reaches the queries through the normed projection.
        out["features"].sum().backward()
        assert enc.readout_query.grad is not None
        # param_groups cover every encoder parameter exactly once.
        grouped = [p for g in enc.param_groups(3e-4) for p in g["params"]]
        assert sorted(p.data_ptr() for p in grouped) == sorted(
            p.data_ptr() for p in enc.parameters()
        )
        # v1 stays byte-compatible: default ctor keeps the bare Linear
        # projection, 4 queries, and the context-token driver.
        v1 = architectures.SharedReadoutEncoder()
        assert isinstance(v1.readout_proj, torch.nn.Linear)
        assert v1.readout_n_queries == 4
        assert not v1.memory_token_driver

    def test_ctxmem_context_token_drives_recurrence(self):
        _seed_all(20)
        enc = architectures.PerceiverCtxMemEncoder()
        game = Game(seed=143)
        s = game.players[0].get_state_dict()
        out = enc.encode_batch([s])
        with torch.no_grad():
            zero_mem = torch.zeros(1, 256)
            expect = enc.memory_gru(out["all_tokens"][:, 0, :], zero_mem)
        assert torch.allclose(out["memory_out"], expect, atol=1e-6)
        # Param-identical to the perceiver encoder (driver change only).
        base = architectures.PerceiverEncoder()
        assert sum(p.numel() for p in enc.parameters()) == sum(
            p.numel() for p in base.parameters()
        )

    def test_base_aux_seam_bit_identical(self):
        _seed_all(18)
        agent = PPOAgent(len(ACTIONS), arch="full")
        game = Game(seed=142)
        s = game.players[0].get_state_dict()
        seq_out = agent.encoder.encode_sequences([[s, s]])
        with torch.no_grad():
            via_seam = agent.critic.aux_sequence_features(seq_out)
            inline = agent.critic.critic_adapter(seq_out["features"])
        assert torch.equal(via_seam, inline)

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
        assert torch.equal(via_seam, inline)


class TestSizeVariants:
    def test_dmodel_shapes_propagate(self):
        for d_model in (128, 512):
            enc = CardReasoningEncoder(
                card_config=CardEmbeddingConfig(), d_model=d_model
            )
            assert enc.d_model == d_model
            game = Game(seed=200 + d_model)
            out = enc.encode_batch([game.players[0].get_state_dict()])
            assert tuple(out["features"].shape) == (1, d_model)
            assert tuple(out["memory_out"].shape) == (1, d_model)

    def test_default_dmodel_matches_historical_pool_widths(self):
        enc = CardReasoningEncoder(card_config=CardEmbeddingConfig())
        assert enc.d_model == 256
        # 256 -> 64/64/32/32: the historical constants exactly.
        assert enc.pool_hand.proj.out_features == 64
        assert enc.pool_blind.proj.out_features == 32

    def test_agent_state_size_follows_encoder(self):
        _seed_all(6)
        agent = PPOAgent(len(ACTIONS), arch="full-dmodel128")
        assert agent.state_size == 128
        mem = agent.get_recurrent_memory(None)
        assert tuple(mem.shape) == (128,)


class TestOneHotState:
    def test_dim_and_determinism(self):
        game = Game(seed=77)
        state = game.players[0].get_state_dict()
        v1 = build_onehot_state(state)
        v2 = build_onehot_state(state)
        assert v1.shape == (ONEHOT_STATE_DIM,)
        assert np.array_equal(v1, v2)

    def test_hand_multi_hot_matches_hand_ids(self):
        game = Game(seed=78)
        state = game.players[2].get_state_dict()
        vec = build_onehot_state(state)
        hand = vec[:34]
        expected = set(int(c) for c in state["hand_ids"] if int(c) > 0)
        assert {i for i in range(34) if hand[i] == 1.0} == expected
        assert hand.sum() == len(expected)

    def test_empty_state_is_zero_safe(self):
        vec = build_onehot_state({})
        # header one-hots for rel-seat value 0 are set; everything else zero
        assert vec.shape == (ONEHOT_STATE_DIM,)
        assert np.isfinite(vec).all()

    def test_encoder_contract_no_hand_tokens(self):
        enc = OneHotFeedForwardEncoder()
        game = Game(seed=79)
        batch = [p.get_state_dict() for p in game.players]
        out = enc.encode_batch(batch)
        assert tuple(out["features"].shape) == (5, 256)
        assert tuple(out["memory_out"].shape) == (5, 256)
        assert "hand_tokens" not in out
        seq_out = enc.encode_sequences([batch, batch[:2]])
        assert tuple(seq_out["features"].shape) == (2, 5, 256)
        assert "hand_tokens" not in seq_out

    def test_memory_actually_recurs(self):
        enc = OneHotFeedForwardEncoder()
        game = Game(seed=80)
        s = game.players[0].get_state_dict()
        out1 = enc.encode_batch([s])
        out2 = enc.encode_batch([s], memory_in=out1["memory_out"])
        assert not torch.equal(out1["features"], out2["features"])


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
