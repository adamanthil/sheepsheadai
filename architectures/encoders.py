"""Encoder variants for the architecture registry.

Subclasses of encoder.CardReasoningEncoder that override its two
documented seams (_pool_fuse_update / _fuse_and_update_memory) to
produce the ablation and redesign rungs. Moved verbatim from the
original single-file architectures.py; state_dicts and RNG behavior
are unchanged (checkpoints store no class paths)."""

import torch
import torch.nn as nn

from encoder import CardEmbeddingConfig, CardReasoningEncoder


# ---------------------------------------------------------------------------
# Pooled-memory encoder (the "no-transformer" rung)
# ---------------------------------------------------------------------------


class PooledMemoryEncoder(CardReasoningEncoder):
    """Embeddings + attention pools + recurrence, no cross-card transformer.

    Restores the pre-transformer recurrence shape (encoder -> LSTM over the
    fused 256-d features -> heads; see ppo.py before commit 0729e11): the
    fused pooled features feed a GRUCell(256, 256) and the heads consume the
    recurrent state itself.

    Why not just n_reasoning_layers=0 on the base class? Without attention
    the base encoder's memory token cannot mix back into the features and
    the context-fed GRU input never sees the cards — memory becomes
    write-only, i.e. the policy would be effectively memoryless. That would
    conflate "no transformer" with "no usable memory". Here memory sees
    everything the heads see, so the rung isolates cross-card attention.

    The inherited memory token machinery (memory_in_proj, card_type id 1)
    stays constructed but is inert with zero reasoning layers (~16k dead
    params — accepted to keep the base encode_batch untouched).
    """

    def __init__(self, card_config: "CardEmbeddingConfig | None" = None):
        super().__init__(
            card_config=card_config or CardEmbeddingConfig(),
            n_reasoning_layers=0,
        )
        # Replace the context-token GRU (d_token -> d_model) with a
        # fused-features GRU.
        self.memory_gru = nn.GRUCell(self.d_model, self.d_model)

    def _fuse_and_update_memory(
        self,
        context_out,
        hand_tok_out,
        hand_vec,
        trick_vec,
        blind_vec,
        bury_vec,
        memory_in,
        all_tokens=None,
        all_mask=None,
    ):
        del all_tokens, all_mask  # not exposed by this rung
        features = self.feature_proj(
            torch.cat([hand_vec, trick_vec, blind_vec, bury_vec, context_out], dim=1)
        )
        memory_out = self.memory_gru(features, memory_in)
        return {
            "features": memory_out,
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
        }


# ---------------------------------------------------------------------------
# Token-readout encoder (the "full-tokenread" rung)
# ---------------------------------------------------------------------------


class TokenReadEncoder(CardReasoningEncoder):
    """`full`'s encoder, additionally exposing the post-reasoning token set.

    Zero new parameters and no change to features or memory: the standard
    outputs are byte-identical to the base class; 'all_tokens'
    (B, 19, d_token) and 'all_mask' (B, 19) are emitted alongside them so a
    readout-equipped actor (ppo.TokenReadActorNetwork) can attend over the
    tokens directly instead of seeing them only through the per-bag
    attention pools. The memory recurrence is untouched: context token →
    GRUCell → d_model state, re-projected to a memory token next step by
    memory_in_proj — and because the post-reasoning memory token is part of
    'all_tokens', the actor's readout reads memory directly too (the base
    architecture discards that token).
    """

    def _fuse_and_update_memory(
        self,
        context_out,
        hand_tok_out,
        hand_vec,
        trick_vec,
        blind_vec,
        bury_vec,
        memory_in,
        all_tokens=None,
        all_mask=None,
    ):
        out = super()._fuse_and_update_memory(
            context_out,
            hand_tok_out,
            hand_vec,
            trick_vec,
            blind_vec,
            bury_vec,
            memory_in,
        )
        out["all_tokens"] = all_tokens
        out["all_mask"] = all_mask
        return out


# ---------------------------------------------------------------------------
# Perceiver-IO encoder (the "perceiver" rung — clean token-centric design)
# ---------------------------------------------------------------------------


class PerceiverEncoder(CardReasoningEncoder):
    """Token-centric encoder: embeddings + transformer + recurrence, nothing
    else. The per-bag attention pools and the fused feature trunk are GONE —
    every consumer (PerceiverActorNetwork, PerceiverCriticNetwork) reads the
    post-reasoning token set through its own attention readout.

    Memory (operator's design, 2026-07-06): the GRU input is the
    post-reasoning MEMORY token — the transformer's own "what to remember"
    slot, which attends over everything during reasoning and which the base
    architecture computes and discards (its GRU reads the *context* token
    instead). Same GRUCell(d_token, d_model) shape, zero parameter change;
    the recurrent state re-enters next step via memory_in_proj as today.

    The 'features' output is set to the recurrent state purely so generic
    plumbing that expects a (B, d_model) tensor keeps working; the perceiver
    actor and critic both ignore it. Removes the pools + feature_proj
    (~178k params) relative to the base encoder.
    """

    def __init__(
        self, card_config: "CardEmbeddingConfig | None" = None, **encoder_kwargs
    ):
        # encoder_kwargs (d_token / d_model / n_reasoning_layers / ...) pass
        # through to CardReasoningEncoder so perceiver size variants exist
        # (the capacity-sweep knobs must be probeable on THIS base too —
        # sweeping only `full` confounds every knob with the pool squeeze).
        super().__init__(
            card_config=card_config or CardEmbeddingConfig(), **encoder_kwargs
        )
        del self.pool_hand
        del self.pool_trick
        del self.pool_blind
        del self.pool_bury
        del self.feature_proj

    def param_groups(self, base_lr: float, card_lr_scale: float = 0.1):
        import itertools

        other_params = itertools.chain(
            self.seat.parameters(),
            self.role.parameters(),
            self.card_type.parameters(),
            self.context_mlp.parameters(),
            self.memory_in_proj.parameters(),
            self.token_mlp_hand.parameters(),
            self.token_mlp_trick.parameters(),
            self.token_mlp_simple.parameters(),
            self.card_reasoner.parameters(),
            self.memory_gru.parameters(),
        )
        return [
            {"params": self.card.parameters(), "lr": base_lr * card_lr_scale},
            {"params": other_params, "lr": base_lr},
        ]

    def _pool_fuse_update(
        self,
        context_out,
        hand_tok_out,
        hand_mask,
        trick_tok_out,
        trick_mask,
        blind_tok_out,
        blind_mask,
        bury_tok_out,
        bury_mask,
        memory_in,
        all_tokens,
        all_mask,
    ):
        # Memory write: the post-reasoning MEMORY token (index 1).
        memory_out = self.memory_gru(all_tokens[:, 1, :], memory_in)
        return {
            "features": memory_out,  # vestigial (see class docstring)
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
            "all_tokens": all_tokens,
            "all_mask": all_mask,
        }


class PerceiverCtxMemEncoder(PerceiverEncoder):
    """PerceiverEncoder with `full`'s memory driver: the GRU input is the
    post-reasoning CONTEXT token (index 0) instead of the memory token.
    Decomposition arm for the perceiver-probe loss (2026-07): the probe
    changed three things at once (actor trunk, critic trunk, memory driver);
    `perceiver` vs `perceiver-ctxmem` isolates the memory-driver change
    alone. Same GRUCell shape, zero parameter change.
    """

    def _pool_fuse_update(
        self,
        context_out,
        hand_tok_out,
        hand_mask,
        trick_tok_out,
        trick_mask,
        blind_tok_out,
        blind_mask,
        bury_tok_out,
        bury_mask,
        memory_in,
        all_tokens,
        all_mask,
    ):
        # Memory write: the post-reasoning CONTEXT token (index 0), as in
        # the base architecture.
        memory_out = self.memory_gru(all_tokens[:, 0, :], memory_in)
        return {
            "features": memory_out,  # vestigial (see PerceiverEncoder)
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
            "all_tokens": all_tokens,
            "all_mask": all_mask,
        }


class SharedReadoutEncoder(PerceiverEncoder):
    """Pool-free encoder with ONE shared token readout producing `features`.

    `full` with the four bag-scoped pools + fusion MLP replaced by a single
    4-query x 4-head readout over all 19 post-reasoning tokens (the same
    module design as AttentionPool, un-scoped). Everything downstream is
    standard: the pointer actor and the aux critic both consume the shared
    256-d `features`, so aux gradients shape the very vector the policy
    reads (full's forcing mechanism, which per-network perceiver readouts
    reduce to indirect token shaping), and decision-time attention runs
    ONCE in the encoder instead of per network. Memory driver is the
    context token, as in `full`. vs full this isolates bag SCOPING itself,
    holding sharing/aux/memory fixed.
    """

    def __init__(
        self,
        card_config: "CardEmbeddingConfig | None" = None,
        n_readout_queries: int = 4,
        n_readout_heads: int = 4,
        normed_readout: bool = False,
        memory_token_driver: bool = False,
        **encoder_kwargs,
    ):
        super().__init__(card_config=card_config, **encoder_kwargs)
        d_token = self.d_token_dim
        d_model = self.d_model
        self.readout_n_queries = int(n_readout_queries)
        self.memory_token_driver = bool(memory_token_driver)
        # Torch-default MHA init + randn queries: the AttentionPool
        # convention (matches the modules this readout replaces).
        self.readout_query = nn.Parameter(torch.randn(self.readout_n_queries, d_token))
        self.readout_mha = nn.MultiheadAttention(
            d_token, int(n_readout_heads), batch_first=True
        )
        # normed_readout matches full's feature_proj convention
        # (Linear + LayerNorm); the default bare Linear preserves
        # state-dict compatibility with pre-v2 perceiver-shared runs.
        if normed_readout:
            self.readout_proj = nn.Sequential(
                nn.Linear(self.readout_n_queries * d_token, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.readout_proj = nn.Linear(self.readout_n_queries * d_token, d_model)

    def param_groups(self, base_lr: float, card_lr_scale: float = 0.1):
        groups = super().param_groups(base_lr, card_lr_scale)
        groups.append(
            {
                "params": [
                    self.readout_query,
                    *self.readout_mha.parameters(),
                    *self.readout_proj.parameters(),
                ],
                "lr": base_lr,
            }
        )
        return groups

    def _pool_fuse_update(
        self,
        context_out,
        hand_tok_out,
        hand_mask,
        trick_tok_out,
        trick_mask,
        blind_tok_out,
        blind_mask,
        bury_tok_out,
        bury_mask,
        memory_in,
        all_tokens,
        all_mask,
    ):
        # Memory write: context token by default (as in the base
        # architecture); v2 drives recurrence from the post-reasoning
        # MEMORY token instead (the original perceiver design — the
        # transformer learns what the GRU needs to see).
        driver = (
            all_tokens[:, 1, :] if self.memory_token_driver else all_tokens[:, 0, :]
        )
        memory_out = self.memory_gru(driver, memory_in)
        B = all_tokens.size(0)
        q = self.readout_query.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.readout_mha(
            q,
            all_tokens,
            all_tokens,
            key_padding_mask=~all_mask,
            need_weights=False,
        )
        features = self.readout_proj(
            attn_out.reshape(B, self.readout_n_queries * self.d_token_dim)
        )
        return {
            "features": features,
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
            "all_tokens": all_tokens,
            "all_mask": all_mask,
        }
