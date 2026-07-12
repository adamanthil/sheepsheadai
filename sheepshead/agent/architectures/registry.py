"""ArchitectureSpec, the factory helpers, and every registered entry.

See the package docstring (architectures/__init__.py) for the design
contract. Entries moved verbatim from the single-file architectures.py."""

from dataclasses import dataclass
from typing import Any, Callable, Dict

import torch.nn as nn

from sheepshead.agent.encoder import CardEmbeddingConfig, CardReasoningEncoder

from .encoders import (
    PerceiverCtxMemEncoder,
    PerceiverEncoder,
    PooledMemoryEncoder,
    SharedReadoutEncoder,
    TokenReadEncoder,
)
from .onehot import FlatHeadActorNetwork, OneHotFeedForwardEncoder


@dataclass(frozen=True)
class ArchitectureSpec:
    """Named bundle of network factories consumed by PPOAgent.__init__.

    Factories import from ppo lazily (inside the function bodies) so this
    module can be imported by ppo.py without a cycle. They must not perform
    any RNG-consuming work beyond the network constructions themselves:
    PPOAgent relies on construction order (encoder -> actor -> critic) for
    seeded reproducibility.
    """

    name: str
    description: str
    build_encoder: Callable[[], nn.Module]
    build_actor: Callable[..., nn.Module]
    build_critic: Callable[[nn.Module], nn.Module]
    # Declarative metadata only; the runtime truth is the built critic's
    # has_aux_heads attribute. tests/test_arch_golden.py welds the two (and
    # the aux modules' state_dict presence) together for every entry.
    has_aux_heads: bool = True


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _pointer_actor(action_size, action_groups, encoder, mappings):
    from sheepshead.agent.ppo import MultiHeadRecurrentActorNetwork

    return MultiHeadRecurrentActorNetwork(
        action_size,
        action_groups,
        d_card=encoder.d_card_dim,
        d_token=encoder.d_token_dim,
        d_model=encoder.d_model,
        **mappings,
    )


def _tokenread_actor(action_size, action_groups, encoder, mappings):
    from sheepshead.agent.ppo import TokenReadActorNetwork

    return TokenReadActorNetwork(
        action_size,
        action_groups,
        d_card=encoder.d_card_dim,
        d_token=encoder.d_token_dim,
        d_model=encoder.d_model,
        **mappings,
    )


def _perceiver_actor(action_size, action_groups, encoder, mappings):
    from sheepshead.agent.ppo import PerceiverActorNetwork

    return PerceiverActorNetwork(
        action_size,
        action_groups,
        d_card=encoder.d_card_dim,
        d_token=encoder.d_token_dim,
        d_model=encoder.d_model,
        **mappings,
    )


def _perceiver_critic(encoder):
    from sheepshead.agent.ppo import PerceiverCriticNetwork

    return PerceiverCriticNetwork(
        d_token=encoder.d_token_dim,
        d_model=encoder.d_model,
    )


def _perceiver_aux_critic(encoder):
    from sheepshead.agent.ppo import PerceiverAuxCriticNetwork

    return PerceiverAuxCriticNetwork(
        d_token=encoder.d_token_dim,
        d_model=encoder.d_model,
        d_card=encoder.d_card_dim,
    )


def _aux_critic(encoder):
    from sheepshead.agent.ppo import RecurrentCriticNetwork

    return RecurrentCriticNetwork(d_card=encoder.d_card_dim, d_model=encoder.d_model)


def _no_aux_critic(encoder):
    from sheepshead.agent.ppo import RecurrentCriticNetwork

    d_card = getattr(encoder, "d_card_dim", 0) or None
    return RecurrentCriticNetwork(
        d_card=d_card,
        use_aux_heads=False,
        d_model=encoder.d_model,
    )


def _full_size_variant(
    name: str, description: str, **encoder_kwargs
) -> ArchitectureSpec:
    """A `full`-shaped spec with one (or more) encoder dimension overridden.

    Actor and critic read their widths off the encoder (d_model / d_token /
    d_card), so a size variant is fully specified by encoder kwargs. Used by
    the capacity sweep: is the current full architecture the right SIZE?
    """
    return ArchitectureSpec(
        name=name,
        description=description,
        build_encoder=lambda: CardReasoningEncoder(
            card_config=CardEmbeddingConfig(), **encoder_kwargs
        ),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )


def _perceiver_size_variant(
    name: str,
    description: str,
    n_readout_queries: int = 4,
    n_readout_heads: int = 4,
    **encoder_kwargs,
) -> ArchitectureSpec:
    """A `perceiver`-shaped spec with one dimension overridden.

    Mirror of _full_size_variant for the token-centric base: if the
    perceiver becomes the default architecture, capacity questions must be
    answered on it directly — a depth/width sweep on `full` is confounded
    by the attention-pool squeeze the perceiver removes. Besides the
    encoder kwargs, the actor/critic readout attention shape
    (n_readout_queries x n_readout_heads, both historically an unexamined
    4) is sweepable; the defaults reproduce the base `perceiver` networks
    exactly.
    """

    def build_actor(action_size, action_groups, encoder, mappings):
        from sheepshead.agent.ppo import PerceiverActorNetwork

        return PerceiverActorNetwork(
            action_size,
            action_groups,
            d_card=encoder.d_card_dim,
            d_token=encoder.d_token_dim,
            d_model=encoder.d_model,
            n_readout_queries=n_readout_queries,
            n_readout_heads=n_readout_heads,
            **mappings,
        )

    def build_critic(encoder):
        from sheepshead.agent.ppo import PerceiverCriticNetwork

        return PerceiverCriticNetwork(
            d_token=encoder.d_token_dim,
            d_model=encoder.d_model,
            n_readout_queries=n_readout_queries,
            n_readout_heads=n_readout_heads,
        )

    return ArchitectureSpec(
        name=name,
        description=description,
        build_encoder=lambda: PerceiverEncoder(
            card_config=CardEmbeddingConfig(), **encoder_kwargs
        ),
        build_actor=build_actor,
        build_critic=build_critic,
        has_aux_heads=False,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES: Dict[str, ArchitectureSpec] = {}


def register(spec: ArchitectureSpec) -> ArchitectureSpec:
    """Add a spec to the registry keyed by its own name (the single
    source of truth for architecture identity); duplicate names fail at
    import time. Registration constructs nothing and consumes no RNG."""
    if spec.name in ARCHITECTURES:
        raise ValueError(f"duplicate architecture name: {spec.name}")
    ARCHITECTURES[spec.name] = spec
    return spec


register(
    ArchitectureSpec(
        name="full",
        description=(
            "Current architecture: informed card embeddings + transformer "
            "card reasoning + GRU memory + pointer/two-tower actor + limited "
            "critic with auxiliary heads."
        ),
        build_encoder=lambda: CardReasoningEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )
)

# --- Ablation ladder (reverse-historical, cumulative) -----------------
register(
    ArchitectureSpec(
        name="no-aux",
        description="full minus the critic's auxiliary heads.",
        build_encoder=lambda: CardReasoningEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="no-transformer",
        description=(
            "no-aux minus transformer card reasoning: PooledMemoryEncoder "
            "(embeddings -> pools -> fused features -> GRU; heads consume "
            "the recurrent state, matching the pre-transformer LSTM shape)."
        ),
        build_encoder=lambda: PooledMemoryEncoder(),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="no-transformer-uninformed",
        description="no-transformer minus informed card-embedding initialization.",
        build_encoder=lambda: PooledMemoryEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        ),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="onehot-ff",
        description=(
            "Legacy baseline: flat one-hot state -> feed-forward MLP + GRU "
            "memory, flat per-phase linear action heads. No card embeddings, "
            "no tokens, no aux heads."
        ),
        build_encoder=lambda: OneHotFeedForwardEncoder(),
        build_actor=lambda action_size, action_groups, encoder, mappings: (
            FlatHeadActorNetwork(action_size, action_groups)
        ),
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

# --- Capacity sweep around `full` (one knob per variant) ---------------
register(
    _full_size_variant(
        "full-dtok32",
        "full with d_token 64 -> 32 (transformer width /2).",
        d_token=32,
    )
)

register(
    _full_size_variant(
        "full-dtok128",
        "full with d_token 64 -> 128 (transformer width x2).",
        d_token=128,
    )
)

register(
    _full_size_variant(
        "full-layers2",
        "full with 2 reasoning layers (depth /2).",
        n_reasoning_layers=2,
    )
)

register(
    _full_size_variant(
        "full-layers6",
        "full with 6 reasoning layers (depth x1.5).",
        n_reasoning_layers=6,
    )
)

register(
    _full_size_variant(
        "full-dmodel128",
        "full with d_model 256 -> 128 (trunk/memory/pool width /2).",
        d_model=128,
    )
)

register(
    _full_size_variant(
        "full-dmodel512",
        "full with d_model 256 -> 512 (trunk/memory/pool width x2).",
        d_model=512,
    )
)

# --- Clean token-centric redesign (Perceiver-IO shape) ------------------
register(
    ArchitectureSpec(
        name="perceiver",
        description=(
            "Token-centric end to end: embeddings + transformer + recurrence "
            "shared; the actor AND the critic each read the 19 post-reasoning "
            "tokens through their own 4-query cross-attention readout. No "
            "per-bag pools, no fused feature trunk; the memory GRU is fed the "
            "post-reasoning memory token. No aux heads — compare against "
            "no-aux as well as full."
        ),
        build_encoder=lambda: PerceiverEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_perceiver_actor,
        build_critic=_perceiver_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-aux",
        description=(
            "perceiver plus the full auxiliary-head stack (the operator's "
            "intended design, added 2026-07-06): the critic's readout feeds "
            "both the value trunk and the aux adapter, so aux gradients "
            "shape the readout + encoder as they shape the pooled trunk in "
            "full. vs perceiver = the aux rung on the token-centric base."
        ),
        build_encoder=lambda: PerceiverEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_perceiver_actor,
        build_critic=_perceiver_aux_critic,
        has_aux_heads=True,
    )
)

# --- Perceiver-loss decomposition arms (2026-07-07) ---------------------
# The perceiver probe changed three things at once relative to no-aux:
# (1) the actor's trunk input (pooled fusion -> token readout), (2) the
# critic's trunk input (same swap), (3) the memory-GRU driver (context
# token -> memory token). These arms flip ONE switch each. The hybrids
# ride on TokenReadEncoder (byte-identical to the base encoder, also
# emits all_tokens), so the pooled path survives for whichever network
# still needs it and the memory driver stays `full`'s. Comparator for
# the hybrids is `no-aux` (all three are aux-free, pooled-encoder runs);
# comparator for perceiver-ctxmem is `perceiver`.
register(
    ArchitectureSpec(
        name="readout-actor",
        description=(
            "Hybrid decomposition arm: perceiver-style actor (token-readout "
            "trunk, ignores pooled features) + no-aux pooled critic on the "
            "tokenread encoder. vs no-aux = the actor-side pool deletion "
            "alone."
        ),
        build_encoder=lambda: TokenReadEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_perceiver_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="readout-critic",
        description=(
            "Hybrid decomposition arm: standard pooled/pointer actor + "
            "perceiver-style critic (token-readout value trunk) on the "
            "tokenread encoder. vs no-aux = the critic-side pool deletion "
            "alone."
        ),
        build_encoder=lambda: TokenReadEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_perceiver_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-shared",
        description=(
            "full with the four bag-scoped pools + fusion MLP replaced by a "
            "single shared 4-query/4-head readout over all 19 post-reasoning "
            "tokens; pointer actor, aux critic, and context-token memory "
            "driver unchanged. Isolates bag SCOPING (holding trunk sharing, "
            "aux forcing, and the memory driver fixed) and is the "
            "throughput-friendly token-centric layout (decision-time "
            "attention once, in the encoder)."
        ),
        build_encoder=lambda: SharedReadoutEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-shared-noaux",
        description=(
            "perceiver-shared minus the aux heads: shared encoder readout "
            "trunk, pointer actor, plain no-aux critic. The aux rung on "
            "the shared-readout base — the stage-1 league aux-contribution "
            "arm pairs this against perceiver-shared (under the oracle "
            "critic the only surviving aux channel is trunk-shaping for "
            "the actor)."
        ),
        build_encoder=lambda: SharedReadoutEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-shared-v2",
        description=(
            "perceiver-shared with the two 2026-07-09 corrections: "
            "16-query/4-head shared readout (64 attention distributions = "
            "channel parity with full's 4 pools x 4q x 4h; the 4q/4h v1 "
            "readout had 1/4 the channels, each softmaxing over all 19 "
            "competing tokens) and LayerNorm after the readout projection "
            "(full's feature_proj convention; v1's bare Linear left the "
            "trunk scale unpinned — consumers renormalize, but gradient "
            "geometry into the readout differs). Context-token memory "
            "driver, pointer actor, and aux critic unchanged (operator "
            "decision 2026-07-09: the context token is a strong prior at "
            "game start and worked in all past training — change fewer "
            "things). The family's best-effort adoption candidate; "
            "attribution of the two changes is registry-recoverable but "
            "secondary."
        ),
        build_encoder=lambda: SharedReadoutEncoder(
            card_config=CardEmbeddingConfig(),
            n_readout_queries=16,
            n_readout_heads=4,
            normed_readout=True,
        ),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-shared-v2-noaux",
        description=(
            "perceiver-shared-v2 minus the aux heads: the missing cell of "
            "the {pools|shared readout} x {aux|noaux} factorial on the "
            "corrected base (full / no-aux / perceiver-shared-v2 / this)."
        ),
        build_encoder=lambda: SharedReadoutEncoder(
            card_config=CardEmbeddingConfig(),
            n_readout_queries=16,
            n_readout_heads=4,
            normed_readout=True,
        ),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    )
)

register(
    ArchitectureSpec(
        name="perceiver-ctxmem",
        description=(
            "perceiver with full's memory driver (GRU fed the post-reasoning "
            "context token instead of the memory token). vs perceiver = the "
            "memory-driver change alone; zero parameter change."
        ),
        build_encoder=lambda: PerceiverCtxMemEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_perceiver_actor,
        build_critic=_perceiver_critic,
        has_aux_heads=False,
    )
)

# --- Perceiver capacity variants (one knob each, mirroring full's) ------
register(
    _perceiver_size_variant(
        "perceiver-dtok32",
        "perceiver with d_token 64 -> 32 (transformer/readout width /2).",
        d_token=32,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-dtok128",
        "perceiver with d_token 64 -> 128 (transformer/readout width x2).",
        d_token=128,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-layers2",
        "perceiver with 2 reasoning layers (depth /2).",
        n_reasoning_layers=2,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-layers6",
        "perceiver with 6 reasoning layers (depth x1.5).",
        n_reasoning_layers=6,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-dmodel128",
        "perceiver with d_model 256 -> 128 (memory/adapter/value width /2).",
        d_model=128,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-dmodel512",
        "perceiver with d_model 256 -> 512 (memory/adapter/value width x2).",
        d_model=512,
    )
)

# Attention-shape knobs: every 4 below (readout queries, readout heads,
# reasoning heads) was chosen without evidence when the transformer was
# first added — these variants finally probe them, one knob at a time.
register(
    _perceiver_size_variant(
        "perceiver-readq2",
        "perceiver with 2 readout queries per network (4 -> 2).",
        n_readout_queries=2,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-readq8",
        "perceiver with 8 readout queries per network (4 -> 8).",
        n_readout_queries=8,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-readheads2",
        "perceiver with 2 readout attention heads (4 -> 2, head_dim 32).",
        n_readout_heads=2,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-readheads8",
        "perceiver with 8 readout attention heads (4 -> 8, head_dim 8).",
        n_readout_heads=8,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-rheads2",
        "perceiver with 2 transformer reasoning heads (4 -> 2).",
        n_reasoning_heads=2,
    )
)

register(
    _perceiver_size_variant(
        "perceiver-rheads8",
        "perceiver with 8 transformer reasoning heads (4 -> 8).",
        n_reasoning_heads=8,
    )
)

# --- Readout variant (pooling-bottleneck hypothesis) --------------------
register(
    ArchitectureSpec(
        name="full-tokenread",
        description=(
            "full plus a cross-attention token readout in the actor: 4 "
            "learned queries attend over all 19 post-reasoning tokens and "
            "the result is fused with the pooled trunk features before the "
            "heads. Tests whether the per-bag attention-pool bottleneck "
            "limits the policy (readout STRUCTURE at fixed width; the "
            "capacity sweep's d_model variants test width). Encoder and "
            "memory recurrence identical to full."
        ),
        build_encoder=lambda: TokenReadEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_tokenread_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )
)

# --- Factorial arm (informed init in the presence of the transformer) --
register(
    ArchitectureSpec(
        name="full-uninformed",
        description="full minus informed card-embedding initialization.",
        build_encoder=lambda: CardReasoningEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        ),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    )
)


def available_architectures() -> list:
    return sorted(ARCHITECTURES.keys())


def get_spec(arch: "str | ArchitectureSpec") -> ArchitectureSpec:
    if isinstance(arch, ArchitectureSpec):
        return arch
    try:
        return ARCHITECTURES[arch]
    except KeyError:
        raise KeyError(
            f"Unknown architecture '{arch}'. Available: {available_architectures()}"
        )


# Placeholder for typing convenience in callers that pass mappings around.
ActorMappings = Dict[str, Any]
