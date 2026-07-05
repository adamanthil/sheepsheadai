"""Architecture registry for PPO network variants.

Each named ``ArchitectureSpec`` bundles the three network factories
(encoder / actor / critic) that ``PPOAgent`` uses to build itself. The
default ``full`` spec constructs byte-for-byte the same networks as the
pre-registry code, so existing checkpoints and training behavior are
unchanged; every other spec is an ablation rung or a future variant.

The registry exists to support controlled architecture ablations: each
adjacent rung of the ladder removes exactly one historical addition
(auxiliary heads, transformer card reasoning, informed embedding
initialization, the card-token pipeline itself), so paired training runs
measure that component's contribution directly. Multi-seed replication is
required for any conclusion — deep-RL comparisons are notoriously
seed-sensitive (Henderson et al. 2018, "Deep Reinforcement Learning that
Matters", arXiv:1709.06560), and controlled/equal-footing evaluation is the
difference between measuring architectures and measuring tuning effort
(Melis et al. 2018, arXiv:1707.05589).

Adding a variant = adding one ``ArchitectureSpec`` entry; the trainers,
checkpoint metadata (``arch`` key), and ``ppo.load_agent`` pick it up by
name. Workers and subprocesses receive the architecture *name*, never the
spec object.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import CardEmbeddingConfig, CardReasoningEncoder


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
    has_aux_heads: bool = True


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
    ):
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
# Legacy-style one-hot state representation (onehot-ff baseline)
# ---------------------------------------------------------------------------

# Card ids are 0 (pad), 1..32 (deck), 33 (UNDER token) -> 34 slots per card
# field. The vector is a pure function of Player.get_state_dict(); no game
# code is touched.
_N_CARD = 34
_ONEHOT_SECTIONS = (
    ("hand_multi_hot", _N_CARD),
    ("trick_card_onehots", 5 * _N_CARD),
    ("trick_is_picker", 5),
    ("trick_is_partner_known", 5),
    ("blind_multi_hot", _N_CARD),
    ("bury_multi_hot", _N_CARD),
    ("called_card_onehot", _N_CARD),
    ("header_scalars", 6),  # partner_mode, is_leaster, play_started,
    #                         alone_called, called_under, current_trick/6
    ("picker_rel_onehot", 6),
    ("partner_rel_onehot", 6),
    ("leader_rel_onehot", 6),
    ("picker_position_onehot", 6),
)
ONEHOT_STATE_DIM = sum(size for _, size in _ONEHOT_SECTIONS)


def _scalar(state: Dict[str, Any], key: str) -> int:
    v = state.get(key, 0)
    arr = np.asarray(v).reshape(-1)
    return int(arr[0]) if arr.size else 0


def build_onehot_state(state: Dict[str, Any]) -> np.ndarray:
    """Flatten a get_state_dict() observation into the legacy one-hot vector."""
    out = np.zeros(ONEHOT_STATE_DIM, dtype=np.float32)
    off = 0

    def multi_hot(ids) -> None:
        for cid in np.asarray(ids).reshape(-1):
            cid = int(cid)
            if cid > 0:
                out[off + cid] = 1.0

    # Hand
    multi_hot(state.get("hand_ids", ()))
    off += _N_CARD
    # Current trick: one card slot per relative seat
    trick_ids = np.asarray(state.get("trick_card_ids", np.zeros(5))).reshape(-1)
    for i in range(5):
        cid = int(trick_ids[i]) if i < trick_ids.size else 0
        if cid > 0:
            out[off + i * _N_CARD + cid] = 1.0
    off += 5 * _N_CARD
    for key in ("trick_is_picker", "trick_is_partner_known"):
        flags = np.asarray(state.get(key, np.zeros(5))).reshape(-1)
        out[off : off + min(5, flags.size)] = flags[:5]
        off += 5
    # Blind / bury (picker only; zeros otherwise, like the token encoder)
    multi_hot(state.get("blind_ids", ()))
    off += _N_CARD
    multi_hot(state.get("bury_ids", ()))
    off += _N_CARD
    # Called card
    called = _scalar(state, "called_card_id")
    if called > 0:
        out[off + called] = 1.0
    off += _N_CARD
    # Header scalars
    out[off + 0] = _scalar(state, "partner_mode")
    out[off + 1] = _scalar(state, "is_leaster")
    out[off + 2] = _scalar(state, "play_started")
    out[off + 3] = _scalar(state, "alone_called")
    out[off + 4] = _scalar(state, "called_under")
    out[off + 5] = _scalar(state, "current_trick") / 6.0
    off += 6
    # Relative seats (0 = unknown, 1..5 = seats) as one-hots of width 6
    for key in ("picker_rel", "partner_rel", "leader_rel", "picker_position"):
        v = _scalar(state, key)
        if 0 <= v < 6:
            out[off + v] = 1.0
        off += 6
    return out


class OneHotFeedForwardEncoder(nn.Module):
    """Legacy-style baseline: flat one-hot state -> MLP -> GRU memory.

    Presents the same interface contract as CardReasoningEncoder
    (encode_batch / encode_sequences returning 'features' (…,256) and
    'memory_out' (B,256)) but emits no per-card tokens — actors that need
    hand_tokens cannot be paired with this encoder.
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ONEHOT_STATE_DIM, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )
        self.memory_gru = nn.GRUCell(256, 256)
        self.fuse = nn.Linear(512, 256)
        self.feature_norm = nn.LayerNorm(256)

        # Interface contract with PPOAgent / actors: no card embedding table.
        self.card = None
        self.d_card_dim = 0
        self.d_token_dim = 0
        self.d_model = 256

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def param_groups(self, base_lr: float, card_lr_scale: float = 0.2):
        # No embedding table to scale; a single group keeps optimizer wiring
        # uniform with the token encoder's param_groups contract.
        return [{"params": list(self.parameters()), "lr": base_lr}]

    def encode_batch(
        self,
        batch: List[Dict[str, Any]],
        memory_in: "torch.Tensor | None" = None,
        device: "torch.device | None" = None,
    ) -> Dict[str, torch.Tensor]:
        B = len(batch)
        if B:
            x_np = np.stack([build_onehot_state(s) for s in batch])
        else:
            x_np = np.zeros((0, ONEHOT_STATE_DIM), dtype=np.float32)
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        if memory_in is None:
            memory_in = torch.zeros((B, 256), dtype=torch.float32, device=device)
        elif device is not None:
            memory_in = memory_in.to(device)
        h = self.mlp(x)
        memory_out = self.memory_gru(h, memory_in)
        features = self.feature_norm(self.fuse(torch.cat([h, memory_out], dim=1)))
        return {"features": features, "memory_out": memory_out}

    def encode_sequences(
        self,
        sequences: List[List[Dict[str, Any]]],
        memory_in: "torch.Tensor | None" = None,
        device: "torch.device | None" = None,
    ) -> Dict[str, torch.Tensor]:
        B = len(sequences)
        T = max((len(seq) for seq in sequences), default=1)
        features_out = torch.zeros((B, T, 256), dtype=torch.float32, device=device)
        if memory_in is None:
            memory_state = torch.zeros((B, 256), dtype=torch.float32, device=device)
        else:
            memory_state = memory_in.to(device) if device is not None else memory_in
        for t in range(T):
            batch_t = []
            for b in range(B):
                if t < len(sequences[b]):
                    batch_t.append(sequences[b][t])
                else:
                    batch_t.append(sequences[b][-1] if sequences[b] else {})
            if not batch_t:
                continue
            out = self.encode_batch(batch_t, memory_in=memory_state, device=device)
            features_out[:, t, :] = out["features"]
            memory_state = out["memory_out"]
        return {"features": features_out, "memory_out": memory_state}


class FlatHeadActorNetwork(nn.Module):
    """Legacy-style actor: per-phase flat linear heads over global action ids.

    Implements the same four-method surface as MultiHeadRecurrentActorNetwork
    (forward / forward_with_logits / _build_logits_from_features /
    set_temperatures) but ignores hand_ids, card embeddings, and hand tokens —
    card actions are scored by absolute action id, not by hand slot.
    """

    def __init__(self, action_size, action_groups):
        super().__init__()
        self.action_size = action_size
        self.action_groups = action_groups

        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(256, len(indices))
                for name, indices in action_groups.items()
            }
        )
        for name, indices in action_groups.items():
            self.register_buffer(
                f"_idx_{name}",
                torch.tensor(sorted(indices), dtype=torch.long),
                persistent=False,
            )

        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def set_temperatures(self, pick=None, partner=None, bury=None, play=None):
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids=None,
        card_embedding=None,
        hand_tokens=None,
        action_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        del hand_ids, card_embedding, hand_tokens  # unused by flat heads
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=actor_features.device)
        feat = self.actor_adapter(actor_features)
        temps = {
            "pick": self.temperature_pick,
            "partner": self.temperature_partner,
            "bury": self.temperature_bury,
            "play": self.temperature_play,
        }
        for name in self.action_groups:
            idx = getattr(self, f"_idx_{name}")
            logits[:, idx] = self.heads[name](feat) / max(temps[name], 1e-6)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)
        return logits

    def forward(self, encoder_out, action_mask, hand_ids, card_embedding):
        logits = self._build_logits_from_features(
            actor_features=encoder_out["features"],
            action_mask=action_mask,
        )
        return F.softmax(logits, dim=-1)

    def forward_with_logits(self, encoder_out, action_mask, hand_ids, card_embedding):
        logits = self._build_logits_from_features(
            actor_features=encoder_out["features"],
            action_mask=action_mask,
        )
        return F.softmax(logits, dim=-1), logits


# ---------------------------------------------------------------------------
# Shared factory helpers
# ---------------------------------------------------------------------------


def _pointer_actor(action_size, action_groups, encoder, mappings):
    from ppo import MultiHeadRecurrentActorNetwork

    return MultiHeadRecurrentActorNetwork(
        action_size,
        action_groups,
        d_card=encoder.d_card_dim,
        d_token=encoder.d_token_dim,
        d_model=getattr(encoder, "d_model", 256),
        **mappings,
    )


def _aux_critic(encoder):
    from ppo import RecurrentCriticNetwork

    return RecurrentCriticNetwork(
        d_card=encoder.d_card_dim, d_model=getattr(encoder, "d_model", 256)
    )


def _no_aux_critic(encoder):
    from ppo import RecurrentCriticNetwork

    d_card = getattr(encoder, "d_card_dim", 0) or None
    return RecurrentCriticNetwork(
        d_card=d_card,
        use_aux_heads=False,
        d_model=getattr(encoder, "d_model", 256),
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES: Dict[str, ArchitectureSpec] = {
    "full": ArchitectureSpec(
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
    ),
    # --- Ablation ladder (reverse-historical, cumulative) -----------------
    "no-aux": ArchitectureSpec(
        name="no-aux",
        description="full minus the critic's auxiliary heads.",
        build_encoder=lambda: CardReasoningEncoder(card_config=CardEmbeddingConfig()),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    ),
    "no-transformer": ArchitectureSpec(
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
    ),
    "no-transformer-uninformed": ArchitectureSpec(
        name="no-transformer-uninformed",
        description="no-transformer minus informed card-embedding initialization.",
        build_encoder=lambda: PooledMemoryEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        ),
        build_actor=_pointer_actor,
        build_critic=_no_aux_critic,
        has_aux_heads=False,
    ),
    "onehot-ff": ArchitectureSpec(
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
    ),
    # --- Capacity sweep around `full` (one knob per variant) ---------------
    "full-dtok32": _full_size_variant(
        "full-dtok32",
        "full with d_token 64 -> 32 (transformer width /2).",
        d_token=32,
    ),
    "full-dtok128": _full_size_variant(
        "full-dtok128",
        "full with d_token 64 -> 128 (transformer width x2).",
        d_token=128,
    ),
    "full-layers2": _full_size_variant(
        "full-layers2",
        "full with 2 reasoning layers (depth /2).",
        n_reasoning_layers=2,
    ),
    "full-layers6": _full_size_variant(
        "full-layers6",
        "full with 6 reasoning layers (depth x1.5).",
        n_reasoning_layers=6,
    ),
    "full-dmodel128": _full_size_variant(
        "full-dmodel128",
        "full with d_model 256 -> 128 (trunk/memory/pool width /2).",
        d_model=128,
    ),
    "full-dmodel512": _full_size_variant(
        "full-dmodel512",
        "full with d_model 256 -> 512 (trunk/memory/pool width x2).",
        d_model=512,
    ),
    # --- Factorial arm (informed init in the presence of the transformer) --
    "full-uninformed": ArchitectureSpec(
        name="full-uninformed",
        description="full minus informed card-embedding initialization.",
        build_encoder=lambda: CardReasoningEncoder(
            card_config=CardEmbeddingConfig(use_informed_init=False)
        ),
        build_actor=_pointer_actor,
        build_critic=_aux_critic,
        has_aux_heads=True,
    ),
}


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
