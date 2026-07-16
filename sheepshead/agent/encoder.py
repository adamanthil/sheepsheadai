import torch
import torch.nn as nn
from typing import List, Dict, Any
import itertools
from dataclasses import dataclass
from sheepshead import DECK_IDS, TRUMP, UNDER_CARD_ID
from sheepshead.agent.token_layout import (
    BASE_TYPE_COUNT,
    BLIND_TOKENS,
    BLIND_TYPE_ID,
    BURY_TOKENS,
    BURY_TYPE_ID,
    CONTEXT_TOKEN,
    HAND_TOKENS,
    HAND_TYPE_ID,
    TRICK_TOKENS,
    TRICK_TYPE_ID,
)


PAD_CARD_ID = 0


@dataclass
class CardEmbeddingConfig:
    use_informed_init: bool = True
    d_card: int = 16  # requires >= 10 for informed initialization
    max_points: float = 11.0


class TransformerCardReasoning(nn.Module):
    """Allows cards to reason about each other through self-attention before pooling."""

    def __init__(self, d_token: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_token = d_token

        # Multi-head self-attention layers
        self.attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_token, num_heads=n_heads, batch_first=True, dropout=0.0
                )
                for _ in range(n_layers)
            ]
        )

        # Feed-forward networks after each attention layer
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_token, d_token * 2),
                    nn.SiLU(),
                    nn.Linear(d_token * 2, d_token),
                )
                for _ in range(n_layers)
            ]
        )

        # Layer norms
        self.ln_attn = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.ln_ffn = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, d_token) where N is total cards
            mask: (B, N) boolean, True = valid card

        Returns:
            tokens: (B, N, d_token) after cross-card reasoning
        """
        # Convert mask to attention mask format (True = ignore)
        attn_mask = ~mask  # (B, N)

        for attn, ffn, ln1, ln2 in zip(
            self.attn_layers, self.ffn_layers, self.ln_attn, self.ln_ffn
        ):
            # Self-attention with residual
            attn_out, _ = attn(
                tokens, tokens, tokens, key_padding_mask=attn_mask, need_weights=False
            )
            tokens = ln1(tokens + attn_out)

            # FFN with residual
            ffn_out = ffn(tokens)
            tokens = ln2(tokens + ffn_out)

        return tokens


class AttentionPool(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Multi-query + multi-head attention pooling:
        #   - Learn M query tokens (glimpses)
        #   - Attend (with H heads) over the bag tokens
        #   - Flatten the M outputs and project to d_out
        #
        # Keep the same public API as the original AttentionPool.
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.n_queries = 4
        self.n_heads = 4
        if self.d_in % self.n_heads != 0:
            raise ValueError(
                f"AttentionPool: d_in={self.d_in} must be divisible by n_heads={self.n_heads}."
            )
        self.query = nn.Parameter(torch.randn(self.n_queries, self.d_in))  # (M, d_in)
        self.mha = nn.MultiheadAttention(self.d_in, self.n_heads, batch_first=True)
        self.proj = nn.Linear(self.n_queries * self.d_in, self.d_out)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, d_in); mask: (B, N) True=keep
        if tokens.numel() == 0:
            return torch.zeros(
                (mask.size(0), self.d_out), device=mask.device, dtype=tokens.dtype
            )

        B = tokens.size(0)

        # nn.MultiheadAttention expects key_padding_mask True = ignore.
        # Guard against rows where all tokens are masked (can produce NaNs).
        no_valid = ~mask.any(dim=1)  # (B,)
        if no_valid.any():
            tokens = tokens.clone()
            mask = mask.clone()
            tokens[no_valid, 0, :] = 0.0
            mask[no_valid, 0] = True

        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_in)
        attn_out, _ = self.mha(
            q,
            tokens,
            tokens,
            key_padding_mask=~mask,
            need_weights=False,
        )  # (B, M, d_in)
        pooled = self.proj(
            attn_out.reshape(B, self.n_queries * self.d_in)
        )  # (B, d_out)
        if no_valid.any():
            pooled = torch.where(
                no_valid.view(B, 1),
                torch.zeros_like(pooled),
                pooled,
            )
        return pooled


class CardReasoningEncoder(nn.Module):
    """Token-centric encoder with transformer reasoning and memory.

    Input: dict observation (from get_state_dict)
    Output: dict with:
      - 'features': (B, 256) fused feature vector (for heads)
      - 'hand_tokens': (B, 8, d_token) reasoning-enhanced (post-attention) hand tokens
      - 'context_token': (B, d_token) post-attention context
      - 'memory_out': (B, 256) updated memory state

    Observation dict fields expected per sample:
      - partner_mode, is_leaster, play_started, current_trick,
        alone_called, called_card_id, called_under,
        picker_rel, partner_rel, leader_rel, picker_position
      - hand_ids (8,), blind_ids (2,), bury_ids (2,)
      - trick_card_ids (5,), trick_is_picker (5,), trick_is_partner_known (5,)
    """

    def __init__(
        self,
        d_card: int = 16,
        d_token: int = 64,
        card_config: CardEmbeddingConfig | None = None,
        n_reasoning_heads: int = 4,
        n_reasoning_layers: int = 4,
        d_model: int = 256,
    ):
        super().__init__()
        # Allow config to override d_card
        if card_config is not None:
            d_card = card_config.d_card

        # Expose configured dimensions for downstream modules
        self.d_card_dim = int(d_card)
        self.d_token_dim = int(d_token)
        # Feature/memory width; pool widths scale with it (256 -> 64/64/32/32,
        # matching the historical constants exactly).
        self.d_model = int(d_model)
        pool_big = self.d_model // 4
        pool_small = self.d_model // 8

        # Embeddings
        self.card = nn.Embedding(34, d_card, padding_idx=PAD_CARD_ID)  # 0..33
        self.seat = nn.Embedding(6, 4)  # 0=unknown, 1-5=relative positions
        self.role = nn.Embedding(4, 4)  # 0=none, 1=picker, 2=partner, 3=both
        self.card_type = nn.Embedding(
            BASE_TYPE_COUNT, d_token
        )  # 0=context, 1=memory, 2=hand, 3=trick, 4=blind, 5=bury

        # Optional informed initialization for card embeddings
        if card_config and card_config.use_informed_init:
            init_table = self._build_informed_card_init(
                d_card=d_card,
                max_points=card_config.max_points,
            )
            with torch.no_grad():
                self.card.weight.data.copy_(init_table)

        # Token projections
        self.context_mlp = nn.Sequential(
            nn.Linear(10 + d_card, d_token),  # header scalars + called_card_emb
            nn.SiLU(),
        )
        self.memory_in_proj = nn.Linear(self.d_model, d_token)
        self.token_mlp_hand = nn.Sequential(
            nn.Linear(d_card + 4, d_token),  # card + actor_role
            nn.SiLU(),
        )
        self.token_mlp_trick = nn.Sequential(
            nn.Linear(d_card + 4 + 4, d_token),  # card + seat + role
            nn.SiLU(),
        )
        self.token_mlp_simple = nn.Sequential(
            nn.Linear(d_card, d_token),  # blind/bury
            nn.SiLU(),
        )

        # Card reasoning via transformer
        self.card_reasoner = TransformerCardReasoning(
            d_token=d_token, n_heads=n_reasoning_heads, n_layers=n_reasoning_layers
        )

        # Pools per bag
        self.pool_hand = AttentionPool(d_token, pool_big)
        self.pool_trick = AttentionPool(d_token, pool_big)
        self.pool_blind = AttentionPool(d_token, pool_small)
        self.pool_bury = AttentionPool(d_token, pool_small)

        # Memory update (GRU cell)
        self.memory_gru = nn.GRUCell(d_token, self.d_model)

        # Fused feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(2 * pool_big + 2 * pool_small + d_token, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def _build_informed_card_init(self, d_card: int, max_points: float) -> torch.Tensor:
        """
        Build an informed (34, d_card) init matrix with layout (requires d_card >= 10):
          [0]  suit_trump (1.0 if trump),
          [1]  suit_clubs,
          [2]  suit_spades,
          [3]  suit_hearts,
          [4]  rank_trump (0..1),
          [5]  rank_clubs (0..1),
          [6]  rank_spades (0..1),
          [7]  rank_hearts (0..1),
          [8]  points_norm (0..1),
          [9]  under_flag (1.0 for UNDER, else 0.0),
          [10..] zeros (reserved for learning offsets)
        """
        if d_card < 10:
            raise ValueError(
                f"d_card must be >= 10 to encode initialization priors. Got {d_card}"
            )

        init = torch.zeros((34, d_card), dtype=torch.float32)

        # Suit channels
        SUIT_T, SUIT_C, SUIT_S, SUIT_H = 0, 1, 2, 3
        RANK_T, RANK_C, RANK_S, RANK_H = 4, 5, 6, 7
        DIM_POINTS = 8
        DIM_UNDER_FLAG = 9

        FAIL_ORDER = ["A", "10", "K", "9", "8", "7"]
        trump_strength = {
            card: (len(TRUMP) - i) / len(TRUMP) for i, card in enumerate(TRUMP)
        }
        points_map = {"Q": 3, "J": 2, "A": 11, "10": 10, "K": 4, "9": 0, "8": 0, "7": 0}

        # Real cards: 1..32
        for card, cid in DECK_IDS.items():
            row = torch.zeros(d_card, dtype=torch.float32)
            is_trump = card in TRUMP
            rank = card[:-1]
            suit = card[-1]

            # Suit one-hot (Trump isolated)
            if is_trump:
                row[SUIT_T] = 1.0
            else:
                if suit == "C":
                    row[SUIT_C] = 1.0
                elif suit == "S":
                    row[SUIT_S] = 1.0
                elif suit == "H":
                    row[SUIT_H] = 1.0

            # Rank strength (per-suit channel; trump rank separate from fail ranks)
            if is_trump:
                row[RANK_T] = float(trump_strength[card])
            else:
                pos = (
                    FAIL_ORDER.index(rank)
                    if rank in FAIL_ORDER
                    else len(FAIL_ORDER) - 1
                )
                norm = float((len(FAIL_ORDER) - 1 - pos) / (len(FAIL_ORDER) - 1))
                if suit == "C":
                    row[RANK_C] = norm
                elif suit == "S":
                    row[RANK_S] = norm
                elif suit == "H":
                    row[RANK_H] = norm

            # Points normalized
            row[DIM_POINTS] = float(points_map.get(rank, 0) / max_points)

            init[cid] = row

        # PAD remains zeros at index 0
        # UNDER (33): distinct via under_flag only
        under_row = torch.zeros(d_card, dtype=torch.float32)
        under_row[DIM_UNDER_FLAG] = 1.0
        init[UNDER_CARD_ID] = under_row

        return init

    def param_groups(self, base_lr: float, card_lr_scale: float = 0.1):
        """
        Return optimizer parameter groups for the encoder:
          - card embedding at scaled LR (e.g., 0.1x)
          - all other encoder params at base LR
        """
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
            self.pool_hand.parameters(),
            self.pool_trick.parameters(),
            self.pool_blind.parameters(),
            self.pool_bury.parameters(),
            self.memory_gru.parameters(),
            self.feature_proj.parameters(),
        )
        return [
            {"params": self.card.parameters(), "lr": base_lr * card_lr_scale},
            {"params": other_params, "lr": base_lr},
        ]

    @staticmethod
    def _stack_uint8(batch: List[Any], key: str, length: int) -> torch.Tensor:
        arr = [torch.as_tensor(s[key], dtype=torch.long) for s in batch]
        out = torch.stack(arr, dim=0)
        # Ensure shape
        if out.dim() == 1:
            out = out.view(-1, length)
        return out

    @staticmethod
    def _stack_scalar(batch: List[Any], key: str) -> torch.Tensor:
        vals = [int(batch[i][key]) for i in range(len(batch))]
        return torch.as_tensor(vals, dtype=torch.float32).view(-1, 1)

    def _embed_hand(
        self, ids: torch.Tensor, actor_role_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed hand cards with actor role conditioning.

        Args:
            ids: (B, 8) card IDs
            actor_role_id: (B,) role ID for the acting player (0=none, 1=picker, 2=partner, 3=both)

        Returns:
            tokens: (B, 8, d_token)
            mask: (B, 8) boolean
        """
        mask = ids.ne(PAD_CARD_ID)
        c = self.card(ids)  # (B, 8, d_card)
        B, N = ids.size(0), ids.size(1)
        role_idx = actor_role_id.view(B, 1).expand(B, N)  # (B, 8)
        r = self.role(role_idx)  # (B, 8, 4)
        tok = torch.cat([c, r], dim=-1)
        tok = self.token_mlp_hand(tok)
        return tok, mask

    def _embed_simple_bag(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed blind/bury cards (no role conditioning)."""
        mask = ids.ne(PAD_CARD_ID)
        tok = self.card(ids)
        tok = self.token_mlp_simple(tok)
        return tok, mask

    def _embed_trick(
        self,
        card_ids: torch.Tensor,
        picker_bits: torch.Tensor,
        partner_bits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed trick cards with seat and role information."""
        B = card_ids.size(0)
        seat_rel = (
            torch.arange(1, 6, device=card_ids.device, dtype=torch.long)
            .view(1, 5)
            .expand(B, 5)
        )
        role_ids = picker_bits.long() * 1 + partner_bits.long() * 2
        mask = card_ids.ne(PAD_CARD_ID)
        c = self.card(card_ids)
        s = self.seat(seat_rel)
        r = self.role(role_ids)
        tok = torch.cat([c, s, r], dim=-1)
        tok = self.token_mlp_trick(tok)
        return tok, mask

    def encode_batch(
        self,
        batch: List[Dict[str, Any]],
        memory_in: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode a batch of observations with memory.

        Args:
            batch: List of observation dicts from get_state_dict
            memory_in: (B, 256) previous memory state, or None to use zeros
            device: Target device

        Returns:
            dict with:
                'features': (B, 256) fused feature vector
                'hand_tokens': (B, 8, d_token) reasoning-enhanced hand tokens
                'context_token': (B, d_token) post-attention context
                'memory_out': (B, 256) updated memory state
        """

        # Move to device lazily
        def to_device(x: torch.Tensor) -> torch.Tensor:
            return x.to(device) if device is not None else x

        B = len(batch)

        # Initialize memory if not provided
        if memory_in is None:
            memory_in = torch.zeros(
                (B, self.d_model), dtype=torch.float32, device=device
            )
        else:
            memory_in = to_device(memory_in)

        # 1. Build header scalar + called_card_emb → context_token
        header_fields = [
            "partner_mode",
            "is_leaster",
            "play_started",
            "current_trick",
            "alone_called",
            "called_under",
            "picker_rel",
            "partner_rel",
            "leader_rel",
            "picker_position",
        ]
        header_cols = [self._stack_scalar(batch, k) for k in header_fields]
        header_scalar = torch.cat(header_cols, dim=1)
        header_scalar = to_device(header_scalar)
        # Normalize header scalars
        norm = torch.tensor(
            [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
            dtype=header_scalar.dtype,
            device=header_scalar.device,
        )
        header_scalar = header_scalar / norm

        # Called card embedding
        called_ids = torch.as_tensor(
            [int(s["called_card_id"]) for s in batch], dtype=torch.long
        )
        called_ids = to_device(called_ids)
        called_emb = self.card(called_ids)  # (B, d_card)

        # Build context token from header + called card
        context_tok = self.context_mlp(
            torch.cat([header_scalar, called_emb], dim=1)
        )  # (B, d_token)

        # 2. Project memory_in → memory_token
        memory_tok = self.memory_in_proj(memory_in)  # (B, d_token)

        # 3. Derive actor role from picker_rel and partner_rel
        picker_rel_raw = torch.as_tensor(
            [int(s["picker_rel"]) for s in batch], dtype=torch.long
        )
        partner_rel_raw = torch.as_tensor(
            [int(s["partner_rel"]) for s in batch], dtype=torch.long
        )
        picker_rel_raw = to_device(picker_rel_raw)
        partner_rel_raw = to_device(partner_rel_raw)
        actor_role_id = (
            picker_rel_raw.eq(1).long() * 1 + partner_rel_raw.eq(1).long() * 2
        )  # 0=none, 1=picker, 2=partner, 3=both

        # 4. Build card tokens
        hand_ids = to_device(self._stack_uint8(batch, "hand_ids", 8))
        blind_ids = to_device(self._stack_uint8(batch, "blind_ids", 2))
        bury_ids = to_device(self._stack_uint8(batch, "bury_ids", 2))
        trick_card_ids = to_device(self._stack_uint8(batch, "trick_card_ids", 5))
        trick_is_picker = to_device(
            self._stack_uint8(batch, "trick_is_picker", 5)
        ).bool()
        trick_is_partner_known = to_device(
            self._stack_uint8(batch, "trick_is_partner_known", 5)
        ).bool()

        hand_tok, hand_mask = self._embed_hand(hand_ids, actor_role_id)
        blind_tok, blind_mask = self._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = self._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = self._embed_trick(
            trick_card_ids, trick_is_picker, trick_is_partner_known
        )

        # 5. Concatenate: [context, memory, hand×8, trick×5, blind×2, bury×2] = 19 tokens
        device_actual = hand_tok.device
        all_tokens = torch.cat(
            [
                context_tok.unsqueeze(1),  # (B, 1, d_token)
                memory_tok.unsqueeze(1),  # (B, 1, d_token)
                hand_tok,  # (B, 8, d_token)
                trick_tok,  # (B, 5, d_token)
                blind_tok,  # (B, 2, d_token)
                bury_tok,  # (B, 2, d_token)
            ],
            dim=1,
        )  # (B, 19, d_token)

        all_mask = torch.cat(
            [
                torch.ones(
                    (B, 1), dtype=torch.bool, device=device_actual
                ),  # context always valid
                torch.ones(
                    (B, 1), dtype=torch.bool, device=device_actual
                ),  # memory always valid
                hand_mask,
                trick_mask,
                blind_mask,
                bury_mask,
            ],
            dim=1,
        )  # (B, 19)

        # 6. Add card_type embeddings
        type_ids = torch.cat(
            [
                torch.zeros(
                    (B, 1), dtype=torch.long, device=device_actual
                ),  # context = 0
                torch.ones(
                    (B, 1), dtype=torch.long, device=device_actual
                ),  # memory = 1
                torch.full(
                    (B, 8), HAND_TYPE_ID, dtype=torch.long, device=device_actual
                ),  # hand = 2
                torch.full(
                    (B, 5), TRICK_TYPE_ID, dtype=torch.long, device=device_actual
                ),  # trick = 3
                torch.full(
                    (B, 2), BLIND_TYPE_ID, dtype=torch.long, device=device_actual
                ),  # blind = 4
                torch.full(
                    (B, 2), BURY_TYPE_ID, dtype=torch.long, device=device_actual
                ),  # bury = 5
            ],
            dim=1,
        )  # (B, 19)
        all_tokens = all_tokens + self.card_type(type_ids)

        # 7. Run transformer
        all_tokens = self.card_reasoner(all_tokens, all_mask)

        # 8. Extract post-attention tokens
        context_out = all_tokens[:, CONTEXT_TOKEN, :]  # (B, d_token)
        hand_tok_out = all_tokens[:, HAND_TOKENS, :]  # (B, 8, d_token)
        trick_tok_out = all_tokens[:, TRICK_TOKENS, :]
        blind_tok_out = all_tokens[:, BLIND_TOKENS, :]
        bury_tok_out = all_tokens[:, BURY_TOKENS, :]

        # 9-11. Pool bags, update memory, fuse features. Overridable seam:
        # pool-free variants (PerceiverEncoder in architectures.encoders) replace
        # this whole tail; the base implementation is byte-identical to the
        # historical inline code.
        return self._pool_fuse_update(
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
        )

    def _pool_fuse_update(
        self,
        context_out: torch.Tensor,
        hand_tok_out: torch.Tensor,
        hand_mask: torch.Tensor,
        trick_tok_out: torch.Tensor,
        trick_mask: torch.Tensor,
        blind_tok_out: torch.Tensor,
        blind_mask: torch.Tensor,
        bury_tok_out: torch.Tensor,
        bury_mask: torch.Tensor,
        memory_in: torch.Tensor,
        all_tokens: torch.Tensor,
        all_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Overridable seam: post-reasoning tokens -> encoder output dict.

        Called once per encode step with the post-reasoning per-bag tokens
        (+ masks), the previous memory state, and the full 19-token set.
        The base implementation pools each bag and delegates to
        _fuse_and_update_memory; pool-free architectures (PerceiverEncoder,
        PerceiverCtxMemEncoder, SharedReadoutEncoder in architectures.encoders)
        override THIS seam to skip the bag pools entirely.

        Contract for overrides: return the encoder output dict
        ('features'/'memory_out' (B, d_model) at minimum), consume no RNG,
        and construct no modules. The base implementation is byte-identical
        to the historical inline code.
        """
        # 9. Pool bags
        hand_vec = self.pool_hand(hand_tok_out, hand_mask)
        trick_vec = self.pool_trick(trick_tok_out, trick_mask)
        blind_vec = self.pool_blind(blind_tok_out, blind_mask)
        bury_vec = self.pool_bury(bury_tok_out, bury_mask)

        # 10-11. Update memory + fuse features (the narrower seam below).
        return self._fuse_and_update_memory(
            context_out,
            hand_tok_out,
            hand_vec,
            trick_vec,
            blind_vec,
            bury_vec,
            memory_in,
            all_tokens=all_tokens,
            all_mask=all_mask,
        )

    def _fuse_and_update_memory(
        self,
        context_out: torch.Tensor,
        hand_tok_out: torch.Tensor,
        hand_vec: torch.Tensor,
        trick_vec: torch.Tensor,
        blind_vec: torch.Tensor,
        bury_vec: torch.Tensor,
        memory_in: torch.Tensor,
        all_tokens: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Overridable seam: pooled bag vectors -> fused features + memory.

        The narrower seam inside _pool_fuse_update, for variants that keep
        the bag pools but change what the memory GRU consumes or what the
        output dict exposes (PooledMemoryEncoder drives the GRU with the
        fused features; TokenReadEncoder additionally emits
        'all_tokens'/'all_mask'). Same override contract as
        _pool_fuse_update: return the encoder output dict, no RNG,
        no module construction; the base implementation is byte-identical
        to the historical inline code.
        """
        # Update memory: memory_out = GRU(context_token_out, memory_in)
        memory_out = self.memory_gru(context_out, memory_in)  # (B, 256)

        # Fuse features
        features = self.feature_proj(
            torch.cat([hand_vec, trick_vec, blind_vec, bury_vec, context_out], dim=1)
        )

        return {
            "features": features,
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
        }

    def encode_sequences(
        self,
        sequences: List[List[Dict[str, Any]]],
        memory_in: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode sequences of observations with recurrent memory.

        Args:
            sequences: List of B sequences, each a list of observation dicts
            memory_in: (B, 256) initial memory state, or None for zeros
            device: Target device

        Returns:
            dict with:
                'features': (B, T, 256)
                'hand_tokens': (B, T, 8, d_token)
                'memory_out': (B, 256) final memory state after processing all timesteps
        """
        B = len(sequences)
        T = max((len(seq) for seq in sequences), default=1)

        # Initialize outputs
        features_out = torch.zeros(
            (B, T, self.d_model), dtype=torch.float32, device=device
        )
        hand_tokens_out = torch.zeros(
            (B, T, 8, self.d_token_dim), dtype=torch.float32, device=device
        )
        # Allocated lazily on the first timestep if the encoder variant emits
        # the full token set (TokenReadEncoder); stays None for the base class.
        all_tokens_out = None
        all_mask_out = None

        # Initialize memory
        if memory_in is None:
            memory_state = torch.zeros(
                (B, self.d_model), dtype=torch.float32, device=device
            )
        else:
            memory_state = memory_in.to(device) if device is not None else memory_in

        # Process each timestep sequentially (memory updates across time)
        for t in range(T):
            # Gather batch for this timestep
            batch_t = []
            for b in range(B):
                if t < len(sequences[b]):
                    batch_t.append(sequences[b][t])
                else:
                    # Pad with empty observation (will be masked out)
                    batch_t.append(sequences[b][-1] if sequences[b] else {})

            if not batch_t:
                continue

            # Encode this timestep
            encoder_out = self.encode_batch(
                batch_t, memory_in=memory_state, device=device
            )

            # Store outputs
            features_out[:, t, :] = encoder_out["features"]
            hand_tokens_out[:, t, :, :] = encoder_out["hand_tokens"]
            step_all_tokens = encoder_out.get("all_tokens")
            if step_all_tokens is not None:
                if all_tokens_out is None:
                    n_all = step_all_tokens.size(1)
                    all_tokens_out = torch.zeros(
                        (B, T, n_all, self.d_token_dim),
                        dtype=torch.float32,
                        device=device,
                    )
                    all_mask_out = torch.zeros(
                        (B, T, n_all), dtype=torch.bool, device=device
                    )
                all_tokens_out[:, t, :, :] = step_all_tokens
                all_mask_out[:, t, :] = encoder_out["all_mask"]

            # Update memory for next timestep
            memory_state = encoder_out["memory_out"]

        out = {
            "features": features_out,
            "hand_tokens": hand_tokens_out,
            "memory_out": memory_state,
        }
        if all_tokens_out is not None:
            out["all_tokens"] = all_tokens_out
            out["all_mask"] = all_mask_out
        return out
