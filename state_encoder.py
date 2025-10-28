import torch
import torch.nn as nn
from typing import List, Dict, Any
import math
import itertools
from dataclasses import dataclass
from sheepshead import DECK_IDS, TRUMP


PAD_CARD_ID = 0
UNDER_CARD_ID = 33


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
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_token,
                num_heads=n_heads,
                batch_first=True,
                dropout=0.0
            )
            for _ in range(n_layers)
        ])

        # Feed-forward networks after each attention layer
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_token, d_token * 2),
                nn.SiLU(),
                nn.Linear(d_token * 2, d_token),
            )
            for _ in range(n_layers)
        ])

        # Layer norms
        self.ln_attn = nn.ModuleList([
            nn.LayerNorm(d_token) for _ in range(n_layers)
        ])
        self.ln_ffn = nn.ModuleList([
            nn.LayerNorm(d_token) for _ in range(n_layers)
        ])

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
                tokens, tokens, tokens,
                key_padding_mask=attn_mask,
                need_weights=False
            )
            tokens = ln1(tokens + attn_out)

            # FFN with residual
            ffn_out = ffn(tokens)
            tokens = ln2(tokens + ffn_out)

        return tokens


class AttentionPool(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_in))
        self.k = nn.Linear(d_in, d_in, bias=False)
        self.v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, d_in); mask: (B, N) True=keep
        if tokens.numel() == 0:
            return torch.zeros((mask.size(0), self.v.out_features), device=mask.device, dtype=tokens.dtype)

        k = self.k(tokens)
        v = self.v(tokens)
        q = self.q.view(1, 1, -1).expand(tokens.size(0), 1, -1)  # (B,1,d_in)
        att = torch.einsum('bnd,bqd->bnq', k, q).squeeze(-1)     # (B,N)
        # Scaled dot-product attention (stabilizes logits)
        att = att / math.sqrt(k.size(-1))

        # Safe masking (avoid all -inf causing NaNs)
        masked_att = att.masked_fill(~mask, -1e9)
        no_valid = ~mask.any(dim=1)
        masked_att = torch.where(
            no_valid.unsqueeze(-1),
            torch.zeros_like(masked_att),
            masked_att,
        )
        w = torch.softmax(masked_att, dim=-1)
        w = torch.where(
            no_valid.unsqueeze(-1),
            torch.zeros_like(w),
            w,
        )
        w = w.unsqueeze(-1)
        return (w * v).sum(dim=1)


class StateEncoder(nn.Module):
    """Encodes dict-based Sheepshead observations into a fixed 256-d feature.

    Observation dict fields expected per sample:
      - partner_mode, is_leaster, play_started, current_trick,
        alone_called, called_card_id, called_under,
        picker_rel, partner_rel, leader_rel, picker_position
      - hand_ids (8,), blind_ids (2,), bury_ids (2,)
      - trick_card_ids (5,), trick_is_picker (5,), trick_is_partner_known (5,)
    """

    def __init__(self, d_card: int = 8, d_seat: int = 4, d_role: int = 4, d_token: int = 32,
                 card_config: CardEmbeddingConfig | None = None,
                 n_reasoning_heads: int = 4, n_reasoning_layers: int = 2):
        super().__init__()
        # Allow config to override d_card and control init
        if card_config is not None:
            d_card = card_config.d_card

        # Expose configured dimensions for downstream modules
        self.d_card_dim = int(d_card)
        self.d_token_dim = int(d_token)

        # Embeddings
        self.card = nn.Embedding(34, d_card, padding_idx=PAD_CARD_ID)  # 0..33
        self.seat = nn.Embedding(6, d_seat)   # 0..5 (0 unused in trick)
        self.role = nn.Embedding(4, d_role)   # 0 none, 1 picker, 2 partner, 3 both
        self.card_type = nn.Embedding(4, d_token)  # 0 hand, 1 trick, 2 blind, 3 bury

        # Optional informed initialization for card embeddings
        if card_config and card_config.use_informed_init:
            init_table = self._build_informed_card_init(
                d_card=d_card,
                max_points=card_config.max_points,
            )
            with torch.no_grad():
                self.card.weight.data.copy_(init_table)

        # Token projections
        self.token_mlp_trick = nn.Sequential(
            nn.Linear(d_card + d_seat + d_role, d_token),
            nn.SiLU(),
        )
        self.token_mlp_simple = nn.Sequential(
            nn.Linear(d_card, d_token),
            nn.SiLU(),
        )

        # Card reasoning via transformer
        self.card_reasoner = TransformerCardReasoning(
            d_token=d_token,
            n_heads=n_reasoning_heads,
            n_layers=n_reasoning_layers
        )

        # Pools per bag
        self.pool_hand = AttentionPool(d_token, 64)
        self.pool_trick = AttentionPool(d_token, 64)
        self.pool_blind = AttentionPool(d_token, 32)
        self.pool_bury = AttentionPool(d_token, 32)

        # Header → 64
        # Header inputs: 10 scalars (excluding called_card_id) + called_card embedding (d_card)
        self.header_mlp = nn.Sequential(
            nn.Linear(10 + d_card, 64),
            nn.SiLU(),
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
            raise ValueError(f"d_card must be >= 10 to encode initialization priors. Got {d_card}")

        init = torch.zeros((34, d_card), dtype=torch.float32)

        # Suit channels
        SUIT_T, SUIT_C, SUIT_S, SUIT_H = 0, 1, 2, 3
        RANK_T, RANK_C, RANK_S, RANK_H = 4, 5, 6, 7
        DIM_POINTS = 8
        DIM_UNDER_FLAG = 9

        FAIL_ORDER = ["A", "10", "K", "9", "8", "7"]
        trump_strength = {card: (len(TRUMP) - i) / len(TRUMP) for i, card in enumerate(TRUMP)}
        points_map = {'Q': 3, 'J': 2, 'A': 11, '10': 10, 'K': 4, '9': 0, '8': 0, '7': 0}

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
                if suit == 'C':
                    row[SUIT_C] = 1.0
                elif suit == 'S':
                    row[SUIT_S] = 1.0
                elif suit == 'H':
                    row[SUIT_H] = 1.0

            # Rank strength (per-suit channel; trump rank separate from fail ranks)
            if is_trump:
                row[RANK_T] = float(trump_strength[card])
            else:
                pos = FAIL_ORDER.index(rank) if rank in FAIL_ORDER else len(FAIL_ORDER) - 1
                norm = float((len(FAIL_ORDER) - 1 - pos) / (len(FAIL_ORDER) - 1))
                if suit == 'C':
                    row[RANK_C] = norm
                elif suit == 'S':
                    row[RANK_S] = norm
                elif suit == 'H':
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
            self.token_mlp_trick.parameters(),
            self.token_mlp_simple.parameters(),
            self.card_reasoner.parameters(),
            self.pool_hand.parameters(),
            self.pool_trick.parameters(),
            self.pool_blind.parameters(),
            self.pool_bury.parameters(),
            self.header_mlp.parameters(),
        )
        return [
            {'params': self.card.parameters(), 'lr': base_lr * card_lr_scale},
            {'params': other_params, 'lr': base_lr},
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

    def _embed_simple_bag(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ids: (B,N) longs
        mask = ids.ne(PAD_CARD_ID)
        tok = self.card(ids)
        tok = self.token_mlp_simple(tok)
        return tok, mask

    def _embed_trick(self, card_ids: torch.Tensor, picker_bits: torch.Tensor, partner_bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # shapes: (B,5)
        B = card_ids.size(0)
        seat_rel = torch.arange(1, 6, device=card_ids.device, dtype=torch.long).view(1, 5).expand(B, 5)
        role_ids = (picker_bits.long() * 1 + partner_bits.long() * 2)
        mask = card_ids.ne(PAD_CARD_ID)
        c = self.card(card_ids)
        s = self.seat(seat_rel)
        r = self.role(role_ids)
        tok = torch.cat([c, s, r], dim=-1)
        tok = self.token_mlp_trick(tok)
        return tok, mask

    def encode_batch(self, batch: List[Dict[str, Any]], device: torch.device | None = None) -> torch.Tensor:
        # Move to device lazily
        def to_device(x: torch.Tensor) -> torch.Tensor:
            return x.to(device) if device is not None else x

        # Header scalars (B, 10) — exclude called_card_id here, embedded separately
        header_fields = [
            'partner_mode', 'is_leaster', 'play_started', 'current_trick',
            'alone_called', 'called_under',
            'picker_rel', 'partner_rel', 'leader_rel', 'picker_position',
        ]
        header_cols = [self._stack_scalar(batch, k) for k in header_fields]
        header_scalar = torch.cat(header_cols, dim=1)
        header_scalar = to_device(header_scalar)
        # Normalize header scalars to comparable ranges
        # partner_mode,is_leaster,play_started (0/1); current_trick (0..6);
        # alone_called,called_under (0/1); picker_rel,partner_rel,leader_rel, picker_position (0..5)
        norm = torch.tensor([1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
                            dtype=header_scalar.dtype, device=header_scalar.device)
        header_scalar = header_scalar / norm

        # Called card embedding (B, d_card). Value 0 means unknown/no call → treat as PAD and embed 0.
        called_ids = torch.as_tensor([int(s['called_card_id']) for s in batch], dtype=torch.long)
        called_ids = to_device(called_ids)
        called_emb = self.card(called_ids)  # (B, d_card)

        # Bags
        hand_ids = to_device(self._stack_uint8(batch, 'hand_ids', 8))
        blind_ids = to_device(self._stack_uint8(batch, 'blind_ids', 2))
        bury_ids = to_device(self._stack_uint8(batch, 'bury_ids', 2))

        trick_card_ids = to_device(self._stack_uint8(batch, 'trick_card_ids', 5))
        trick_is_picker = to_device(self._stack_uint8(batch, 'trick_is_picker', 5)).bool()
        trick_is_partner_known = to_device(self._stack_uint8(batch, 'trick_is_partner_known', 5)).bool()

        # Embed all cards
        hand_tok, hand_mask = self._embed_simple_bag(hand_ids)
        blind_tok, blind_mask = self._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = self._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = self._embed_trick(trick_card_ids, trick_is_picker, trick_is_partner_known)

        # Add card type embeddings to distinguish card sources
        B = hand_tok.size(0)
        device_actual = hand_tok.device

        type_ids = torch.cat([
            torch.zeros((B, 8), dtype=torch.long, device=device_actual),  # hand = 0
            torch.ones((B, 5), dtype=torch.long, device=device_actual),   # trick = 1
            torch.full((B, 2), 2, dtype=torch.long, device=device_actual),  # blind = 2
            torch.full((B, 2), 3, dtype=torch.long, device=device_actual),  # bury = 3
        ], dim=1)  # (B, 17)

        # Concatenate all card tokens
        all_tokens = torch.cat([hand_tok, trick_tok, blind_tok, bury_tok], dim=1)  # (B, 17, d_token)
        all_mask = torch.cat([hand_mask, trick_mask, blind_mask, bury_mask], dim=1)  # (B, 17)

        # Add type embeddings
        all_tokens = all_tokens + self.card_type(type_ids)

        # Apply transformer reasoning - cards attend to each other
        all_tokens = self.card_reasoner(all_tokens, all_mask)

        # Split back into separate bags
        hand_tok = all_tokens[:, :8, :]
        trick_tok = all_tokens[:, 8:13, :]
        blind_tok = all_tokens[:, 13:15, :]
        bury_tok = all_tokens[:, 15:17, :]

        # Pool reasoning-enhanced tokens
        hand_vec = self.pool_hand(hand_tok, hand_mask)
        trick_vec = self.pool_trick(trick_tok, trick_mask)
        blind_vec = self.pool_blind(blind_tok, blind_mask)
        bury_vec = self.pool_bury(bury_tok, bury_mask)

        header_vec = self.header_mlp(torch.cat([header_scalar, called_emb], dim=1))

        fused = torch.cat([hand_vec, trick_vec, blind_vec, bury_vec, header_vec], dim=-1)
        return fused

    def encode_sequences(self, sequences: List[List[Dict[str, Any]]], device: torch.device | None = None) -> torch.Tensor:
        # Determine max T
        B = len(sequences)
        T = max((len(seq) for seq in sequences), default=1)
        out = torch.zeros((B, T, 256), dtype=torch.float32, device=device)
        for b, seq in enumerate(sequences):
            if not seq:
                continue
            feats = self.encode_batch(seq, device=device)  # (t,256)
            out[b, :feats.size(0), :] = feats
        return out


