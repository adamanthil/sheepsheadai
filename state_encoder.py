import torch
import torch.nn as nn
from typing import List, Dict, Any
import math


PAD_CARD_ID = 0
UNDER_CARD_ID = 33


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

    def __init__(self, d_card: int = 8, d_seat: int = 4, d_role: int = 4, d_token: int = 32):
        super().__init__()
        # Embeddings
        self.card = nn.Embedding(34, d_card)  # 0..33
        self.seat = nn.Embedding(6, d_seat)   # 0..5 (0 unused in trick)
        self.role = nn.Embedding(4, d_role)   # 0 none, 1 picker, 2 partner, 3 both

        # Token projections
        self.token_mlp_trick = nn.Sequential(
            nn.Linear(d_card + d_seat + d_role, d_token),
            nn.SiLU(),
        )
        self.token_mlp_simple = nn.Sequential(
            nn.Linear(d_card, d_token),
            nn.SiLU(),
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

        # Embedding + pooling
        hand_tok, hand_mask = self._embed_simple_bag(hand_ids)
        blind_tok, blind_mask = self._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = self._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = self._embed_trick(trick_card_ids, trick_is_picker, trick_is_partner_known)

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


