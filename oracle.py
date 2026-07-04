"""Privileged (oracle) critic: centralized value estimation for PPO training.

Centralized-training / decentralized-execution (CTDE): during training, a
critic that sees ALL hidden information (every hand, the true blind/bury,
the under card, the secret partner's seat) estimates V for the GAE baseline,
removing the hidden-card component of advantage variance. Actors — and the
deployed agent — never see any of this; the oracle exists only inside
`PPOAgent.update()`.

Literature this design follows:
  * Asymmetric actor-critic: train the critic on privileged state while the
    actor acts on observations (Pinto et al. 2017, arXiv:1710.06542).
  * Centralized value functions in multi-agent PPO: MAPPO (Yu et al. 2021,
    arXiv:2103.01955), COMA (Foerster et al. 2018, arXiv:1705.08926); at
    scale, OpenAI Five's value function saw hidden state (Berner et al.
    2019, arXiv:1912.06680) and AlphaStar's saw both players' observations
    (Vinyals et al. 2019, Nature 575). In hidden-information card games
    specifically, Suphx used perfect-information "oracle guiding" for
    Mahjong (Li et al. 2020, arXiv:2003.13590).
  * Conditioning: a critic conditioned on the hidden STATE ALONE is biased
    under partial observability; the sound target is the history-state
    value U(h, s) (Baisero & Amato 2022, arXiv:2105.11674; see also Lyu et
    al. 2023, JAIR 76, on centralized-critic bias/variance trade-offs).
    Hence the oracle input is a strict superset of the actor's observation
    (`Player.get_oracle_state_dict`) and the encoder is RECURRENT over the
    same event stream the actor sees — it estimates U(h, s), not V(s).
  * The baseline itself: GAE (Schulman et al. 2016, arXiv:1506.02438) with
    the oracle's values in place of the limited critic's.

Isolation guarantee: the oracle owns its encoder — zero parameters are
shared with the policy encoder/actor/limited critic — so privileged
gradients cannot shape the actor, and the limited critic + auxiliary heads
train exactly as they do without the oracle (bit-comparable baseline arm).
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from encoder import (
    AttentionPool,
    CardEmbeddingConfig,
    CardReasoningEncoder,
    PAD_CARD_ID,
)


class OracleCriticEncoder(CardReasoningEncoder):
    """CardReasoningEncoder over the full-information observation.

    Reuses the policy encoder's machinery wholesale (card embeddings with
    informed init, seat/role embeddings, transformer reasoning, attention
    pools, GRU memory, `encode_sequences`) and extends the token set with
    the hidden state. Token layout (51 tokens vs the base encoder's 19):

        [context, memory, hand x8, trick x5, blind x2, bury x2, opp x32]

    where the blind/bury slots carry the TRUE cards for every seat and the
    32 opponent-hand tokens carry seat + picker/secret-partner role info.
    The context token additionally sees points_taken, the secret partner's
    relative seat, and the under card. `encode_batch` is an adapted copy of
    the base implementation rather than a hook-based refactor so the policy
    encoder's forward path stays byte-identical.
    """

    # card_type ids: 0=context, 1=memory, 2=hand, 3=trick, 4=blind, 5=bury
    OPP_TYPE_ID = 6

    def __init__(
        self,
        d_card: int = 16,
        d_token: int = 64,
        card_config: CardEmbeddingConfig | None = None,
        n_reasoning_heads: int = 4,
        n_reasoning_layers: int = 4,
    ):
        super().__init__(
            d_card=d_card,
            d_token=d_token,
            card_config=card_config,
            n_reasoning_heads=n_reasoning_heads,
            n_reasoning_layers=n_reasoning_layers,
        )
        d_card = self.d_card_dim
        d_token = self.d_token_dim

        # Context gains: 5 points_taken_rel (/120), secret_partner_rel (/5),
        # and the under-card embedding alongside the called-card embedding.
        self.context_mlp = nn.Sequential(
            nn.Linear(10 + 5 + 1 + 2 * d_card, d_token),
            nn.SiLU(),
        )
        # +1 token type for opponent-hand cards.
        self.card_type = nn.Embedding(7, d_token)
        # Opponent-hand tokens: card + seat + role (same shape as trick MLP).
        self.token_mlp_opp = nn.Sequential(
            nn.Linear(d_card + 4 + 4, d_token),
            nn.SiLU(),
        )
        self.pool_opp = AttentionPool(d_token, 64)
        # Fusion gains the pooled opponent bag.
        self.feature_proj = nn.Sequential(
            nn.Linear(64 + 64 + 32 + 32 + 64 + d_token, 256),
            nn.LayerNorm(256),
        )

    def _embed_opp_hands(
        self,
        opp_ids: torch.Tensor,
        picker_rel: torch.Tensor,
        secret_partner_rel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the 4x8 opponent-hand cards with seat/role conditioning.

        Args:
            opp_ids: (B, 32) card ids, seats rel 2..5 in blocks of 8
            picker_rel / secret_partner_rel: (B,) relative seats (0 = none)
        """
        B = opp_ids.size(0)
        seat_rel = (
            torch.arange(2, 6, device=opp_ids.device, dtype=torch.long)
            .repeat_interleave(8)
            .view(1, 32)
            .expand(B, 32)
        )
        role_ids = (
            seat_rel.eq(picker_rel.view(B, 1)).long() * 1
            + seat_rel.eq(secret_partner_rel.view(B, 1)).long() * 2
        )
        mask = opp_ids.ne(PAD_CARD_ID)
        tok = torch.cat(
            [self.card(opp_ids), self.seat(seat_rel), self.role(role_ids)], dim=-1
        )
        return self.token_mlp_opp(tok), mask

    def encode_batch(
        self,
        batch: List[Dict[str, Any]],
        memory_in: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Adapted copy of CardReasoningEncoder.encode_batch over the
        51-token full-information layout. Returns the same keys."""

        def to_device(x: torch.Tensor) -> torch.Tensor:
            return x.to(device) if device is not None else x

        B = len(batch)
        if memory_in is None:
            memory_in = torch.zeros((B, 256), dtype=torch.float32, device=device)
        else:
            memory_in = to_device(memory_in)

        # 1. Context token: base header + privileged scalars + card embs.
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
        header_scalar = to_device(torch.cat(header_cols, dim=1))
        norm = torch.tensor(
            [1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0],
            dtype=header_scalar.dtype,
            device=header_scalar.device,
        )
        header_scalar = header_scalar / norm

        points_rel = to_device(
            self._stack_uint8(batch, "points_taken_rel", 5).float() / 120.0
        )
        secret_rel_raw = to_device(
            torch.as_tensor(
                [int(s["secret_partner_rel"]) for s in batch], dtype=torch.long
            )
        )
        secret_scalar = secret_rel_raw.float().view(B, 1) / 5.0

        called_ids = to_device(
            torch.as_tensor([int(s["called_card_id"]) for s in batch], dtype=torch.long)
        )
        under_ids = to_device(
            torch.as_tensor([int(s["under_card_id"]) for s in batch], dtype=torch.long)
        )
        context_tok = self.context_mlp(
            torch.cat(
                [
                    header_scalar,
                    points_rel,
                    secret_scalar,
                    self.card(called_ids),
                    self.card(under_ids),
                ],
                dim=1,
            )
        )

        # 2. Memory token.
        memory_tok = self.memory_in_proj(memory_in)

        # 3. Actor role (as in the base encoder).
        picker_rel_raw = to_device(
            torch.as_tensor([int(s["picker_rel"]) for s in batch], dtype=torch.long)
        )
        partner_rel_raw = to_device(
            torch.as_tensor([int(s["partner_rel"]) for s in batch], dtype=torch.long)
        )
        actor_role_id = (
            picker_rel_raw.eq(1).long() * 1 + partner_rel_raw.eq(1).long() * 2
        )

        # 4. Card tokens.
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
        opp_ids = to_device(self._stack_uint8(batch, "opp_hand_ids", 32)).reshape(B, 32)

        hand_tok, hand_mask = self._embed_hand(hand_ids, actor_role_id)
        blind_tok, blind_mask = self._embed_simple_bag(blind_ids)
        bury_tok, bury_mask = self._embed_simple_bag(bury_ids)
        trick_tok, trick_mask = self._embed_trick(
            trick_card_ids, trick_is_picker, trick_is_partner_known
        )
        opp_tok, opp_mask = self._embed_opp_hands(
            opp_ids, picker_rel_raw, secret_rel_raw
        )

        # 5. Concatenate 51 tokens + masks + type embeddings.
        dev = hand_tok.device
        all_tokens = torch.cat(
            [
                context_tok.unsqueeze(1),
                memory_tok.unsqueeze(1),
                hand_tok,
                trick_tok,
                blind_tok,
                bury_tok,
                opp_tok,
            ],
            dim=1,
        )
        ones = torch.ones((B, 1), dtype=torch.bool, device=dev)
        all_mask = torch.cat(
            [ones, ones, hand_mask, trick_mask, blind_mask, bury_mask, opp_mask],
            dim=1,
        )
        type_ids = torch.cat(
            [
                torch.zeros((B, 1), dtype=torch.long, device=dev),
                torch.ones((B, 1), dtype=torch.long, device=dev),
                torch.full((B, 8), 2, dtype=torch.long, device=dev),
                torch.full((B, 5), 3, dtype=torch.long, device=dev),
                torch.full((B, 2), 4, dtype=torch.long, device=dev),
                torch.full((B, 2), 5, dtype=torch.long, device=dev),
                torch.full((B, 32), self.OPP_TYPE_ID, dtype=torch.long, device=dev),
            ],
            dim=1,
        )
        all_tokens = all_tokens + self.card_type(type_ids)

        # 6. Reason, extract, pool, update memory, fuse.
        all_tokens = self.card_reasoner(all_tokens, all_mask)
        context_out = all_tokens[:, 0, :]
        hand_tok_out = all_tokens[:, 2:10, :]
        trick_tok_out = all_tokens[:, 10:15, :]
        blind_tok_out = all_tokens[:, 15:17, :]
        bury_tok_out = all_tokens[:, 17:19, :]
        opp_tok_out = all_tokens[:, 19:51, :]

        hand_vec = self.pool_hand(hand_tok_out, hand_mask)
        trick_vec = self.pool_trick(trick_tok_out, trick_mask)
        blind_vec = self.pool_blind(blind_tok_out, blind_mask)
        bury_vec = self.pool_bury(bury_tok_out, bury_mask)
        opp_vec = self.pool_opp(opp_tok_out, opp_mask)

        memory_out = self.memory_gru(context_out, memory_in)
        features = self.feature_proj(
            torch.cat(
                [hand_vec, trick_vec, blind_vec, bury_vec, opp_vec, context_out],
                dim=1,
            )
        )
        return {
            "features": features,
            "hand_tokens": hand_tok_out,
            "context_token": context_out,
            "memory_out": memory_out,
        }


class OracleValueNetwork(nn.Module):
    """Recurrent U(h, s) value network over full-information observations.

    Own encoder, own trunk, no auxiliary heads, no parameter sharing with
    the policy stack. Trained only at update time on lambda-returns; used
    only to supply the GAE baseline (see module docstring).
    """

    def __init__(self):
        super().__init__()
        self.encoder = OracleCriticEncoder(card_config=CardEmbeddingConfig())
        act = nn.SiLU
        # Same shape as RecurrentCriticNetwork.value_trunk.
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            act(),
            nn.Linear(256, 256),
            act(),
        )
        self.value_head = nn.Linear(256, 1)
        self.value_trunk.apply(self._init_weights)
        self.value_head.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward_sequences(
        self,
        sequences: List[List[Dict[str, Any]]],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode B sequences of oracle observations (fresh zero memory per
        sequence, the same protocol as the limited critic's training
        forward) and return values of shape (B, T)."""
        out = self.encoder.encode_sequences(sequences, memory_in=None, device=device)
        return self.value_head(self.value_trunk(out["features"])).squeeze(-1)

    def param_groups(self, base_lr: float, card_lr_scale: float = 0.2):
        """Card embeddings at scaled LR, everything else at base LR.

        Name-filtered (unlike CardReasoningEncoder.param_groups, which
        enumerates submodules and would silently drop the oracle's extra
        modules)."""
        card_params = list(self.encoder.card.parameters())
        card_ids = {id(p) for p in card_params}
        rest = [p for p in self.parameters() if id(p) not in card_ids]
        return [
            {"params": card_params, "lr": base_lr * card_lr_scale},
            {"params": rest, "lr": base_lr},
        ]
