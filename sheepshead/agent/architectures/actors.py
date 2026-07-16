"""Actor network classes shared by the architecture registry's factories."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sheepshead import ACTION_IDS


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network consuming encoder features, with separate heads for
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(
        self,
        action_size,
        action_groups,
        *,
        d_card: int,
        d_token: int,
        d_model: int = 256,
        map_cid_to_play_action_index: torch.Tensor,
        map_cid_to_bury_action_index: torch.Tensor,
        map_cid_to_under_action_index: torch.Tensor,
        call_action_global_indices: torch.Tensor,
        call_card_ids: torch.Tensor,
        play_under_action_index: int,
    ):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.action_size = action_size
        self.action_groups = (
            action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'
        )
        d_model = int(d_model)
        self._d_model = d_model

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(d_model, len(action_groups["pick"]))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(d_model, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(d_model, 1)

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(d_model, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(d_model, self.tw_latent)
        self.tw_We = nn.Linear(self._d_card, self.tw_latent)

        # Action index mappings
        self._map_cid_to_play_action_index = map_cid_to_play_action_index.clone().long()
        self._map_cid_to_bury_action_index = map_cid_to_bury_action_index.clone().long()
        self._map_cid_to_under_action_index = (
            map_cid_to_under_action_index.clone().long()
        )
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(
        self, feat: torch.Tensor, card_embedding: nn.Embedding
    ) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)  # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight  # (34, d_card)
        K = self.tw_We(table)  # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())  # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(
        self,
        feat: torch.Tensor,
        hand_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_tokens: (B, N, d_token)
        Returns: s_slots (B, N)
        """
        tok = hand_tokens
        B, N = tok.size(0), tok.size(1)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1)  # (B, N, h)
        t = self.pointer_Wt(tok)  # (B, N, h)
        e = torch.tanh(g + t)  # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)  # (B, N)
        return s

    def _adapt_features(
        self,
        actor_features: torch.Tensor,
        all_tokens: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Turn encoder features into the head input. Overridable seam:
        readout variants (see TokenReadActorNetwork) additionally attend over
        the full post-reasoning token set; the base actor ignores it."""
        return self.actor_adapter(actor_features)

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        all_tokens: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None
        all_tokens : Tensor (K, N_all, d_token) | None
            Full post-reasoning token set (only emitted by tokenread encoder
            variants; consumed by TokenReadActorNetwork, ignored here).
        all_mask : BoolTensor (K, N_all) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self._adapt_features(actor_features, all_tokens, all_mask)

        # PICK / PASS
        pick_logits = self.pick_head(feat)
        logits[:, self.action_groups["pick"]] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)]
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(
            feat,
            hand_tokens,
        )  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]  # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[
                    valid_b, i
                ]
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[
                    valid_u, i
                ]

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[
                    valid_p, i
                ]

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1)
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features' and 'hand_tokens'
        action_mask : Bool Tensor (batch, action_size)
        hand_ids : Tensor (batch, 8)
        card_embedding : nn.Embedding

        Returns
        -------
        probs : Tensor (batch, action_size)
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out["features"],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out["hand_tokens"],
            action_mask=action_mask,
            all_tokens=encoder_out.get("all_tokens"),
            all_mask=encoder_out.get("all_mask"),
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    def forward_with_logits(
        self,
        encoder_out: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        """
        logits = self._build_logits_from_features(
            actor_features=encoder_out["features"],
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            hand_tokens=encoder_out["hand_tokens"],
            action_mask=action_mask,
            all_tokens=encoder_out.get("all_tokens"),
            all_mask=encoder_out.get("all_mask"),
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class TokenReadActorNetwork(MultiHeadRecurrentActorNetwork):
    """Pointer actor plus a cross-attention readout over ALL post-reasoning
    tokens (the `full-tokenread` architecture rung).

    Hypothesis under test: the per-bag attention pools between the token
    stack and the 256-d trunk (hand→64, trick→64, blind→32, bury→32 dims)
    throw away card-level information before the policy heads see it. Here
    the heads' input is `fuse(actor_adapter(features) ⊕ readout)`, where the
    readout is M learned queries attending over the full 19-token
    post-reasoning set — an unmediated read path from tokens to every head
    (including the pointer's situation-conditioning and the two-tower CALL
    scorer). The encoder — including the memory recurrence (context token →
    GRUCell → 256-d state) — is byte-identical to `full`; only the actor
    grows. Requires an encoder that emits 'all_tokens'/'all_mask'
    (TokenReadEncoder in architectures.encoders).
    """

    def __init__(
        self,
        *args,
        n_readout_queries: int = 4,
        n_readout_heads: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = self._d_model
        # Same construction style as encoder.AttentionPool; created after
        # super().__init__ so, like the pointer/two-tower layers, they keep
        # default (non-orthogonal) init.
        self.readout_n_queries = int(n_readout_queries)
        self.readout_query = nn.Parameter(
            torch.randn(self.readout_n_queries, self._d_token)
        )
        self.readout_mha = nn.MultiheadAttention(
            self._d_token, int(n_readout_heads), batch_first=True
        )
        self.readout_proj = nn.Linear(self.readout_n_queries * self._d_token, d_model)
        self.readout_fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
        )

    def _adapt_features(
        self,
        actor_features: torch.Tensor,
        all_tokens: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if all_tokens is None or all_mask is None:
            raise RuntimeError(
                "TokenReadActorNetwork requires encoder outputs 'all_tokens'/"
                "'all_mask' (use a tokenread encoder variant)."
            )
        feat = self.actor_adapter(actor_features)
        B = all_tokens.size(0)
        q = self.readout_query.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_token)
        # Context + memory tokens are always valid, so no all-masked-row guard
        # is needed (cf. encoder.AttentionPool).
        attn_out, _ = self.readout_mha(
            q,
            all_tokens,
            all_tokens,
            key_padding_mask=~all_mask,
            need_weights=False,
        )  # (B, M, d_token)
        readout = self.readout_proj(
            attn_out.reshape(B, self.readout_n_queries * self._d_token)
        )  # (B, d_model)
        return self.readout_fuse(torch.cat([feat, readout], dim=1))


class PerceiverActorNetwork(MultiHeadRecurrentActorNetwork):
    """Actor whose ONLY input is a cross-attention readout over the full
    post-reasoning token set (the `perceiver` architecture rung).

    Unlike TokenReadActorNetwork (additive probe: readout fused WITH the
    pooled trunk features), this actor ignores `actor_features` entirely:
    M learned queries attend over the 19 tokens, the flattened result is
    projected to d_model and passed through the standard actor_adapter
    before the heads. Requires an encoder emitting 'all_tokens'/'all_mask'
    (PerceiverEncoder in architectures.encoders).
    """

    def __init__(
        self,
        *args,
        n_readout_queries: int = 4,
        n_readout_heads: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_model = self._d_model
        self.readout_n_queries = int(n_readout_queries)
        self.readout_query = nn.Parameter(
            torch.randn(self.readout_n_queries, self._d_token)
        )
        self.readout_mha = nn.MultiheadAttention(
            self._d_token, int(n_readout_heads), batch_first=True
        )
        self.readout_proj = nn.Linear(self.readout_n_queries * self._d_token, d_model)

    def _adapt_features(
        self,
        actor_features: torch.Tensor,
        all_tokens: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if all_tokens is None or all_mask is None:
            raise RuntimeError(
                "PerceiverActorNetwork requires encoder outputs 'all_tokens'/"
                "'all_mask' (use the perceiver encoder variant)."
            )
        del actor_features  # perceiver heads read tokens only
        B = all_tokens.size(0)
        q = self.readout_query.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.readout_mha(
            q,
            all_tokens,
            all_tokens,
            key_padding_mask=~all_mask,
            need_weights=False,
        )
        readout = self.readout_proj(
            attn_out.reshape(B, self.readout_n_queries * self._d_token)
        )
        return self.actor_adapter(readout)
