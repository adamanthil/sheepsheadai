"""Critic network classes shared by the architecture registry's factories."""

from typing import Dict

import torch
import torch.nn as nn

from sheepshead import DECK_IDS, TRUMP


class _TokenReadoutValueMixin:
    """Token-readout value path shared by the perceiver critics.

    Provides the cross-attention readout construction and the value
    forward/sequence path that PerceiverCriticNetwork and
    PerceiverAuxCriticNetwork have in common (the latter previously
    borrowed these methods by class-attribute assignment). Plain mixin,
    not an nn.Module: it registers modules only inside _build_readout,
    which the host class calls at the historical point in its __init__ —
    attribute order (query -> MHA -> proj) and RNG consumption are
    load-bearing for state_dict layout and seeded bit-identity.
    """

    def _build_readout(
        self, d_token: int, d_model: int, n_readout_queries: int, n_readout_heads: int
    ):
        self._d_token = int(d_token)
        self.readout_n_queries = int(n_readout_queries)
        self.readout_query = nn.Parameter(
            torch.randn(self.readout_n_queries, self._d_token)
        )
        self.readout_mha = nn.MultiheadAttention(
            self._d_token, int(n_readout_heads), batch_first=True
        )
        self.readout_proj = nn.Linear(
            self.readout_n_queries * self._d_token, int(d_model)
        )

    def _readout(self, all_tokens: torch.Tensor, all_mask: torch.Tensor):
        B = all_tokens.size(0)
        q = self.readout_query.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.readout_mha(
            q,
            all_tokens,
            all_tokens,
            key_padding_mask=~all_mask,
            need_weights=False,
        )
        return self.readout_proj(
            attn_out.reshape(B, self.readout_n_queries * self._d_token)
        )

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        feat = self.value_trunk(
            self._readout(encoder_out["all_tokens"], encoder_out["all_mask"])
        )
        return self.value_head(feat)

    def sequence_values(self, encoder_out_seq: Dict[str, torch.Tensor]):
        at = encoder_out_seq["all_tokens"]  # (B, T, N, d_token)
        am = encoder_out_seq["all_mask"]  # (B, T, N)
        B, T, N, d = at.shape
        r = self._readout(at.reshape(B * T, N, d), am.reshape(B * T, N))
        return self.value_head(self.value_trunk(r)).view(B, T)


class PerceiverCriticNetwork(_TokenReadoutValueMixin, nn.Module):
    """Value network with its own cross-attention readout over the token set.

    The `perceiver` rung's critic: no shared trunk features — M learned
    queries attend over the 19 post-reasoning tokens, then the same deep
    value trunk shape as RecurrentCriticNetwork produces the value. No
    auxiliary heads (compare perceiver against `no-aux` as well as `full`).
    """

    def __init__(
        self,
        d_token: int = 64,
        d_model: int = 256,
        n_readout_queries: int = 4,
        n_readout_heads: int = 4,
    ):
        super().__init__()
        self.has_aux_heads = False
        d_model = int(d_model)
        self._build_readout(d_token, d_model, n_readout_queries, n_readout_heads)
        # Same shape as RecurrentCriticNetwork.value_trunk / value_head.
        act = nn.SiLU
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            act(),
            nn.Linear(d_model, d_model),
            act(),
        )
        self.value_head = nn.Linear(d_model, 1)
        # Orthogonal init for the dense stack (critic convention); the MHA
        # keeps torch defaults (encoder AttentionPool convention).
        for mod in (self.readout_proj, self.value_trunk, self.value_head):
            mod.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def aux_predictions(self, *args, **kwargs):
        raise RuntimeError(
            "aux predictions require auxiliary heads, but the perceiver "
            "critic is built without them."
        )


class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(
        self,
        d_card: int | None = None,
        use_aux_heads: bool = True,
        d_model: int = 256,
    ):
        super().__init__()
        self.has_aux_heads = bool(use_aux_heads)
        if d_card is None and self.has_aux_heads:
            raise ValueError(
                "RecurrentCriticNetwork requires card embedding dimension (d_card)."
            )
        d_model = int(d_model)

        act = nn.SiLU
        if self.has_aux_heads:
            # Adapter feeding the auxiliary heads (win/return/points/trump). Kept
            # shallow and shared so the aux tasks continue to shape the encoder.
            self.critic_adapter = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                act(),
            )
        # Dedicated, deep value trunk. Decoupled from the aux adapter so the
        # value head has capacity to fit the spread of returns instead of
        # regressing to the mean (addresses measured under-dispersion).
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            act(),
            nn.Linear(d_model, d_model),
            act(),
        )
        self.value_head = nn.Linear(d_model, 1)
        if self.has_aux_heads:
            # Auxiliary heads
            self.win_head = nn.Linear(d_model, 1)
            self.return_head = nn.Linear(d_model, 1)
            self.secret_partner_head = nn.Linear(d_model, 1)
            # Per-player point prediction head (relative seating order, length-5 vector)
            self.points_head = nn.Linear(d_model, 5)

            # Trump-tracking auxiliaries:
            #  - seen_trump_mask: multi-label (len(TRUMP)) logits, 1 = seen/known
            #  - unseen_trump_higher_than_hand: binary logit, 1 = exists unseen trump higher than best trump in hand
            self.trump_aux = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 128),
                nn.SiLU(),
            )
            # Seen-mask head uses the shared card embedding table:
            #   q = Wq(trump_aux(feat))  (B, d_key)
            #   k_i = Wk(card_embedding[trump_i])  (14, d_key)
            #   logit_i = q · k_i
            self.seen_trump_key_dim = 64
            self.seen_trump_query = nn.Linear(128, self.seen_trump_key_dim)
            self.seen_trump_key = nn.Linear(d_card, self.seen_trump_key_dim, bias=False)
            # Per-trump key offsets for better separation of trump cards with very similar embeddings.
            self.seen_trump_key_delta = nn.Parameter(
                torch.zeros(len(TRUMP), self.seen_trump_key_dim)
            )
            self.register_buffer(
                "trump_card_ids",
                torch.tensor([DECK_IDS[c] for c in TRUMP], dtype=torch.long),
                persistent=False,
            )
            self.unseen_trump_higher_than_hand_head = nn.Linear(128, 1)

        self.apply(self._init_weights)

    def _require_aux(self, what: str):
        if not self.has_aux_heads:
            raise RuntimeError(
                f"{what} requires auxiliary heads, but this critic was built "
                "without them (a no-aux architecture variant)."
            )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_out: Dict[str, torch.Tensor]):
        """Forward pass using encoder output.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        value : Tensor (batch, 1)
        """
        feat = self.value_trunk(encoder_out["features"])
        value = self.value_head(feat)
        return value

    def sequence_values(self, encoder_out_seq: Dict[str, torch.Tensor]):
        """Values over an encoded sequence batch, (B, T). Overridable seam:
        token-readout critics (PerceiverCriticNetwork) consume the token set
        instead of the fused features."""
        value_feat_bt = self.value_trunk(encoder_out_seq["features"])
        return self.value_head(value_feat_bt).squeeze(-1)

    def aux_sequence_features(self, encoder_out_seq: Dict[str, torch.Tensor]):
        """Adapter features for the aux heads over a sequence batch,
        (B, T, d_model). Overridable seam: token-readout critics
        (PerceiverAuxCriticNetwork) compute these from their own attention
        readout instead of the fused features."""
        return self.critic_adapter(encoder_out_seq["features"])

    def _aux_features_single(self, encoder_out: Dict[str, torch.Tensor]):
        """Single-step adapter features for aux_predictions. Same seam as
        aux_sequence_features, batch-of-one shape."""
        return self.critic_adapter(encoder_out["features"])

    def aux_predictions(self, encoder_out: Dict[str, torch.Tensor]):
        """Return auxiliary predictions as scalars and per-seat point estimates.

        Parameters
        ----------
        encoder_out : dict
            Output from CardReasoningEncoder with 'features'

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        secret_partner_prob : float
            Sigmoid of secret partner logits
        points_vector : list[float] | None
            Per-seat point predictions (0–120) in relative seat order.
        """
        self._require_aux("aux_predictions")
        with torch.no_grad():
            aux_feat = self._aux_features_single(encoder_out)
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
            secret_logit = self.secret_partner_head(aux_feat).squeeze(-1)
            points_pred = torch.clamp(self.points_head(aux_feat), min=0.0, max=120)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        secret_prob = float(torch.sigmoid(secret_logit).item())
        points_vector = [float(v) for v in points_pred[0].tolist()]
        return win_prob, expected_final, secret_prob, points_vector

    def seen_trump_mask_logits(
        self, feat: torch.Tensor, card_embedding: nn.Embedding
    ) -> torch.Tensor:
        """Return logits for a length-len(TRUMP) seen/known mask using card embeddings."""
        self._require_aux("seen_trump_mask_logits")
        h = self.trump_aux(feat)
        q = self.seen_trump_query(h)  # (..., d_key)
        flat_q = q.reshape(-1, q.size(-1))
        trump_ids = self.trump_card_ids.to(card_embedding.weight.device)
        table = card_embedding.weight.index_select(0, trump_ids)  # (14, d_card)
        k = self.seen_trump_key(table)  # (14, d_key)
        k = k + self.seen_trump_key_delta
        logits = torch.matmul(flat_q, k.t())  # (N, 14)
        return logits.view(*q.shape[:-1], len(TRUMP))

    def unseen_trump_higher_than_hand_logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Return logits for 'exists unseen trump higher than best trump in hand'."""
        self._require_aux("unseen_trump_higher_than_hand_logits")
        h = self.trump_aux(feat)
        return self.unseen_trump_higher_than_hand_head(h).squeeze(-1)


class PerceiverAuxCriticNetwork(_TokenReadoutValueMixin, RecurrentCriticNetwork):
    """Perceiver critic WITH the full auxiliary-head stack.

    The operator's intended perceiver design: inherits every aux head
    (win/return/secret-partner/points + trump tracking) and the deep value
    trunk from RecurrentCriticNetwork, and replaces the feature source —
    one cross-attention readout over the post-reasoning token set feeds
    BOTH the value trunk and the shallow aux adapter, so aux gradients
    shape the readout + encoder exactly as they shape the pooled trunk in
    `full`. The readout and the value forward/sequence path come from
    _TokenReadoutValueMixin (ahead of RecurrentCriticNetwork in the MRO),
    shared with PerceiverCriticNetwork.
    """

    def __init__(
        self,
        d_token: int = 64,
        d_model: int = 256,
        d_card: int = 16,
        n_readout_queries: int = 4,
        n_readout_heads: int = 4,
    ):
        super().__init__(d_card=d_card, use_aux_heads=True, d_model=d_model)
        self._build_readout(d_token, d_model, n_readout_queries, n_readout_heads)
        # super().__init__ already orthogonal-initialized the inherited stack;
        # only the projection is initialized here (MHA keeps torch defaults,
        # the encoder AttentionPool convention).
        self.readout_proj.apply(self._init_weights)

    def _readout_bt(self, encoder_out_seq: Dict[str, torch.Tensor]):
        at = encoder_out_seq["all_tokens"]  # (B, T, N, d_token)
        am = encoder_out_seq["all_mask"]  # (B, T, N)
        B, T, N, d = at.shape
        return self._readout(at.reshape(B * T, N, d), am.reshape(B * T, N)).view(
            B, T, -1
        )

    def aux_sequence_features(self, encoder_out_seq: Dict[str, torch.Tensor]):
        return self.critic_adapter(self._readout_bt(encoder_out_seq))

    def _aux_features_single(self, encoder_out: Dict[str, torch.Tensor]):
        return self.critic_adapter(
            self._readout(encoder_out["all_tokens"], encoder_out["all_mask"])
        )
