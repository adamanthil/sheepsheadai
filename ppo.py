import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import architectures
from sheepshead import (
    ACTION_IDS,
    BURY_ACTIONS,
    CALL_ACTIONS,
    DECK_IDS,
    PLAY_ACTIONS,
    TRUMP,
    UNDER_ACTIONS,
    UNDER_TOKEN,
)
from training_utils import RETURN_SCALE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Manual scaling constants for critic heads
POINTS_SCALE = 10.0  # Bring 0–120 point regression into ~0–12 range


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

        # Shared actor adapter to add nonlinearity and specialization before heads
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups["pick"]))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Per-head temperatures (τ): logits will be divided by τ before softmax
        # τ > 1.0 softens (higher entropy), τ < 1.0 sharpens. Defaults to 1.0.
        self.temperature_pick = 1.0
        self.temperature_partner = 1.0
        self.temperature_bury = 1.0
        self.temperature_play = 1.0

        # Init (generic weights)
        self.apply(self._init_weights)

        # ---- Card-conditioned configuration (dims + mappings) ----
        self._d_card = int(d_card)
        self._d_token = int(d_token)
        # Pointer scorer (Bahdanau style)
        self.pointer_hidden = 64
        self.pointer_Wg = nn.Linear(256, self.pointer_hidden)
        self.pointer_Wt = nn.Linear(self._d_token, self.pointer_hidden)
        self.pointer_v = nn.Linear(self.pointer_hidden, 1, bias=False)
        # Two-tower (card CALL scoring)
        self.tw_latent = 64
        self.tw_Wg = nn.Linear(256, self.tw_latent)
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

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        hand_tokens: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        hand_tokens : Tensor (K, N, d_token)
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # Adapt features
        feat = self.actor_adapter(actor_features)

        # PICK / PASS
        pick_logits = self.pick_head(feat) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups["pick"]] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(feat) / max(
            self.temperature_partner, 1e-6
        )
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(feat, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(
            self.temperature_partner, 1e-6
        )
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
                ] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[
                    valid_u, i
                ] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[
                    valid_p, i
                ] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(feat).squeeze(-1) / max(
                self.temperature_play, 1e-6
            )
            logits[:, self._play_under_action_index] = play_under_logit

        # Apply action mask if provided
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        return logits

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def set_temperatures(
        self,
        pick: float | None = None,
        partner: float | None = None,
        bury: float | None = None,
        play: float | None = None,
    ):
        """Set per-head softmax temperatures.

        Parameters
        ----------
        pick, partner, bury, play : float | None
            If provided, sets the corresponding head's temperature τ. Values are
            clamped to a small positive minimum to avoid divide-by-zero.
        """
        eps = 1e-6
        if pick is not None:
            self.temperature_pick = max(float(pick), eps)
        if partner is not None:
            self.temperature_partner = max(float(partner), eps)
        if bury is not None:
            self.temperature_bury = max(float(bury), eps)
        if play is not None:
            self.temperature_play = max(float(play), eps)

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
        )
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class RecurrentCriticNetwork(nn.Module):
    """Critic head using encoder features directly."""

    def __init__(self, d_card: int | None = None, use_aux_heads: bool = True):
        super().__init__()
        self.has_aux_heads = bool(use_aux_heads)
        if d_card is None and self.has_aux_heads:
            raise ValueError(
                "RecurrentCriticNetwork requires card embedding dimension (d_card)."
            )

        act = nn.SiLU
        if self.has_aux_heads:
            # Adapter feeding the auxiliary heads (win/return/points/trump). Kept
            # shallow and shared so the aux tasks continue to shape the encoder.
            self.critic_adapter = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                act(),
            )
        # Dedicated, deep value trunk. Decoupled from the aux adapter so the
        # value head has capacity to fit the spread of returns instead of
        # regressing to the mean (addresses measured under-dispersion).
        self.value_trunk = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            act(),
            nn.Linear(256, 256),
            act(),
        )
        self.value_head = nn.Linear(256, 1)
        if self.has_aux_heads:
            # Auxiliary heads
            self.win_head = nn.Linear(256, 1)
            self.return_head = nn.Linear(256, 1)
            self.secret_partner_head = nn.Linear(256, 1)
            # Per-player point prediction head (relative seating order, length-5 vector)
            self.points_head = nn.Linear(256, 5)

            # Trump-tracking auxiliaries:
            #  - seen_trump_mask: multi-label (len(TRUMP)) logits, 1 = seen/known
            #  - unseen_trump_higher_than_hand: binary logit, 1 = exists unseen trump higher than best trump in hand
            self.trump_aux = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 128),
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
            aux_feat = self.critic_adapter(encoder_out["features"])
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


class PPOAgent:
    def __init__(
        self,
        action_size,
        lr_actor=3e-4,
        lr_critic=3e-4,
        critic_mode="limited",
        arch="full",
    ):
        # Networks are built from a named ArchitectureSpec (architectures.py).
        # The default "full" spec constructs exactly the pre-registry
        # networks, in the same order, so seeded runs are bit-identical.
        spec = architectures.get_spec(arch)
        self.arch_name = spec.name
        self.arch_spec = spec

        # Dict-based encoder → fixed 256-d features
        self.state_size = 256
        self.action_size = action_size

        # --------------------------------------------------
        # Action groups for multi–head policy
        # --------------------------------------------------
        pick_indices = [ACTION_IDS["PICK"] - 1, ACTION_IDS["PASS"] - 1]

        # Partner-selection head: ALONE, JD PARTNER, all partner CALL actions
        partner_selection_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS
        partner_indices = sorted({ACTION_IDS[a] - 1 for a in partner_selection_actions})

        # Bury head: explicit bury actions and UNDER actions
        bury_indices = sorted(
            {ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)}
        )

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            "pick": sorted(pick_indices),
            "partner": sorted(partner_indices),
            "bury": sorted(bury_indices),
            "play": sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Encoder with memory
        self.encoder = spec.build_encoder().to(device)

        # Per-player memory tracking
        self._player_memories = {}

        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        actor_mappings = {
            "map_cid_to_play_action_index": map_cid_to_play_action_index,
            "map_cid_to_bury_action_index": map_cid_to_bury_action_index,
            "map_cid_to_under_action_index": map_cid_to_under_action_index,
            "call_action_global_indices": call_action_global_indices,
            "call_card_ids": call_card_ids,
            "play_under_action_index": play_under_action_index,
        }
        self.actor = spec.build_actor(
            action_size, self.action_groups, self.encoder, actor_mappings
        ).to(device)

        self.critic = spec.build_critic(self.encoder).to(device)

        # Optimizers (include encoder params with scaled LR for card embeddings)
        encoder_groups = self.encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.2)
        actor_groups = [
            {"params": self.actor.parameters(), "lr": lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Privileged (oracle) critic: a fully separate value network over
        # full-information observations, used only as the GAE baseline during
        # update() (asymmetric actor-critic / CTDE; see oracle.py for the
        # literature). The limited critic above keeps training unchanged —
        # its aux heads shape the shared trunk and its value head stays the
        # deployable, observation-only estimator. A third optimizer keeps
        # existing checkpoints' critic_optimizer state loadable.
        self.critic_mode = critic_mode
        self.oracle_critic = None
        self.oracle_optimizer = None
        self.oracle_value_loss_coeff = 1.0
        self._oracle_lr_ratios = [0.2, 1.0]  # card embeddings, rest
        if critic_mode == "oracle":
            from oracle import OracleValueNetwork

            self.oracle_critic = OracleValueNetwork().to(device)
            self.oracle_optimizer = optim.Adam(
                self.oracle_critic.param_groups(
                    base_lr=lr_critic, card_lr_scale=self._oracle_lr_ratios[0]
                )
            )
        elif critic_mode != "limited":
            raise ValueError(
                f"critic_mode must be 'limited' or 'oracle': {critic_mode}"
            )

        # Track LR ratios for actor optimizer param groups (relative to group 0)
        # Used to maintain correct relative scaling when updating learning rates
        self._actor_lr_ratios = self._capture_actor_lr_ratios(base_lr=float(lr_actor))

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 1.0
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.20
        self.clip_epsilon_partner = 0.25
        self.clip_epsilon_bury = 0.15
        self.clip_epsilon_play = 0.20
        self.value_clip_epsilon = 0.20

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients
        self.win_loss_coeff = 0.05
        self.return_loss_coeff = 0.1
        self.secret_loss_coeff = 0.1
        self.points_loss_coeff = 0.2
        self.seen_trump_mask_loss_coeff = 0.2
        self.unseen_trump_higher_than_hand_loss_coeff = 0.1

        # Stage C: ISMCTS soft-teacher distillation. On transitions carrying a
        # confident search target (ESS >= floor) the policy is trained by forward-KL
        # distillation toward pi'; the value loss still runs on every transition.
        # search_distill_coeff scales this KL(pi' || pi_theta) term (plan §3: 1.0).
        self.search_distill_coeff = 1.0
        # A/B knob for the searched-transition policy loss (plan §4):
        #   0.0 -> hard PG-mask (drop the PPO term on searched states; distillation
        #          owns them) -- the default.
        #   1.0 -> additive form (keep the PPO clip term AND add distillation).
        #   0<w<1 -> residual PG weight (fallback if the hard mask is too noisy).
        # Unsearched transitions always keep full PG. See _actor_critic_losses.
        self.searched_ppo_weight = 0.0

        # Bidding-head KL anchor (ExIt warm-start guard): when enabled via
        # set_anchor(), the actor loss gains
        #   anchor_coeff * KL(pi_ref || pi_theta)
        # on pick/partner/bury transitions, toward a frozen reference policy.
        # Learner-side only (collection/workers untouched); the play head is not
        # anchored. 0.0 / None disables.
        self.anchor_coeff = 0.0
        self._anchor_agent = None

        # Storage for trajectory data
        self.reset_storage()

    def get_recurrent_memory(
        self, player_id: int | None, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return the recurrent memory vector for a player, or a zero vector if unset.

        The returned tensor is (256,) on the requested device.
        """
        target_device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        mem = self._player_memories.get(player_id) if player_id is not None else None
        if mem is None:
            return torch.zeros(256, device=target_device)
        return mem.to(target_device) if mem.device != target_device else mem

    def set_recurrent_memory(
        self, player_id: int | None, memory_vector: torch.Tensor
    ) -> None:
        """Set the recurrent memory vector for a player."""
        self._player_memories[player_id] = memory_vector

    def _build_action_index_mappings(self):
        """Precompute global action index mappings for card-specific actions."""
        map_cid_to_play_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_bury_action_index = torch.full((34,), -1, dtype=torch.long)
        map_cid_to_under_action_index = torch.full((34,), -1, dtype=torch.long)

        # PLAY actions
        play_under_index = None
        for a in PLAY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            if a == f"PLAY {UNDER_TOKEN}":
                play_under_index = a_idx
                continue
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_play_action_index[cid] = a_idx

        # BURY actions
        for a in BURY_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_bury_action_index[cid] = a_idx

        # UNDER actions (choose card from hand to set as under)
        for a in UNDER_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            card = a.split()[1]
            cid = DECK_IDS[card]
            map_cid_to_under_action_index[cid] = a_idx

        # CALL actions
        idx_call_global_list = []
        call_card_ids_list = []
        for a in CALL_ACTIONS:
            a_idx = ACTION_IDS[a] - 1
            parts = a.split()
            # a is either "CALL X" or "CALL X UNDER"
            card = parts[1]
            cid = DECK_IDS[card]
            idx_call_global_list.append(a_idx)
            call_card_ids_list.append(cid)

        idx_call_global = torch.tensor(sorted(idx_call_global_list), dtype=torch.long)
        # Align cid list to the sorted order of indices
        # Build mapping from index to cid then reorder
        idx_to_cid = {
            idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)
        }
        call_card_ids = torch.tensor(
            [idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long
        )

        return (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            idx_call_global,
            call_card_ids,
            play_under_index,
        )

    def set_head_temperatures(
        self,
        pick: float | None = None,
        partner: float | None = None,
        bury: float | None = None,
        play: float | None = None,
    ):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def _capture_actor_lr_ratios(self, base_lr: float | None = None) -> list[float]:
        """Return per-param-group LR ratios for the actor optimizer.

        Ratios are defined relative to the base LR (actor optimizer group 0).
        This preserves the intended scaling between the actor params and encoder
        param groups (e.g., card embeddings at a smaller LR).
        """
        if base_lr is None:
            base_lr = float(self.actor_optimizer.param_groups[0]["lr"])
        if base_lr <= 0.0:
            existing = getattr(self, "_actor_lr_ratios", None)
            if existing is not None:
                return existing
            return [1.0 for _ in self.actor_optimizer.param_groups]
        return [
            float(group["lr"]) / base_lr for group in self.actor_optimizer.param_groups
        ]

    def set_learning_rates(
        self, actor_lr: float | None = None, critic_lr: float | None = None
    ):
        """Set learning rates for actor and/or critic optimizers.

        For the actor optimizer, maintains correct relative scaling across param groups:
        - Group 0: actor parameters at actor_lr
        - Group 1: card embeddings at actor_lr * ratio[1] (typically 0.2)
        - Group 2: other encoder params at actor_lr * ratio[2] (typically 1.0)

        Parameters
        ----------
        actor_lr : float | None
            Base learning rate for actor optimizer (applied to group 0).
            If None, actor LR is not changed.
        critic_lr : float | None
            Learning rate for critic optimizer.
            If None, critic LR is not changed.
        """
        if actor_lr is not None:
            actor_lr = float(actor_lr)
            # Refresh ratios in case optimizer state was loaded
            self._actor_lr_ratios = self._capture_actor_lr_ratios()
            # Apply actor LR to all param groups maintaining relative ratios
            for i, group in enumerate(self.actor_optimizer.param_groups):
                group["lr"] = actor_lr * self._actor_lr_ratios[i]

        if critic_lr is not None:
            critic_lr = float(critic_lr)
            self.critic_optimizer.param_groups[0]["lr"] = critic_lr
            # The oracle critic rides the same LR schedule as the limited one.
            if self.oracle_optimizer is not None:
                for group, ratio in zip(
                    self.oracle_optimizer.param_groups, self._oracle_lr_ratios
                ):
                    group["lr"] = critic_lr * ratio

    def set_anchor(self, ref_agent, coeff: float):
        """Enable (or disable) the bidding-head KL anchor toward a frozen
        reference policy: actor loss gains coeff * KL(pi_ref || pi_theta) on
        pick/partner/bury transitions. ``ref_agent`` is a loaded PPOAgent whose
        encoder/actor produce the reference logits; it is frozen and put in eval
        mode here. Pass (None, 0.0) to disable (population snapshots do this so
        they don't carry the reference copy)."""
        self._anchor_agent = ref_agent
        self.anchor_coeff = float(coeff) if ref_agent is not None else 0.0
        if ref_agent is not None:
            for net in (ref_agent.encoder, ref_agent.actor, ref_agent.critic):
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)

    def strip_oracle(self):
        """Drop the oracle critic and its optimizer (reverting to limited
        mode). Population snapshots call this so league members don't carry
        (or persist) the privileged network, which is a training-time-only
        construct — inference never uses it."""
        self.oracle_critic = None
        self.oracle_optimizer = None
        self.critic_mode = "limited"

    def reset_storage(self):
        # Ordered list of events: each is a dict with keys:
        # kind: 'action' | 'obs'
        # state: np.array state vector
        # mask: torch.bool mask or convertible
        # If kind == 'action': action (0-indexed), reward, value, log_prob, done
        # GAE writes back: 'advantage' and 'return'
        self.events = []

    def get_action_mask(self, valid_actions, action_size):
        """Convert valid actions set to boolean mask"""
        mask = torch.zeros(action_size, dtype=torch.bool)
        for action in valid_actions:
            mask[action - 1] = True  # Convert from 1-indexed to 0-indexed
        return mask

    def reset_recurrent_state(self):
        """Clear memory states (call at the start of every new game)."""
        self._player_memories = {}

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        # Encode dict state with memory
        encoder_out = self.encoder.encode_batch(
            [state], memory_in=memory_in.unsqueeze(0), device=device
        )

        # Store updated memory
        if player_id is not None:
            self.set_recurrent_memory(player_id, encoder_out["memory_out"][0])

        action_mask_t = (
            self.get_action_mask(valid_actions, self.action_size)
            .unsqueeze(0)
            .to(device)
        )
        hand_ids_t = torch.as_tensor(
            state["hand_ids"], dtype=torch.long, device=device
        ).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Get or init memory for this player
            memory_in = self.get_recurrent_memory(player_id, device=device)

            # Encode with memory
            encoder_out = self.encoder.encode_batch(
                [state], memory_in=memory_in.unsqueeze(0), device=device
            )

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out["memory_out"][0])

            # Get action probabilities
            action_mask_t = (
                self.get_action_mask(valid_actions, self.action_size)
                .unsqueeze(0)
                .to(device)
            )
            hand_ids_t = torch.as_tensor(
                state["hand_ids"], dtype=torch.long, device=device
            ).view(1, -1)

            action_probs = self.actor(
                encoder_out,
                action_mask_t,
                hand_ids_t,
                self.encoder.card,
            )

            # Get value
            value = self.critic(encoder_out)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return (
            action.item() + 1,
            log_prob.item(),
            value.item(),
        )  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None):
        """Propagate an observation through the encoder to update memory
        *without* sampling an action or storing any transition.  Useful for
        non-decision environment ticks such as the end of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent memory state (1-5 in game).
        """
        # Get or init memory for this player
        memory_in = self.get_recurrent_memory(player_id, device=device)

        with torch.no_grad():
            # Encode with memory to update memory state
            encoder_out = self.encoder.encode_batch(
                [state], memory_in=memory_in.unsqueeze(0), device=device
            )

            # Store updated memory
            if player_id is not None:
                self.set_recurrent_memory(player_id, encoder_out["memory_out"][0])

    # ------------------------------------------------------------------
    # Episode-level ingestion API
    # ------------------------------------------------------------------
    def store_episode_events(self, events: list) -> None:
        """Store a single episode as a chronological list of events.

        Each event must be one of:
          - Observation:
              {
                'kind': 'observation',
                'state': dict,
              }
          - Action:
              {
                'kind': 'action',
                'state': dict,
                'action': int (1-indexed),
                'log_prob': float,
                'value': float,
                'valid_actions': set[int],
                'reward': float,
                'win_label': float (optional),
                'final_return_label': float (optional),
                'seen_trump_mask_label': list[int] (optional; length len(TRUMP)),
                'unseen_trump_higher_than_hand_label': float (optional; 0/1),
              }

        """
        # Identify last action index to set done flag
        last_action_idx = -1
        for idx, ev in enumerate(events):
            if ev.get("kind") == "action":
                last_action_idx = idx

        for idx, ev in enumerate(events):
            if ev.get("kind") == "action":
                seen_mask = ev.get("seen_trump_mask_label") or [0] * len(TRUMP)
                unseen_higher = float(
                    ev.get("unseen_trump_higher_than_hand_label", 0.0) or 0.0
                )
                mask = self.get_action_mask(ev["valid_actions"], self.action_size)
                # Stage C: optional ISMCTS soft-teacher target pi'(a) over the
                # action set, plus whether the search produced a confident (ESS >=
                # floor) target. When absent the transition trains via plain PG.
                raw_target = ev.get("search_target")
                has_search_target = bool(ev.get("has_search_target", False))
                if has_search_target and raw_target is not None:
                    search_target = [float(x) for x in raw_target]
                else:
                    search_target = [0.0] * self.action_size
                    has_search_target = False
                record = {
                    "kind": "action",
                    "state": ev["state"],
                    "mask": mask,
                    "action": int(ev["action"]) - 1,
                    "reward": float(ev["reward"]),
                    "value": float(ev["value"]),
                    "log_prob": float(ev["log_prob"]),
                    "done": (idx == last_action_idx),
                    "win": float(ev.get("win_label", 0.0) or 0.0),
                    "final_return": float(ev.get("final_return_label", 0.0) or 0.0),
                    "secret_partner": float(ev.get("secret_partner_label", 0.0) or 0.0),
                    "points_rel": [
                        float(x) for x in (ev.get("points_label") or [0.0] * 5)
                    ],
                    "seen_trump_mask": [float(x) for x in seen_mask],
                    "unseen_trump_higher_than_hand": float(unseen_higher),
                    "search_target": search_target,
                    "has_search_target": has_search_target,
                }
                # Oracle mode: full-information observation captured at
                # decision time, consumed by _fill_oracle_values(). Only added
                # when collected so limited-mode events stay byte-identical.
                if ev.get("oracle_state") is not None:
                    record["oracle_state"] = ev["oracle_state"]
                self.events.append(record)
            else:
                # Observation: mask = all ones by default
                mask = torch.ones(self.action_size, dtype=torch.bool)
                record = {
                    "kind": "observation",
                    "state": ev["state"],
                    "mask": mask,
                }
                if ev.get("oracle_state") is not None:
                    record["oracle_state"] = ev["oracle_state"]
                self.events.append(record)

    @staticmethod
    def _gae_1d(rewards, values, dones, gamma, gae_lambda):
        """Plain GAE (Schulman et al. 2016) over aligned 1-D arrays.

        ``values``/``dones`` carry one trailing bootstrap element beyond
        ``rewards``. Returns (advantages, lambda-returns)."""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def compute_gae(self):
        """Compute GAE over action events; write results back into events."""
        action_idxs = [i for i, e in enumerate(self.events) if e["kind"] == "action"]
        if not action_idxs:
            return np.array([]), np.array([])

        rewards = np.array([self.events[i]["reward"] for i in action_idxs])
        values = np.array([self.events[i]["value"] for i in action_idxs] + [0.0])
        dones = np.array([self.events[i]["done"] for i in action_idxs] + [False])

        advantages, returns = self._gae_1d(
            rewards, values, dones, self.gamma, self.gae_lambda
        )

        for i, adv, ret in zip(action_idxs, advantages, returns):
            self.events[i]["advantage"] = float(adv)
            self.events[i]["return"] = float(ret)

        return advantages, returns

    def _fill_oracle_values(self, chunk_size: int = 64):
        """Batch-compute oracle values for every stored action event.

        Runs the recurrent oracle critic (no grad) over each episode's full
        event stream — observation events advance the memory exactly as they
        do for the limited critic's training forward — and writes
        ``value_oracle`` into action events. Called once at the top of
        update(); rollout/workers never touch the oracle."""
        kinds = [e["kind"] for e in self.events]
        segments = self._segments_from_events(kinds)
        if not segments:
            return
        for i, e in enumerate(self.events):
            if e.get("oracle_state") is None:
                raise ValueError(
                    "critic_mode='oracle' requires every event to carry an "
                    f"oracle_state (missing at event {i}); collect episodes "
                    "with collect_oracle=True"
                )
        with torch.no_grad():
            for c in range(0, len(segments), chunk_size):
                chunk = segments[c : c + chunk_size]
                seqs = [
                    [self.events[i]["oracle_state"] for i in range(s, e + 1)]
                    for (s, e) in chunk
                ]
                vals = self.oracle_critic.forward_sequences(seqs, device=device)
                for b, (s, e) in enumerate(chunk):
                    for t, i in enumerate(range(s, e + 1)):
                        if kinds[i] == "action":
                            self.events[i]["value_oracle"] = float(vals[b, t].item())

    def compute_gae_dual(self):
        """Dual GAE for oracle mode.

        Pass 1 (privileged baseline): advantages for the POLICY come from the
        oracle critic's values — the asymmetric-actor-critic payoff — and the
        oracle regresses to its own lambda-returns (``return_oracle``).
        Pass 2: the LIMITED critic's target ``return`` is computed from its
        own rollout values, exactly as compute_gae() does, so the deployable
        value head (and everything downstream of it) trains identically to a
        limited-mode run. Returns (advantages, oracle returns)."""
        action_idxs = [i for i, e in enumerate(self.events) if e["kind"] == "action"]
        if not action_idxs:
            return np.array([]), np.array([])

        rewards = np.array([self.events[i]["reward"] for i in action_idxs])
        dones = np.array([self.events[i]["done"] for i in action_idxs] + [False])
        values_oracle = np.array(
            [self.events[i]["value_oracle"] for i in action_idxs] + [0.0]
        )
        values_limited = np.array(
            [self.events[i]["value"] for i in action_idxs] + [0.0]
        )

        adv_oracle, ret_oracle = self._gae_1d(
            rewards, values_oracle, dones, self.gamma, self.gae_lambda
        )
        _, ret_limited = self._gae_1d(
            rewards, values_limited, dones, self.gamma, self.gae_lambda
        )

        for i, adv, ret_o, ret_l in zip(
            action_idxs, adv_oracle, ret_oracle, ret_limited
        ):
            self.events[i]["advantage"] = float(adv)
            self.events[i]["return_oracle"] = float(ret_o)
            self.events[i]["return"] = float(ret_l)

        return adv_oracle, ret_oracle

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e["state"] for e in self.events]
        masks_t = [
            (
                e["mask"].to(device)
                if isinstance(e["mask"], torch.Tensor)
                else torch.as_tensor(e["mask"], dtype=torch.bool, device=device)
            )
            for e in self.events
        ]
        kinds = [e["kind"] for e in self.events]
        return states, masks_t, kinds

    def _segments_from_events(self, kinds: list[str]):
        segments: list[tuple[int, int]] = []
        ev_idxs = list(range(len(self.events)))
        if not ev_idxs:
            return segments

        action_ev_idxs = [i for i in ev_idxs if kinds[i] == "action"]
        if not action_ev_idxs:
            return segments

        dones = [self.events[i]["done"] for i in action_ev_idxs]
        start = ev_idxs[0]
        a_ptr = 0
        for i in ev_idxs:
            if kinds[i] == "action":
                if dones[a_ptr]:
                    segments.append((start, i))
                    start = i + 1
                a_ptr += 1
        if start <= ev_idxs[-1]:
            segments.append((start, ev_idxs[-1]))
        return segments

    @staticmethod
    def _pad_to_bt(lst, lengths, fill):
        T = max(lengths)
        out = []
        for i, tlen in enumerate(lengths):
            pad = T - tlen
            if pad > 0:
                pad_shape = list(lst[i].shape)
                pad_shape[0] = pad
                pad_tensor = torch.full(
                    pad_shape,
                    fill,
                    device=lst[i].device,
                    dtype=lst[i].dtype,
                )
                out.append(torch.cat([lst[i], pad_tensor], dim=0))
            else:
                out.append(lst[i])
        return torch.stack(out, dim=0), T

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        old_value_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []
        secret_list_all = []
        points_list_all = []
        seen_trump_mask_list_all = []
        unseen_trump_higher_than_hand_list_all = []
        search_target_list_all = []
        has_search_list_all = []

        for seg_start, seg_end in batch:
            ev_range = [i for i in range(seg_start, seg_end + 1)]
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if kinds[i] == "action" else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, old_value_bt, ret_bt, adv_bt = [], [], [], [], []
            win_bt, final_ret_bt, secret_bt = [], [], []
            points_bt = []
            seen_trump_mask_bt = []
            unseen_trump_higher_than_hand_bt = []
            search_target_bt = []
            has_search_bt = []
            for i in ev_range:
                if kinds[i] == "action":
                    act_bt.append(
                        torch.tensor(
                            self.events[i]["action"], dtype=torch.long, device=device
                        )
                    )
                    olp_bt.append(
                        torch.tensor(
                            self.events[i]["log_prob"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    old_value_bt.append(
                        torch.tensor(
                            self.events[i]["value"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    ret_bt.append(
                        torch.tensor(
                            self.events[i]["return"], dtype=torch.float32, device=device
                        )
                    )
                    adv_bt.append(
                        torch.tensor(
                            self.events[i]["advantage"],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    win_lbl = self.events[i].get("win", 0.0) or 0.0
                    final_return_lbl = self.events[i].get("final_return", 0.0) or 0.0
                    secret_lbl = self.events[i].get("secret_partner", 0.0) or 0.0
                    win_bt.append(
                        torch.tensor(float(win_lbl), dtype=torch.float32, device=device)
                    )
                    final_ret_bt.append(
                        torch.tensor(
                            float(final_return_lbl), dtype=torch.float32, device=device
                        )
                    )
                    secret_bt.append(
                        torch.tensor(
                            float(secret_lbl), dtype=torch.float32, device=device
                        )
                    )
                    pts_lbl = self.events[i].get("points_rel", [0.0] * 5)
                    points_bt.append(
                        torch.tensor(pts_lbl, dtype=torch.float32, device=device)
                    )
                    seen_mask_lbl = self.events[i].get("seen_trump_mask") or [
                        0.0
                    ] * len(TRUMP)
                    seen_trump_mask_bt.append(
                        torch.tensor(seen_mask_lbl, dtype=torch.float32, device=device)
                    )
                    unseen_higher_lbl = float(
                        self.events[i].get("unseen_trump_higher_than_hand", 0.0) or 0.0
                    )
                    unseen_trump_higher_than_hand_bt.append(
                        torch.tensor(
                            unseen_higher_lbl, dtype=torch.float32, device=device
                        )
                    )
                    search_tgt_lbl = (
                        self.events[i].get("search_target") or [0.0] * self.action_size
                    )
                    search_target_bt.append(
                        torch.tensor(search_tgt_lbl, dtype=torch.float32, device=device)
                    )
                    has_search_bt.append(
                        torch.tensor(
                            1.0 if self.events[i].get("has_search_target") else 0.0,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    old_value_bt.append(
                        torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(
                        torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    secret_bt.append(
                        torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    points_bt.append(torch.zeros(5, dtype=torch.float32, device=device))
                    seen_trump_mask_bt.append(
                        torch.zeros(len(TRUMP), dtype=torch.float32, device=device)
                    )
                    unseen_trump_higher_than_hand_bt.append(
                        torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
                    search_target_bt.append(
                        torch.zeros(
                            self.action_size, dtype=torch.float32, device=device
                        )
                    )
                    has_search_bt.append(
                        torch.tensor(0.0, dtype=torch.float32, device=device)
                    )
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            old_value_list.append(torch.stack(old_value_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))
            secret_list_all.append(torch.stack(secret_bt, dim=0))
            points_list_all.append(torch.stack(points_bt, dim=0))
            seen_trump_mask_list_all.append(torch.stack(seen_trump_mask_bt, dim=0))
            unseen_trump_higher_than_hand_list_all.append(
                torch.stack(unseen_trump_higher_than_hand_bt, dim=0)
            )
            search_target_list_all.append(torch.stack(search_target_bt, dim=0))
            has_search_list_all.append(torch.stack(has_search_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        old_value_bt, _ = self._pad_to_bt(old_value_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        secret_bt, _ = self._pad_to_bt(secret_list_all, lengths, 0.0)
        points_bt, _ = self._pad_to_bt(points_list_all, lengths, 0.0)
        seen_trump_mask_bt, _ = self._pad_to_bt(seen_trump_mask_list_all, lengths, 0.0)
        unseen_trump_higher_than_hand_bt, _ = self._pad_to_bt(
            unseen_trump_higher_than_hand_list_all, lengths, 0.0
        )
        search_target_bt, _ = self._pad_to_bt(search_target_list_all, lengths, 0.0)
        has_search_bt, _ = self._pad_to_bt(has_search_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return (
            states_seqs,
            masks_bt,
            is_action_bt,
            actions_bt,
            old_lp_bt,
            old_value_bt,
            returns_bt,
            adv_bt,
            lengths_bt,
            win_bt,
            final_ret_bt,
            secret_bt,
            points_bt,
            seen_trump_mask_bt,
            unseen_trump_higher_than_hand_bt,
            search_target_bt,
            has_search_bt,
        )

    def _build_oracle_minibatch(self, batch, kinds):
        """Oracle-mode companion to _build_minibatch_tensors: per-segment
        oracle observation sequences plus (B, T) targets. Kept separate so
        the limited-mode tensor path stays byte-identical."""
        oracle_seqs = []
        ret_list = []
        old_v_list = []
        lengths = []
        for seg_start, seg_end in batch:
            ev_range = range(seg_start, seg_end + 1)
            lengths.append(seg_end - seg_start + 1)
            oracle_seqs.append([self.events[i]["oracle_state"] for i in ev_range])
            ret_list.append(
                torch.tensor(
                    [
                        self.events[i].get("return_oracle", 0.0)
                        if kinds[i] == "action"
                        else 0.0
                        for i in ev_range
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
            old_v_list.append(
                torch.tensor(
                    [
                        self.events[i].get("value_oracle", 0.0)
                        if kinds[i] == "action"
                        else 0.0
                        for i in ev_range
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            )
        returns_oracle_bt, _ = self._pad_to_bt(ret_list, lengths, 0.0)
        old_value_oracle_bt, _ = self._pad_to_bt(old_v_list, lengths, 0.0)
        return oracle_seqs, returns_oracle_bt, old_value_oracle_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        """Vectorized forward pass for training with recurrent memory.

        Args:
            states_input: List of B sequences, each a list of observation dicts
            masks_bt: (B, T, action_size) action masks
            lengths_bt: (B,) sequence lengths

        Returns:
            logits_bt, values_bt, win_logits_bt, ret_pred_bt, secret_logits_bt, points_pred_bt
        """
        B = len(states_input)
        T = masks_bt.size(1)

        # Initialize memory to zeros for each segment
        memory_init = torch.zeros((B, 256), device=device)

        # Encode sequences with memory
        encoder_out_seq = self.encoder.encode_sequences(
            states_input, memory_in=memory_init, device=device
        )

        # Extract features (B, T, 256)
        features_bt = encoder_out_seq["features"]

        # Build hand_ids_bt (B, T, N) from dict states
        N = 8  # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s["hand_ids"], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, : min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = features_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)
        # Token-less encoders (onehot-ff) emit no hand_tokens; their actors
        # ignore the argument.
        hand_tokens_bt = encoder_out_seq.get("hand_tokens")  # (B, T, N, d_token)
        flat_tokens = (
            hand_tokens_bt.reshape(B * T, N, -1) if hand_tokens_bt is not None else None
        )

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.encoder.card,
            hand_tokens=flat_tokens,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        # Critic values (dedicated deep trunk, decoupled from aux adapter)
        value_feat_bt = self.critic.value_trunk(features_bt)
        values_bt = self.critic.value_head(value_feat_bt).squeeze(-1)

        if self.critic.has_aux_heads:
            # Auxiliary preds share the shallow critic_adapter (gradients flow to encoder)
            aux_feat_bt = self.critic.critic_adapter(features_bt)
            win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
            ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
            secret_logits_bt = self.critic.secret_partner_head(aux_feat_bt).squeeze(-1)
            points_pred_bt = self.critic.points_head(aux_feat_bt)
            seen_trump_mask_logits_bt = self.critic.seen_trump_mask_logits(
                aux_feat_bt, self.encoder.card
            )
            unseen_trump_higher_than_hand_logits_bt = (
                self.critic.unseen_trump_higher_than_hand_logits(aux_feat_bt)
            )
        else:
            # No-aux variants: shape-correct zero placeholders keep the
            # downstream flatten/stats plumbing unchanged; update() skips the
            # aux losses entirely, so these never contribute gradients.
            zeros_bt = values_bt.detach() * 0.0
            win_logits_bt = zeros_bt
            ret_pred_bt = zeros_bt
            secret_logits_bt = zeros_bt
            points_pred_bt = zeros_bt.unsqueeze(-1).expand(B, T, 5)
            seen_trump_mask_logits_bt = zeros_bt.unsqueeze(-1).expand(B, T, len(TRUMP))
            unseen_trump_higher_than_hand_logits_bt = zeros_bt

        return (
            logits_bt,
            values_bt,
            win_logits_bt,
            ret_pred_bt,
            secret_logits_bt,
            points_pred_bt,
            seen_trump_mask_logits_bt,
            unseen_trump_higher_than_hand_logits_bt,
        )

    @staticmethod
    def _flatten_action_steps(
        is_action_bt,
        logits_bt,
        values_bt,
        actions_bt,
        old_lp_bt,
        old_value_bt,
        returns_bt,
        adv_bt,
        win_logits_bt,
        ret_pred_bt,
        win_bt,
        final_ret_bt,
        secret_logits_bt,
        secret_bt,
        masks_bt,
        seen_trump_mask_logits_bt,
        seen_trump_mask_bt,
        unseen_trump_higher_than_hand_logits_bt,
        unseen_trump_higher_than_hand_bt,
        search_target_bt,
        has_search_bt,
    ):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            old_value_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            secret_logits_bt.view(-1)[flat_mask],
            secret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
            seen_trump_mask_logits_bt.view(-1, seen_trump_mask_logits_bt.size(-1))[
                flat_mask
            ],
            seen_trump_mask_bt.view(-1, seen_trump_mask_bt.size(-1))[flat_mask],
            unseen_trump_higher_than_hand_logits_bt.view(-1)[flat_mask],
            unseen_trump_higher_than_hand_bt.view(-1)[flat_mask],
            search_target_bt.view(-1, search_target_bt.size(-1))[flat_mask],
            has_search_bt.view(-1)[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(
        self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
    ):
        probs_pick = probs_flat[:, pick_idx_t]
        probs_partner = probs_flat[:, partner_idx_t]
        probs_bury = probs_flat[:, bury_idx_t]
        probs_play = probs_flat[:, play_idx_t]
        return (
            self._entropy_from_probs(probs_pick),
            self._entropy_from_probs(probs_partner),
            self._entropy_from_probs(probs_bury),
            self._entropy_from_probs(probs_play),
        )

    def _per_head_weights(
        self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template
    ):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int(
            (count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0)
        )

        def per_head_weight(count: int) -> float:
            if heads_present == 0 or count == 0:
                return 0.0
            return total_count / (heads_present * float(count))

        w_pick = per_head_weight(count_pick)
        w_partner = per_head_weight(count_partner)
        w_bury = per_head_weight(count_bury)
        w_play = per_head_weight(count_play)

        head_weight = torch.ones_like(template)
        if w_pick > 0.0:
            head_weight[is_pick] = head_weight[is_pick] * w_pick
        else:
            head_weight[is_pick] = 0.0
        if w_partner > 0.0:
            head_weight[is_partner] = head_weight[is_partner] * w_partner
        else:
            head_weight[is_partner] = 0.0
        if w_bury > 0.0:
            head_weight[is_bury] = head_weight[is_bury] * w_bury
        else:
            head_weight[is_bury] = 0.0
        if w_play > 0.0:
            head_weight[is_play] = head_weight[is_play] * w_play
        else:
            head_weight[is_play] = 0.0

        return head_weight, is_pick, is_partner, is_bury

    def _actor_critic_losses(
        self,
        logits_flat,
        mask_flat,
        actions_flat,
        old_lp_flat,
        old_value_flat,
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
        search_target_flat,
        has_search_flat,
        anchor_logits_flat=None,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = (
            self._head_entropies(
                probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
            )
        )
        entropy_term = (
            self.entropy_coeff_pick * pick_entropy
            + self.entropy_coeff_partner * partner_entropy
            + self.entropy_coeff_bury * bury_entropy
            + self.entropy_coeff_play * play_entropy
        )

        ratios = torch.exp(new_lp_flat - old_lp_flat)
        head_weight, is_pick, is_partner, is_bury = self._per_head_weights(
            actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, ratios
        )

        eps_flat = torch.full_like(ratios, self.clip_epsilon_play)
        eps_flat[is_pick] = self.clip_epsilon_pick
        eps_flat[is_partner] = self.clip_epsilon_partner
        eps_flat[is_bury] = self.clip_epsilon_bury

        surr1 = ratios * adv_flat
        clipped = torch.clamp(ratios, 1 - eps_flat, 1 + eps_flat) * adv_flat
        pg_loss_elements = -torch.min(surr1, clipped)

        # Stage C searched-transition policy loss (A/B knob ``searched_ppo_weight``):
        # on transitions carrying a confident search target the PPO clip term is
        # scaled by ``searched_ppo_weight`` and the policy is also trained by
        # distillation toward pi' (below). At the default 0.0 this is the hard
        # PG-mask (PPO dropped on searched, distillation owns them): there PPO's
        # advantage is dominated by hidden-hand variance and blind to the small EV
        # gaps the teacher corrects, so the two objectives would fight. At 1.0 it is
        # the additive form (keep PPO AND add distillation); in between it leaves a
        # residual PG signal. Unsearched transitions always keep full PG.
        searched = has_search_flat > 0.5
        pg_keep = torch.where(
            searched,
            torch.full_like(pg_loss_elements, float(self.searched_ppo_weight)),
            torch.ones_like(pg_loss_elements),
        )
        pg_loss_elements = pg_loss_elements * pg_keep
        policy_loss = (pg_loss_elements * head_weight).mean()

        # Forward-KL distillation toward pi' on the searched transitions:
        #   L_distill = mean_searched( sum_a pi'(a) * (log pi'(a) - log pi_theta(a)) )
        # Reported KL(pi' || pi_theta), pi' entropy and the masked fraction are
        # detached for logging.
        if searched.any():
            pit = search_target_flat[searched]
            logp_theta = torch.log(probs_all[searched].clamp(min=1e-12))
            logp_it = torch.log(pit.clamp(min=1e-12))
            search_distill_per = (pit * (logp_it - logp_theta)).sum(dim=1)
            search_distill_loss = search_distill_per.mean()
            with torch.no_grad():
                teacher_kl = search_distill_per.mean()
                pi_target_entropy = -(pit * logp_it).sum(dim=1).mean()
        else:
            search_distill_loss = logits_flat.new_zeros(())
            teacher_kl = logits_flat.new_zeros(())
            pi_target_entropy = logits_flat.new_zeros(())
        masked_fraction = searched.to(torch.float32).mean()

        # Bidding-head KL anchor: forward KL(pi_ref || pi_theta) on the
        # pick/partner/bury rows toward the frozen reference logits (already
        # action-masked by the reference actor). Gradient flows through
        # log pi_theta only; the play head is untouched.
        anchor_kl = logits_flat.new_zeros(())
        if anchor_logits_flat is not None:
            bidding_rows = is_pick | is_partner | is_bury
            if bidding_rows.any():
                p_ref = F.softmax(anchor_logits_flat[bidding_rows], dim=-1)
                logp_ref = torch.log(p_ref.clamp(min=1e-12))
                logp_cur = torch.log(probs_all[bidding_rows].clamp(min=1e-12))
                anchor_kl = (p_ref * (logp_ref - logp_cur)).sum(dim=1).mean()

        returns_target = returns_flat.view(-1)
        values_old = old_value_flat.view(-1)
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(
            values_flat, returns_target, reduction="none"
        )
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target, reduction="none")
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

        actor_loss = (
            policy_loss
            - entropy_term
            + self.kl_coef * approx_kl_t
            + self.anchor_coeff * anchor_kl
        )
        return (
            actor_loss,
            critic_loss,
            approx_kl_t,
            (pick_entropy, partner_entropy, bury_entropy, play_entropy),
            search_distill_loss,
            {
                "teacher_kl": teacher_kl,
                "pi_target_entropy": pi_target_entropy,
                "masked_fraction": masked_fraction,
                "anchor_kl": anchor_kl,
            },
        )

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns. In oracle mode the policy's
        # advantages come from the privileged critic (asymmetric
        # actor-critic); the limited critic's targets are computed from its
        # own rollout values either way (see compute_gae_dual).
        oracle_active = self.critic_mode == "oracle" and self.oracle_critic is not None
        oracle_stats = None
        if oracle_active:
            self._fill_oracle_values()
            advantages, returns = self.compute_gae_dual()
            if advantages.size:
                # Explained variance of each critic against the SAME target —
                # the empirical discounted return G (lambda=1 with zero
                # values) — the headline variance-reduction diagnostic.
                acts = [e for e in self.events if e["kind"] == "action"]
                rew = np.array([e["reward"] for e in acts])
                dns = np.array([e["done"] for e in acts] + [False])
                zeros = np.zeros(len(acts) + 1)
                _, g_emp = self._gae_1d(rew, zeros, dns, self.gamma, 1.0)
                v_ora = np.array([e["value_oracle"] for e in acts])
                v_lim = np.array([e["value"] for e in acts])
                var_g = float(np.var(g_emp)) + 1e-8
                oracle_stats = {
                    "ev_oracle": float(1.0 - np.var(g_emp - v_ora) / var_g),
                    "ev_limited": float(1.0 - np.var(g_emp - v_lim) / var_g),
                }
        else:
            advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            "mean": float(np.mean(raw_advantages)),
            "std": float(np.std(raw_advantages)),
            "min": float(np.min(raw_advantages)),
            "max": float(np.max(raw_advantages)),
        }
        # Per-head RAW advantage std (diagnostic only; not used for
        # normalization). Advantages are normalized globally, so a head whose
        # raw std is small relative to the global std contributes a
        # correspondingly small policy-gradient after normalization -- the play
        # head's std vs pick's tells how far below the play-PG scale the play
        # entropy coefficient must sit to avoid an entropy-driven collapse to
        # uniform. ``advantages`` aligns 1:1 with the in-order action events.
        if advantages.size:
            groups = {k: set(v) for k, v in self.action_groups.items()}
            head_advs: dict[str, list] = {k: [] for k in groups}
            action_events = (e for e in self.events if e.get("kind") == "action")
            for e, adv in zip(action_events, raw_advantages):
                for k, s in groups.items():
                    if e["action"] in s:
                        head_advs[k].append(adv)
                        break
            advantage_stats["head_std"] = {
                k: (float(np.std(v)) if v else 0.0) for k, v in head_advs.items()
            }
            advantage_stats["head_n"] = {k: len(v) for k, v in head_advs.items()}

        value_target_stats = {
            "mean": float(np.mean(returns)) if advantages.size else 0.0,
            "std": float(np.std(returns)) if advantages.size else 0.0,
            "min": float(np.min(returns)) if advantages.size else 0.0,
            "max": float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get("kind") == "action":
                    e["advantage"] = float((e["advantage"] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds = self._prepare_training_views()
        segments = self._segments_from_events(kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups["pick"], device=device)
        partner_idx_tensor_static = torch.tensor(
            self.action_groups["partner"], device=device
        )
        bury_idx_tensor_static = torch.tensor(self.action_groups["bury"], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups["play"], device=device)

        t_build_end = time.time()

        # Training epochs – vectorized by batching segments
        forward_time = 0.0
        backward_time = 0.0
        step_time = 0.0
        optimizer_steps = 0
        early_stop_triggered = False
        last_approx_kl = 0.0

        # Instrumentation accumulators
        ent_pick_sum = 0.0
        ent_partner_sum = 0.0
        ent_bury_sum = 0.0
        ent_play_sum = 0.0
        ent_batches = 0
        value_loss_sum = 0.0
        value_loss_count = 0
        oracle_loss_sum = 0.0
        oracle_loss_count = 0
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0
        win_loss_sum = 0.0
        win_loss_count = 0
        return_loss_sum = 0.0
        return_loss_count = 0
        seen_trump_mask_loss_sum = 0.0
        seen_trump_mask_loss_count = 0
        unseen_trump_higher_than_hand_loss_sum = 0.0
        unseen_trump_higher_than_hand_loss_count = 0
        points_loss_sum = 0.0
        points_loss_count = 0
        secret_loss_sum = 0.0
        secret_loss_count = 0
        search_distill_loss_sum = 0.0
        teacher_kl_sum = 0.0
        pi_target_entropy_sum = 0.0
        masked_fraction_sum = 0.0
        search_distill_batches = 0
        anchor_kl_sum = 0.0
        anchor_batches = 0
        anchor_active = self._anchor_agent is not None and self.anchor_coeff > 0.0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start : mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    old_value_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                    secret_bt,
                    points_bt,
                    seen_trump_mask_bt,
                    unseen_trump_higher_than_hand_bt,
                    search_target_bt,
                    has_search_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds)

                # Vectorized forward
                t_fwd = time.time()
                (
                    logits_bt,
                    values_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    secret_logits_bt,
                    points_pred_bt,
                    seen_trump_mask_logits_bt,
                    unseen_trump_higher_than_hand_logits_bt,
                ) = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt,
                    logits_bt,
                    values_bt,
                    actions_bt,
                    old_lp_bt,
                    old_value_bt,
                    returns_bt,
                    adv_bt,
                    win_logits_bt,
                    ret_pred_bt,
                    win_bt,
                    final_ret_bt,
                    secret_logits_bt,
                    secret_bt,
                    masks_bt,
                    seen_trump_mask_logits_bt,
                    seen_trump_mask_bt,
                    unseen_trump_higher_than_hand_logits_bt,
                    unseen_trump_higher_than_hand_bt,
                    search_target_bt,
                    has_search_bt,
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    old_value_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    secret_logits_flat,
                    secret_labels_flat,
                    mask_flat,
                    seen_trump_mask_logits_flat,
                    seen_trump_mask_labels_flat,
                    unseen_trump_higher_than_hand_logits_flat,
                    unseen_trump_higher_than_hand_labels_flat,
                    search_target_flat,
                    has_search_flat,
                ) = flat

                # Bidding-head KL anchor: frozen-reference logits on the same
                # minibatch (no grad), flattened to the action rows like the
                # policy logits above.
                anchor_logits_flat = None
                if anchor_active:
                    with torch.no_grad():
                        ref_logits_bt = self._anchor_agent._forward_vectorized(
                            states_bt, masks_bt, lengths_bt
                        )[0]
                    anchor_logits_flat = ref_logits_bt.view(-1, ref_logits_bt.size(-1))[
                        is_action_bt.view(-1)
                    ]

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = actions_flat == self.pick_action_index
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = actions_flat == self.pass_action_index
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                    search_distill_loss,
                    search_distill_metrics,
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    old_value_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                    search_target_flat,
                    has_search_flat,
                    anchor_logits_flat=anchor_logits_flat,
                )
                last_approx_kl = float(approx_kl_t.item())

                if anchor_active:
                    anchor_kl_sum += search_distill_metrics["anchor_kl"].item()
                    anchor_batches += 1

                value_loss_sum += critic_loss.detach().item()
                value_loss_count += 1

                # Stage C distillation accumulation
                search_distill_loss_sum += search_distill_loss.detach().item()
                teacher_kl_sum += search_distill_metrics["teacher_kl"].item()
                pi_target_entropy_sum += search_distill_metrics[
                    "pi_target_entropy"
                ].item()
                masked_fraction_sum += search_distill_metrics["masked_fraction"].item()
                search_distill_batches += 1

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses (skipped entirely for no-aux architecture
                # variants: the placeholder logits carry no gradients, so the
                # losses would be meaningless constants anyway).
                if self.critic.has_aux_heads:
                    win_loss = F.binary_cross_entropy_with_logits(
                        win_logits_flat, win_labels_flat
                    )
                    return_loss = F.smooth_l1_loss(
                        ret_pred_flat / RETURN_SCALE,
                        final_ret_labels_flat / RETURN_SCALE,
                    )
                    secret_loss = F.binary_cross_entropy_with_logits(
                        secret_logits_flat, secret_labels_flat
                    )
                    # Per-player points auxiliary loss (regression on per-seat totals, 0–120)
                    # points_pred_flat and labels are (N, 5); smooth L1 stabilizes training.
                    points_pred_flat = points_pred_bt.view(-1, points_pred_bt.size(-1))[
                        is_action_bt.view(-1)
                    ]
                    points_labels_flat = points_bt.view(-1, points_bt.size(-1))[
                        is_action_bt.view(-1)
                    ]
                    points_loss = F.smooth_l1_loss(
                        points_pred_flat / POINTS_SCALE,
                        points_labels_flat / POINTS_SCALE,
                    )

                    seen_trump_mask_loss = F.binary_cross_entropy_with_logits(
                        seen_trump_mask_logits_flat,
                        seen_trump_mask_labels_flat,
                    )
                    unseen_trump_higher_than_hand_loss = (
                        F.binary_cross_entropy_with_logits(
                            unseen_trump_higher_than_hand_logits_flat,
                            unseen_trump_higher_than_hand_labels_flat,
                        )
                    )
                else:
                    aux_zero = torch.zeros((), device=device)
                    win_loss = aux_zero
                    return_loss = aux_zero
                    secret_loss = aux_zero
                    points_loss = aux_zero
                    seen_trump_mask_loss = aux_zero
                    unseen_trump_higher_than_hand_loss = aux_zero
                win_loss_sum += win_loss.detach().item()
                win_loss_count += 1
                return_loss_sum += return_loss.detach().item()
                return_loss_count += 1
                secret_loss_sum += secret_loss.detach().item()
                secret_loss_count += 1
                points_loss_sum += points_loss.detach().item()
                points_loss_count += 1
                seen_trump_mask_loss_sum += seen_trump_mask_loss.detach().item()
                seen_trump_mask_loss_count += 1
                unseen_trump_higher_than_hand_loss_sum += (
                    unseen_trump_higher_than_hand_loss.detach().item()
                )
                unseen_trump_higher_than_hand_loss_count += 1

                # Oracle value loss: with-grad forward of the privileged
                # critic on the same minibatch, clipped-MSE against its own
                # lambda-returns (same form as the limited critic's loss).
                # Its graph is disjoint from encoder/actor/critic, so sharing
                # total_loss.backward() cannot leak privileged gradients into
                # the policy trunk.
                oracle_loss = None
                if oracle_active:
                    (
                        oracle_seqs,
                        returns_oracle_bt,
                        old_value_oracle_bt,
                    ) = self._build_oracle_minibatch(batch, kinds)
                    t_fwd = time.time()
                    values_oracle_bt = self.oracle_critic.forward_sequences(
                        oracle_seqs, device=device
                    )
                    forward_time += time.time() - t_fwd
                    flat_idx = is_action_bt.view(-1)
                    v_o = values_oracle_bt.reshape(-1)[flat_idx]
                    ret_o = returns_oracle_bt.reshape(-1)[flat_idx]
                    old_o = old_value_oracle_bt.reshape(-1)[flat_idx]
                    v_o_clipped = old_o + torch.clamp(
                        v_o - old_o, -self.value_clip_epsilon, self.value_clip_epsilon
                    )
                    oracle_loss = torch.max(
                        F.mse_loss(v_o, ret_o, reduction="none"),
                        F.mse_loss(v_o_clipped, ret_o, reduction="none"),
                    ).mean()
                    oracle_loss_sum += oracle_loss.detach().item()
                    oracle_loss_count += 1

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = (
                    actor_loss
                    + self.search_distill_coeff * search_distill_loss
                    + self.value_loss_coeff * critic_loss
                    + self.win_loss_coeff * win_loss
                    + self.return_loss_coeff * return_loss
                    + self.secret_loss_coeff * secret_loss
                    + self.points_loss_coeff * points_loss
                    + self.seen_trump_mask_loss_coeff * seen_trump_mask_loss
                    + self.unseen_trump_higher_than_hand_loss_coeff
                    * unseen_trump_higher_than_hand_loss
                )
                if oracle_loss is not None:
                    total_loss = total_loss + self.oracle_value_loss_coeff * oracle_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                if oracle_active:
                    self.oracle_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                if oracle_active:
                    torch.nn.utils.clip_grad_norm_(
                        self.oracle_critic.parameters(), self.max_grad_norm
                    )
                    self.oracle_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e["kind"] == "action")

        if oracle_stats is not None:
            oracle_stats["value_loss"] = self.oracle_value_loss_coeff * (
                oracle_loss_sum / max(oracle_loss_count, 1)
            )

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            "build_s": t_build_end - t_build_start,
            "forward_s": forward_time,
            "backward_s": backward_time,
            "step_s": step_time,
            "total_update_s": t_end - t_update_start,
            "optimizer_steps": optimizer_steps,
        }
        return {
            "advantage_stats": advantage_stats,
            "value_target_stats": value_target_stats,
            "oracle": oracle_stats,
            "num_transitions": transitions,
            "approx_kl": last_approx_kl,
            "early_stop": early_stop_triggered,
            "timing": timing,
            "head_entropy": {
                "pick": (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                "partner": (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                "bury": (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                "play": (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            "pick_pass_adv": {
                "pick_mean": (pick_adv_sum / pick_adv_count)
                if pick_adv_count > 0
                else 0.0,
                "pick_count": pick_adv_count,
                "pass_mean": (pass_adv_sum / pass_adv_count)
                if pass_adv_count > 0
                else 0.0,
                "pass_count": pass_adv_count,
            },
            "distill": {
                "loss": self.search_distill_coeff
                * (search_distill_loss_sum / max(search_distill_batches, 1)),
                "teacher_kl": teacher_kl_sum / max(search_distill_batches, 1),
                "pi_target_entropy": pi_target_entropy_sum
                / max(search_distill_batches, 1),
                "pg_masked_fraction": masked_fraction_sum
                / max(search_distill_batches, 1),
            },
            "anchor": {
                "active": anchor_active,
                "kl": anchor_kl_sum / max(anchor_batches, 1),
                "loss": self.anchor_coeff * (anchor_kl_sum / max(anchor_batches, 1)),
            },
            "critic_losses": {
                "value": self.value_loss_coeff
                * (value_loss_sum / max(value_loss_count, 1)),
                "win": self.win_loss_coeff * (win_loss_sum / max(win_loss_count, 1)),
                "return": self.return_loss_coeff
                * (return_loss_sum / max(return_loss_count, 1)),
                "points": self.points_loss_coeff
                * (points_loss_sum / max(points_loss_count, 1)),
                "secret_partner": self.secret_loss_coeff
                * (secret_loss_sum / max(secret_loss_count, 1)),
                "seen_trump_mask": self.seen_trump_mask_loss_coeff
                * (seen_trump_mask_loss_sum / max(seen_trump_mask_loss_count, 1)),
                "unseen_trump_higher_than_hand": self.unseen_trump_higher_than_hand_loss_coeff
                * (
                    unseen_trump_higher_than_hand_loss_sum
                    / max(unseen_trump_higher_than_hand_loss_count, 1)
                ),
            },
        }

    def save(self, filepath):
        """Save model parameters.

        Oracle-mode agents additionally persist the privileged critic under
        OPTIONAL keys; every existing checkpoint consumer ignores them, and
        limited-mode saves are byte-compatible with the historical format."""
        payload = {
            "arch": self.arch_name,
            "encoder_state_dict": self.encoder.state_dict(),
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        if self.oracle_critic is not None:
            payload["critic_mode"] = self.critic_mode
            payload["oracle_state_dict"] = self.oracle_critic.state_dict()
            payload["oracle_optimizer"] = self.oracle_optimizer.state_dict()
        torch.save(payload, filepath)

    def load(self, filepath, load_optimizers: bool = True, checkpoint=None):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        checkpoint : dict, optional
            A checkpoint already read via ``torch.load``. Lets callers that
            build many agents from one file (e.g. the game server) skip the
            per-agent disk read; ``load_state_dict`` copies tensors, so agents
            never alias the shared checkpoint.
        """
        if checkpoint is None:
            checkpoint = torch.load(filepath, map_location=device)

        # Architecture guard: refuse to copy tensors across architecture
        # variants. Checkpoints predating the registry carry no "arch" key
        # and are, by construction, the full architecture.
        ckpt_arch = checkpoint.get("arch", "full")
        if ckpt_arch != self.arch_name:
            raise ValueError(
                f"Checkpoint arch '{ckpt_arch}' does not match agent arch "
                f"'{self.arch_name}' ({filepath}). Construct the agent with "
                f"arch='{ckpt_arch}' or use ppo.load_agent()."
            )

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        # The critic load is non-strict because older checkpoints (e.g. the
        # pfsp-ppo-30M-baseline tag, final_pfsp_swish_ppo.pt) predate the
        # dedicated deep ``value_trunk`` and trained the value head off
        # ``critic_adapter`` directly. A naive strict=False load leaves the new
        # ``value_trunk.*`` at random init, so ``forward`` bootstraps the value
        # off noise (~uncorrelated with the trained value fn) -- silently
        # corrupting any ISMCTS critic bootstrap. Detect that case and re-point
        # the value path at the trained ``critic_adapter`` so legacy checkpoints
        # evaluate exactly as they were trained (inference-compatibility shim).
        missing, unexpected = self.critic.load_state_dict(
            checkpoint["critic_state_dict"], strict=False
        )
        ckpt_has_value_trunk = any(
            k.startswith("value_trunk") for k in checkpoint["critic_state_dict"]
        )
        if (
            any(k.startswith("value_trunk") for k in missing)
            and not ckpt_has_value_trunk
        ):
            self.critic.value_trunk = self.critic.critic_adapter
            print(
                f"Note: {filepath} predates value_trunk; routing the value head "
                f"through the trained critic_adapter (legacy critic compatibility)."
            )
        elif missing or unexpected:
            print(
                f"Warning: critic load mismatch for {filepath}: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )

        # Oracle critic: optional checkpoint keys. An oracle-mode agent
        # resuming a limited checkpoint keeps its fresh-init oracle — the
        # expected baseline→oracle warm start. Limited agents ignore the keys.
        if self.oracle_critic is not None:
            if "oracle_state_dict" in checkpoint:
                self.oracle_critic.load_state_dict(checkpoint["oracle_state_dict"])
            else:
                print(
                    f"Note: {filepath} has no oracle critic; warm-starting the "
                    "oracle value network from fresh init."
                )

        # Optimizers: best-effort restore; fall back to fresh states if shapes changed
        if load_optimizers:
            try:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
                # Refresh LR ratios after loading optimizer state (ratios may have changed)
                self._actor_lr_ratios = self._capture_actor_lr_ratios()
            except ValueError as e:
                print(f"Warning: could not load optimizer state from {filepath}: {e}")
            if self.oracle_optimizer is not None and "oracle_optimizer" in checkpoint:
                try:
                    self.oracle_optimizer.load_state_dict(
                        checkpoint["oracle_optimizer"]
                    )
                except ValueError as e:
                    print(
                        f"Warning: could not load oracle optimizer state from "
                        f"{filepath}: {e}"
                    )

        # Reset memory states after loading
        self._player_memories = {}


def load_agent(
    filepath,
    *,
    load_optimizers: bool = False,
    checkpoint=None,
) -> "PPOAgent":
    """Construct a PPOAgent matching a checkpoint's recorded metadata.

    Reads the checkpoint once, extracts the architecture name (``arch``,
    default "full" for pre-registry checkpoints) and ``critic_mode``
    (default "limited"), builds the matching agent, and loads the weights.
    This is the canonical loader for eval tooling, league members, and the
    game server — anywhere the caller does not already know the arch.
    """
    from sheepshead import ACTIONS

    if checkpoint is None:
        checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint.get("arch", "full")
    critic_mode = checkpoint.get("critic_mode", "limited")
    agent = PPOAgent(len(ACTIONS), critic_mode=critic_mode, arch=arch)
    agent.load(filepath, load_optimizers=load_optimizers, checkpoint=checkpoint)
    return agent
