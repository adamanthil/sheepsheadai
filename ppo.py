import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS, DECK_IDS, UNDER_TOKEN
from state_encoder import StateEncoder, CardEmbeddingConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreNormResidual(nn.Module):
    """Pre-norm residual MLP block: y = x + Linear(LN(x) -> hidden -> act -> dropout -> dim)."""
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1, activation=F.silu):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.activation(self.fc1(y))
        y = self.dropout(self.fc2(y))
        return x + y


class SharedRecurrentBackbone(nn.Module):
    """Shared encoder + LSTM + post-LSTM trunk used by both actor and critic.

    Layout:
      - enc_proj: Linear(state_size -> 256)
      - enc_blocks: 2 × PreNormResidual(256)
      - lstm: LSTM(256 -> 256)
      - trunk_blocks: 2 × PreNormResidual(256)
    """

    def __init__(self, state_size: int, activation: str = 'swish'):
        super().__init__()
        self.state_size = state_size

        if activation == 'swish':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder projection and residual blocks (256 width)
        self.enc_proj = nn.Linear(state_size, 256)
        self.enc_block1 = PreNormResidual(256, 256, dropout=0.1, activation=self.activation)
        self.enc_block2 = PreNormResidual(256, 256, dropout=0.1, activation=self.activation)

        # Recurrent core
        self.lstm = nn.LSTM(256, 256, batch_first=True)

        # Shared post-LSTM trunk (256 width)
        self.trunk_block1 = PreNormResidual(256, 256, dropout=0.1, activation=self.activation)
        self.trunk_block2 = PreNormResidual(256, 256, dropout=0.1, activation=self.activation)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param.data, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

    def forward(self, state: torch.Tensor, hidden_in=None, return_hidden: bool = False):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Encoder (expects width=state_size; we set state_size=256 for pre-encoded features)
        x = self.activation(self.enc_proj(state))
        x = self.enc_block1(x)
        x = self.enc_block2(x)

        # LSTM expects time dimension
        x = x.unsqueeze(1)
        lstm_out, new_hidden = self.lstm(x, hidden_in)
        feat = lstm_out.squeeze(1)

        # Shared trunk
        feat = self.trunk_block1(feat)
        feat = self.trunk_block2(feat)

        if return_hidden:
            return feat, new_hidden
        return feat

    def forward_sequence(self, states_bt: torch.Tensor, lengths: torch.Tensor, return_hidden: bool = False):
        """Vectorized forward over a batch of sequences.

        Parameters
        ----------
        states_bt : Tensor (B, T, state_size)
        lengths : LongTensor (B,) true sequence lengths
        return_hidden : bool

        Returns
        -------
        feat_bt : Tensor (B, T, 256)
        (optionally) (h, c) final LSTM state
        """
        # Encoder over (B, T, state_size). If inputs are pre-encoded features of width 256,
        # enc_proj will learn a light adapter.
        x = self.activation(self.enc_proj(states_bt))
        x = self.enc_block1(x)
        x = self.enc_block2(x)

        # Pack and run LSTM
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.lstm(packed)
        lstm_out_bt, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Shared trunk on (B, T, 256)
        feat_bt = self.trunk_block1(lstm_out_bt)
        feat_bt = self.trunk_block2(feat_bt)

        if return_hidden:
            return feat_bt, hidden
        return feat_bt


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network with an LSTM core and separate linear heads for the
    pick / partner-selection / bury / play phases. The four heads' logits are
    concatenated back into the full action space order so existing masking
    logic continues to work unchanged.
    """

    def __init__(self,
                 backbone: SharedRecurrentBackbone,
                 action_size,
                 action_groups,
                 activation='swish',
                 *,
                 d_card: int,
                 d_token: int,
                 map_cid_to_play_action_index: torch.Tensor,
                 map_cid_to_bury_action_index: torch.Tensor,
                 map_cid_to_under_action_index: torch.Tensor,
                 call_action_global_indices: torch.Tensor,
                 call_card_ids: torch.Tensor,
                 play_under_action_index: int):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.backbone = backbone  # registered; owns shared params
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Activation selector
        if activation == 'swish':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # === Actor-specific adapter after shared trunk ===
        self.actor_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )

        # === Heads ===
        self.pick_head = nn.Linear(256, len(action_groups['pick']))

        # Partner head is split: basic (ALONE, JD PARTNER) + CALL actions via two-tower
        self.partner_basic_head = nn.Linear(256, 2)

        # Bury and Play are produced via pointer over hand tokens; keep a dedicated
        # scalar for PLAY UNDER which is not a hand-slot action
        self.play_under_head = nn.Linear(256, 1)

        # Buffer to hold hidden states for each player id (1-5).  Populated on the fly.
        self._hidden_states = {}

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
        self._map_cid_to_under_action_index = map_cid_to_under_action_index.clone().long()
        self._call_action_global_indices = call_action_global_indices.clone().long()
        self._call_card_ids = call_card_ids.clone().long()
        self._play_under_action_index = int(play_under_action_index)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------ internal helpers ------------------------
    def _score_cards_two_tower(self, feat: torch.Tensor, card_embedding: nn.Embedding) -> torch.Tensor:
        """Return per-card scores S for all 34 card ids using two-tower scorer.
        feat: (B, 256)
        Returns: (B, 34)
        """
        q = self.tw_Wg(feat)                      # (B, k)        <- tower 1 (actor features)
        table = card_embedding.weight             # (34, d_card)
        K = self.tw_We(table)                     # (34, k)       <- tower 2 (card embeddings)
        S = torch.matmul(q, K.t())    # (B, 34)       <- similarity scores
        return S

    def _score_hand_pointer(self, feat: torch.Tensor, hand_ids: torch.Tensor,
                            card_embedding: nn.Embedding,
                            token_mlp_simple: nn.Module) -> torch.Tensor:
        """Score each hand slot using pointer scorer.
        feat: (B, 256); hand_ids: (B, N)
        Returns: s_slots (B, N)
        """
        B, N = hand_ids.size(0), hand_ids.size(1)
        emb = card_embedding(hand_ids)                          # (B, N, d_card)
        tok = token_mlp_simple(emb)                             # (B, N, d_token)
        g = self.pointer_Wg(feat).unsqueeze(1).expand(B, N, -1) # (B, N, h)
        t = self.pointer_Wt(tok)                                # (B, N, h)
        e = torch.tanh(g + t)                             # (B, N, h)
        s = self.pointer_v(e).squeeze(-1)                       # (B, N)
        return s

    def _build_logits_from_features(
        self,
        actor_features: torch.Tensor,
        hand_ids: torch.Tensor,
        card_embedding: nn.Embedding,
        token_mlp_simple: nn.Module,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble full action logits from actor features and hand ids.

        Parameters
        ----------
        actor_features : Tensor (K, 256)
        hand_ids : LongTensor (K, N)
        card_embedding : nn.Embedding
        token_mlp_simple : nn.Module
        action_mask : BoolTensor (K, A) | None

        Returns
        -------
        logits : Tensor (K, A)
        """
        device = actor_features.device
        K = actor_features.size(0)
        logits = torch.full((K, self.action_size), -1e8, device=device)

        # PICK / PASS
        pick_logits = self.pick_head(actor_features) / max(self.temperature_pick, 1e-6)
        logits[:, self.action_groups['pick']] = pick_logits

        # PARTNER: basic (ALONE, JD PARTNER)
        partner_basic = self.partner_basic_head(actor_features) / max(self.temperature_partner, 1e-6)
        idx_alone = ACTION_IDS["ALONE"] - 1
        idx_jd = ACTION_IDS["JD PARTNER"] - 1
        logits[:, idx_alone] = partner_basic[:, 0]
        logits[:, idx_jd] = partner_basic[:, 1]

        # PARTNER: CALL actions via two-tower card scoring
        card_scores = self._score_cards_two_tower(actor_features, card_embedding)  # (K, 34)
        call_scores = card_scores[:, self._call_card_ids.to(device)] / max(self.temperature_partner, 1e-6)
        logits[:, self._call_action_global_indices.to(device)] = call_scores

        # Pointer scores over hand tokens (compute once, reuse for bury/under/play)
        slot_scores = self._score_hand_pointer(actor_features, hand_ids, card_embedding, token_mlp_simple)  # (K, N)
        K_, N = slot_scores.size(0), slot_scores.size(1)
        cids = hand_ids.long()

        # BURY and UNDER scatter
        idx_bury = self._map_cid_to_bury_action_index.to(device)[cids]   # (K, N)
        idx_under = self._map_cid_to_under_action_index.to(device)[cids] # (K, N)
        for i in range(N):
            b_idx = idx_bury[:, i]
            u_idx = idx_under[:, i]
            valid_b = b_idx.ge(0)
            valid_u = u_idx.ge(0)
            if valid_b.any():
                logits.view(K_, -1)[valid_b, b_idx[valid_b]] = slot_scores[valid_b, i] / max(self.temperature_bury, 1e-6)
            if valid_u.any():
                logits.view(K_, -1)[valid_u, u_idx[valid_u]] = slot_scores[valid_u, i] / max(self.temperature_bury, 1e-6)

        # PLAY scatter
        idx_play = self._map_cid_to_play_action_index.to(device)[cids]  # (K, N)
        for i in range(N):
            p_idx = idx_play[:, i]
            valid_p = p_idx.ge(0)
            if valid_p.any():
                logits.view(K_, -1)[valid_p, p_idx[valid_p]] = slot_scores[valid_p, i] / max(self.temperature_play, 1e-6)

        # PLAY UNDER scalar
        if self._play_under_action_index is not None:
            play_under_logit = self.play_under_head(actor_features).squeeze(-1) / max(self.temperature_play, 1e-6)
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
    def reset_hidden(self):
        """Erase all stored hidden states (call at the start of every new game)."""
        self._hidden_states = {}

    def set_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
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
        state,
        action_mask=None,
        player_id=None,
        hidden_in=None,
        return_hidden=False,
        hand_ids: torch.Tensor = None,
        card_embedding: nn.Embedding = None,
        token_mlp_simple: nn.Module = None
    ):
        """Unified forward pass supporting both cached and explicit hidden states.

        Parameters
        ----------
        state : Tensor  (batch, state_size)  OR (state_size,) for single sample
        action_mask : Bool Tensor broadcastable to (batch, action_size) or None
        player_id : int | None  – if provided for single-sample inference and
            hidden_in is None, uses and updates an internal hidden cache.
        hidden_in : tuple(h, c) | None – explicit LSTM state for training-time
            sequence unrolling. When provided, the internal cache is ignored.
        return_hidden : bool – if True, also return the new hidden state.
        """

        logits, new_hidden = self._compute_logits_and_hidden(state, action_mask, player_id, hidden_in,
                                                             hand_ids=hand_ids,
                                                             card_embedding=card_embedding,
                                                             token_mlp_simple=token_mlp_simple)
        probs = F.softmax(logits, dim=-1)
        if return_hidden:
            return probs, new_hidden
        return probs

    def forward_with_logits(
        self,
        state,
        action_mask=None,
        player_id=None,
        hidden_in=None,
        return_hidden=False,
        hand_ids: torch.Tensor = None,
        card_embedding: nn.Embedding = None,
        token_mlp_simple: nn.Module = None
    ):
        """Forward pass that also returns pre-softmax logits.

        Returns
        -------
        probs : Tensor
        logits : Tensor
        (optionally) new_hidden : tuple(h, c)
        """
        logits, new_hidden = self._compute_logits_and_hidden(
            state,
            action_mask,
            player_id,
            hidden_in,
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            token_mlp_simple=token_mlp_simple
        )
        probs = F.softmax(logits, dim=-1)
        if return_hidden:
            return probs, logits, new_hidden
        return probs, logits

    def _compute_logits_and_hidden(
        self,
        state,
        action_mask=None,
        player_id=None,
        hidden_in=None,
        hand_ids: torch.Tensor = None,
        card_embedding: nn.Embedding = None,
        token_mlp_simple: nn.Module = None
    ):
        """Core helper to compute masked logits and manage hidden state updates."""
        # Ensure 2-D tensor (batch_first)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        batch_size = state.size(0)

        # Choose hidden state source (for shared LSTM)
        chosen_hidden = hidden_in
        if chosen_hidden is None and player_id is not None and batch_size == 1:
            chosen_hidden = self._hidden_states.get(player_id, None)

        # Shared backbone → features (256) + new hidden
        backbone_features, new_hidden = self.backbone(state, hidden_in=chosen_hidden, return_hidden=True)

        # Actor adapter to produce actor-specific features
        actor_features = self.actor_adapter(backbone_features)

        # Persist hidden state for this player (only if single sample and not using explicit hidden)
        if hidden_in is None and player_id is not None and batch_size == 1:
            self._hidden_states[player_id] = (new_hidden[0].detach(), new_hidden[1].detach())

        # Build logits using shared helper
        logits = self._build_logits_from_features(
            actor_features=actor_features,
            hand_ids=hand_ids,
            card_embedding=card_embedding,
            token_mlp_simple=token_mlp_simple,
            action_mask=action_mask,
        )

        return logits, new_hidden


class RecurrentCriticNetwork(nn.Module):
    """Critic head using the shared backbone with a critic-specific adapter layer."""

    def __init__(self, backbone: SharedRecurrentBackbone, activation='swish'):
        super().__init__()
        # Store a non-registered reference to the shared backbone to avoid
        # duplicating parameters in the critic and to keep optimizer ownership
        object.__setattr__(self, "_backbone", backbone)

        if activation == 'swish':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.SiLU()
        )
        self.value_head = nn.Linear(256, 1)
        # Auxiliary heads (trained on detached critic adapter features)
        self.win_head = nn.Linear(256, 1)
        self.return_head = nn.Linear(256, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state, hidden_in=None, return_hidden=False):
        feat, new_hidden = self._backbone(state, hidden_in=hidden_in, return_hidden=True)
        feat = self.critic_adapter(feat)
        value = self.value_head(feat)
        if return_hidden:
            return value, new_hidden
        return value

    def aux_predictions(self, state, hidden_in=None, return_hidden: bool = False):
        """Return auxiliary predictions as scalars: (win_prob, expected_final_return).

        Parameters
        ----------
        state : Tensor
            Shape (state_size,) or (1, state_size)
        hidden_in : tuple(h, c) | None
            Optional LSTM hidden state to use for recurrent context
        return_hidden : bool
            If True, also return the new hidden state from the backbone

        Returns
        -------
        win_prob : float
            Sigmoid of win logits
        expected_final_return : float
            Linear head output for expected final return
        (optionally) new_hidden : tuple(h, c)
        """
        with torch.no_grad():
            feat, new_hidden = self._backbone(state, hidden_in=hidden_in, return_hidden=True)
            aux_feat = self.critic_adapter(feat)
            win_logit = self.win_head(aux_feat).squeeze(-1)
            expected_return = self.return_head(aux_feat).squeeze(-1)
        win_prob_t = torch.sigmoid(win_logit)
        win_prob = float(win_prob_t.item())
        expected_final = float(expected_return.item())
        if return_hidden:
            return win_prob, expected_final, new_hidden
        return win_prob, expected_final

class PPOAgent:
    def __init__(self, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
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
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + UNDER_ACTIONS)})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'partner': sorted(partner_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

        # Explicit indices for PICK and PASS actions (0-indexed)
        self.pick_action_index = ACTION_IDS["PICK"] - 1
        self.pass_action_index = ACTION_IDS["PASS"] - 1

        # Shared backbone and networks
        self.state_encoder = StateEncoder(card_config=CardEmbeddingConfig()).to(device)
        self.backbone = SharedRecurrentBackbone(self.state_size, activation=activation).to(device)
        # Build mappings before constructing actor
        (
            map_cid_to_play_action_index,
            map_cid_to_bury_action_index,
            map_cid_to_under_action_index,
            call_action_global_indices,
            call_card_ids,
            play_under_action_index,
        ) = self._build_action_index_mappings()

        self.actor = MultiHeadRecurrentActorNetwork(
            self.backbone,
            action_size,
            self.action_groups,
            activation=activation,
            d_card=self.state_encoder.d_card_dim,
            d_token=self.state_encoder.d_token_dim,
            map_cid_to_play_action_index=map_cid_to_play_action_index,
            map_cid_to_bury_action_index=map_cid_to_bury_action_index,
            map_cid_to_under_action_index=map_cid_to_under_action_index,
            call_action_global_indices=call_action_global_indices,
            call_card_ids=call_card_ids,
            play_under_action_index=play_under_action_index,
        ).to(device)

        self.critic = RecurrentCriticNetwork(self.backbone, activation=activation).to(device)

        # Optimizers (actor owns backbone; include state encoder params)
        # Use encoder param groups to reduce LR for card embeddings to 0.1x
        encoder_groups = self.state_encoder.param_groups(base_lr=lr_actor, card_lr_scale=0.1)
        actor_groups = [
            {'params': self.actor.parameters(), 'lr': lr_actor},
            *encoder_groups,
        ]
        self.actor_optimizer = optim.Adam(actor_groups)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.95
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_partner = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.3
        self.clip_epsilon_pick = 0.15
        self.clip_epsilon_partner = 0.2
        self.clip_epsilon_bury = 0.1
        self.clip_epsilon_play = 0.15
        self.value_clip_epsilon = 0.15

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.0

        # Auxiliary loss coefficients (aux heads are detached from trunk)
        self.win_loss_coeff = 0.1
        self.return_loss_coeff = 0.1

        # Storage for trajectory data
        self.reset_storage()

        # --------------------------------------------------
        # Partner-call mixture: epsilon floor over CALL actions
        # --------------------------------------------------
        # Subset of partner head actions eligible for CALL-uniform mixing
        self.partner_call_subindices = sorted(
            {ACTION_IDS["JD PARTNER"] - 1} | {ACTION_IDS[a] - 1 for a in CALL_ACTIONS}
        )
        # Blending coefficient ε (0 disables; typical 0.02–0.10)
        self.partner_call_epsilon = 0.0

        # PASS-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pass_floor_epsilon = 0.0
        # PICK-floor mixing epsilon (applies on PICK/PASS decision steps)
        self.pick_floor_epsilon = 0.0

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
        idx_to_cid = {idx: cid for idx, cid in zip(idx_call_global_list, call_card_ids_list)}
        call_card_ids = torch.tensor([idx_to_cid[int(x)] for x in idx_call_global.tolist()], dtype=torch.long)

        return map_cid_to_play_action_index, map_cid_to_bury_action_index, map_cid_to_under_action_index, idx_call_global, call_card_ids, play_under_index

    def set_head_temperatures(self, pick: float | None = None, partner: float | None = None, bury: float | None = None, play: float | None = None):
        """Convenience proxy to set per-head temperatures on the actor."""
        self.actor.set_temperatures(pick=pick, partner=partner, bury=bury, play=play)

    def set_partner_call_epsilon(self, eps: float):
        """Set ε for uniform CALL mixture on partner steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.partner_call_epsilon = eps

    def set_pass_floor_epsilon(self, eps: float):
        """Set ε for PASS-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pass_floor_epsilon = eps

    def set_pick_floor_epsilon(self, eps: float):
        """Set ε for PICK-floor mixing on pick/pass steps (0.0 disables)."""
        eps = float(eps)
        if eps < 0.0:
            eps = 0.0
        if eps > 0.2:
            eps = 0.2
        self.pick_floor_epsilon = eps

    def _apply_epsilon_mixing(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply partner-call ε mixture and PASS-floor ε mixture to probability
        distributions in a batched way without in-place ops.

        Parameters
        ----------
        probs : Tensor
            Shape (B, A) probabilities over actions
        mask : Bool Tensor
            Shape (B, A) of valid actions
        """
        mixed = probs
        if self.partner_call_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            call_vec = torch.zeros(A, dtype=torch.bool, device=mixed.device)
            call_idx = torch.tensor(self.partner_call_subindices, dtype=torch.long, device=mixed.device)
            call_vec[call_idx] = True
            valid_call = mask & call_vec.view(1, A).expand(B, A)
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(
                has,
                valid_call.float() / count.clamp_min(1.0),
                torch.zeros_like(mixed),
            )
            eps = torch.where(
                has,
                torch.full_like(count, self.partner_call_epsilon),
                torch.zeros_like(count),
            )
            mixed = (1.0 - eps) * mixed + eps * ucall

        if self.pick_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pick = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand(B, A)
            valid_pick = mask[:, self.pick_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * mixed + self.pick_floor_epsilon * one_hot_pick,
                mixed,
            )

        if self.pass_floor_epsilon > 0.0:
            B, A = mixed.size(0), mixed.size(1)
            one_hot_pass = torch.zeros(A, dtype=mixed.dtype, device=mixed.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand(B, A)
            valid_pass = mask[:, self.pass_action_index].unsqueeze(-1)
            mixed = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * mixed + self.pass_floor_epsilon * one_hot_pass,
                mixed,
            )

        return mixed

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
        """Clear hidden states in the actor (call at the start of every new game)."""
        if hasattr(self.actor, 'reset_hidden'):
            self.actor.reset_hidden()

    def get_action_probs_with_logits(self, state, valid_actions, player_id=None):
        """Return post-mixture action probabilities and pre-mix logits for a single dict state.

        Applies partner CALL-uniform mixture if enabled. Keeps PPO on-policy by
        exposing the same transformed distribution that sampling uses.
        """
        # Encode dict state → (1, 256) features
        state_t = self.state_encoder.encode_batch([state], device=device)
        action_mask_t = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)

        # Extract hand_ids for pointer and pass encoder modules explicitly
        hand_ids_np = state['hand_ids']
        hand_ids_t = torch.as_tensor(hand_ids_np, dtype=torch.long, device=device).view(1, -1)

        with torch.no_grad():
            probs, logits = self.actor.forward_with_logits(
                state_t,
                action_mask_t,
                player_id,
                hand_ids=hand_ids_t,
                card_embedding=self.state_encoder.card,
                token_mlp_simple=self.state_encoder.token_mlp_simple,
            )
            if self.pass_floor_epsilon > 0.0 or self.partner_call_epsilon > 0.0 or self.pick_floor_epsilon > 0.0:
                probs = self._apply_epsilon_mixing(probs, action_mask_t)

        return probs, logits

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        with torch.no_grad():
            # Use the actor's previous hidden state for the critic so value matches recurrent context
            prev_hidden = None
            if player_id is not None:
                prev_hidden = self.actor._hidden_states.get(player_id, None)
            action_probs, _ = self.get_action_probs_with_logits(state, valid_actions, player_id)
            state_t = self.state_encoder.encode_batch([state], device=device)
            value = self.critic(state_t, hidden_in=prev_hidden)

        # Create distribution for consistent log probability calculation
        dist = torch.distributions.Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            action = dist.sample()

        # Get log probability from the distribution for consistency
        log_prob = dist.log_prob(action)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

    # ------------------------------------------------------------------
    # Observation-only step (no action sampled / no transition stored)
    # ------------------------------------------------------------------
    def observe(self, state, player_id=None, valid_actions=None):
        """Propagate an observation through the actor/critic to update
        the recurrent hidden state *without* sampling an action or storing any
        transition.  Useful for non-decision environment ticks such as the end
        of a trick.

        Parameters
        ----------
        state : dict
            Structured observation for the player.
        player_id : int | None
            Identifier to associate a persistent hidden state (1-5 in game).
        valid_actions : set[int] | None
            Optional set of currently valid action IDs – used only for masking
            so the hidden state sees the same inputs as `act()` would.
        """

        if valid_actions is not None:
            action_mask = self.get_action_mask(valid_actions, self.action_size)
        else:
            action_mask = torch.ones(self.action_size, dtype=torch.bool)

        state_t = self.state_encoder.encode_batch([state], device=device)
        # Provide hand_ids for pointer path and pass encoder modules explicitly
        hand_ids_np = state['hand_ids']
        hand_ids_t = torch.as_tensor(hand_ids_np, dtype=torch.long, device=device).view(1, -1)
        action_mask_t = action_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            _ = self.actor(
                state_t,
                action_mask_t,
                player_id,
                hand_ids=hand_ids_t,
                card_embedding=self.state_encoder.card,
                token_mlp_simple=self.state_encoder.token_mlp_simple,
            )
            _ = self.critic(state_t)

    def store_transition(self, state, action, reward, value, log_prob, done, valid_actions, player_id=None, win_label=None, final_return_label=None):
        """Store transition data"""
        action_mask = self.get_action_mask(valid_actions, self.action_size)
        self.events.append({
            'kind': 'action',
            'state': state,
            'mask': action_mask,
            'action': action - 1,  # store 0-indexed
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done,
            'player_id': player_id,
            'win': float(win_label) if win_label is not None else 0.0,
            'final_return': float(final_return_label) if final_return_label is not None else 0.0,
        })

    def store_observation(self, state, valid_actions=None, player_id=None):
        """Store an observation-only frame to be used for recurrent unrolling during training.
        Does not contribute to GAE or PPO loss.
        """
        if valid_actions is not None:
            mask = self.get_action_mask(valid_actions, self.action_size)
        else:
            mask = torch.ones(self.action_size, dtype=torch.bool)
        self.events.append({
            'kind': 'obs',
            'state': state,
            'mask': mask,
            'player_id': player_id,
        })

    def compute_gae(self, next_value=0):
        """Compute GAE per player over action events; write results back into events."""
        # Group action indices by player
        actions_by_player: dict[int | None, list[int]] = {}
        for i, e in enumerate(self.events):
            if e['kind'] == 'action':
                pid = e.get('player_id', None)
                actions_by_player.setdefault(pid, []).append(i)

        all_advantages = []
        all_returns = []
        for pid, idxs in actions_by_player.items():
            if not idxs:
                continue
            rewards = np.array([self.events[i]['reward'] for i in idxs])
            # Bootstrap next value as 0.0 per player sequence
            values = np.array([self.events[i]['value'] for i in idxs] + [0.0])
            dones = np.array([self.events[i]['done'] for i in idxs] + [False])

            advantages = np.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values[:-1]

            for i, adv, ret in zip(idxs, advantages, returns):
                self.events[i]['advantage'] = float(adv)
                self.events[i]['return'] = float(ret)

            all_advantages.append(advantages)
            all_returns.append(returns)

        if not all_advantages:
            return np.array([]), np.array([])
        return np.concatenate(all_advantages), np.concatenate(all_returns)

    # ------------------------------------------------------------------
    # Internal helpers for PPO update
    # ------------------------------------------------------------------
    def _prepare_training_views(self):
        # Keep raw states (dicts) to encode inside the update for gradient flow
        states = [e['state'] for e in self.events]
        masks_t = [
            (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device))
            for e in self.events
        ]
        kinds = [e['kind'] for e in self.events]
        pids = [e.get('player_id', None) for e in self.events]
        return states, masks_t, kinds, pids

    @staticmethod
    def _index_events_by_player(pids: list[int | None]) -> dict[int | None, list[int]]:
        events_by_player: dict[int | None, list[int]] = {}
        for idx, pid in enumerate(pids):
            events_by_player.setdefault(pid, []).append(idx)
        return events_by_player

    def _segments_from_events(self, events_by_player: dict[int | None, list[int]], kinds: list[str]):
        segments: list[tuple[int | None, int, int]] = []
        for pid, ev_idxs in events_by_player.items():
            action_ev_idxs = [i for i in ev_idxs if kinds[i] == 'action']
            if not action_ev_idxs:
                continue
            # Build done flags for this player's actions in order
            dones_pid = [self.events[i]['done'] for i in action_ev_idxs]
            start = ev_idxs[0]
            a_ptr = 0
            for i in ev_idxs:
                if kinds[i] == 'action':
                    if dones_pid[a_ptr]:
                        segments.append((pid, start, i))
                        start = i + 1
                    a_ptr += 1
            if start <= ev_idxs[-1]:
                segments.append((pid, start, ev_idxs[-1]))
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

    def _build_minibatch_tensors(self, batch, states, masks_t, kinds, pids):
        lengths = []
        states_seqs = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []
        win_list_all = []
        final_ret_list_all = []

        for pid, seg_start, seg_end in batch:
            ev_range = list(range(seg_start, seg_end + 1))
            lengths.append(len(ev_range))
            states_seqs.append([states[i] for i in ev_range])
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            win_bt, final_ret_bt = [], []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                    win_lbl = self.events[i].get('win', 0.0) or 0.0
                    final_return_lbl = self.events[i].get('final_return', 0.0) or 0.0
                    win_bt.append(torch.tensor(float(win_lbl), dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(float(final_return_lbl), dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    win_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    final_ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))
            win_list_all.append(torch.stack(win_bt, dim=0))
            final_ret_list_all.append(torch.stack(final_ret_bt, dim=0))

        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        win_bt, _ = self._pad_to_bt(win_list_all, lengths, 0.0)
        final_ret_bt, _ = self._pad_to_bt(final_ret_list_all, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_seqs, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt, win_bt, final_ret_bt

    def _forward_vectorized(self, states_input, masks_bt, lengths_bt):
        # states_input is a list of sequences (new dict API). Encode to (B,T,256), then run shared backbone.
        encoded_bt = self.state_encoder.encode_sequences(states_input, device=device)
        feat_bt = self.backbone.forward_sequence(encoded_bt, lengths_bt)
        actor_feat_bt = self.actor.actor_adapter(feat_bt)
        B, T = masks_bt.size(0), masks_bt.size(1)

        # Build hand_ids_bt (B,T,N) from dict states
        N = 8 # Maximum hand size
        hand_ids_bt = torch.zeros((B, T, N), dtype=torch.long, device=device)
        for b, seq in enumerate(states_input):
            for t, s in enumerate(seq):
                if t >= T:
                    break
                arr = torch.as_tensor(s['hand_ids'], dtype=torch.long, device=device)
                if arr.dim() == 1:
                    arr = arr.view(-1)
                hand_ids_bt[b, t, :min(N, arr.numel())] = arr[:N]

        # Flatten time dimension to reuse single helper, then reshape back
        flat_feat = actor_feat_bt.reshape(B * T, -1)
        flat_hand = hand_ids_bt.reshape(B * T, N)
        flat_mask = masks_bt.view(B * T, -1)

        logits_flat = self.actor._build_logits_from_features(
            actor_features=flat_feat,
            hand_ids=flat_hand,
            card_embedding=self.state_encoder.card,
            token_mlp_simple=self.state_encoder.token_mlp_simple,
            action_mask=flat_mask,
        )
        logits_bt = logits_flat.view(B, T, -1)

        critic_feat_bt = self.critic.critic_adapter(feat_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)
        # Auxiliary preds use critic_adapter on detached backbone features (no grad into backbone)
        aux_feat_bt = self.critic.critic_adapter(feat_bt.detach())
        win_logits_bt = self.critic.win_head(aux_feat_bt).squeeze(-1)
        ret_pred_bt = self.critic.return_head(aux_feat_bt).squeeze(-1)
        return logits_bt, values_bt, win_logits_bt, ret_pred_bt

    @staticmethod
    def _flatten_action_steps(is_action_bt, logits_bt, values_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, win_logits_bt, ret_pred_bt, win_bt, final_ret_bt, masks_bt):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            logits_bt.view(-1, logits_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
            win_logits_bt.view(-1)[flat_mask],
            ret_pred_bt.view(-1)[flat_mask],
            win_bt.view(-1)[flat_mask],
            final_ret_bt.view(-1)[flat_mask],
            masks_bt.view(-1, masks_bt.size(-1))[flat_mask],
        )

    @staticmethod
    def _entropy_from_probs(sub):
        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

    def _head_entropies(self, probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t):
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

    def _per_head_weights(self, actions_flat, pick_idx_t, partner_idx_t, bury_idx_t, template):
        is_pick = torch.isin(actions_flat, pick_idx_t)
        is_partner = torch.isin(actions_flat, partner_idx_t)
        is_bury = torch.isin(actions_flat, bury_idx_t)
        is_play = ~(is_pick | is_partner | is_bury)

        count_pick = is_pick.sum().item()
        count_partner = is_partner.sum().item()
        count_bury = is_bury.sum().item()
        count_play = is_play.sum().item()
        total_count = float(count_pick + count_partner + count_bury + count_play)
        heads_present = int((count_pick > 0) + (count_partner > 0) + (count_bury > 0) + (count_play > 0))

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
        values_flat,
        returns_flat,
        adv_flat,
        pick_idx_t,
        partner_idx_t,
        bury_idx_t,
        play_idx_t,
    ):
        # Build probabilities fresh from logits to avoid in-place softmax conflicts
        probs_all = F.softmax(logits_flat, dim=-1)
        # Partner-call mixture on-the-fly
        if self.partner_call_epsilon > 0.0:
            A = probs_all.size(-1)
            call_mask = torch.zeros(A, dtype=torch.bool, device=probs_all.device)
            call_mask[self.partner_call_subindices] = True
            valid_call = call_mask.view(1, A).expand_as(mask_flat) & mask_flat
            count = valid_call.float().sum(dim=-1, keepdim=True)
            has = count > 0.5
            ucall = torch.where(has, valid_call.float() / count.clamp_min(1.0), torch.zeros_like(probs_all))
            eps = torch.where(has, torch.full_like(count, self.partner_call_epsilon), torch.zeros_like(count))
            probs_all = (1.0 - eps) * probs_all + eps * ucall

        # PICK-floor mixing on-the-fly
        if self.pick_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pick = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pick[self.pick_action_index] = 1.0
            one_hot_pick = one_hot_pick.view(1, A).expand_as(probs_all)
            valid_pick = mask_flat[:, self.pick_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pick,
                (1.0 - self.pick_floor_epsilon) * probs_all + self.pick_floor_epsilon * one_hot_pick,
                probs_all,
            )

        # PASS-floor mixing on-the-fly
        if self.pass_floor_epsilon > 0.0:
            A = probs_all.size(-1)
            one_hot_pass = torch.zeros(A, dtype=probs_all.dtype, device=probs_all.device)
            one_hot_pass[self.pass_action_index] = 1.0
            one_hot_pass = one_hot_pass.view(1, A).expand_as(probs_all)
            valid_pass = mask_flat[:, self.pass_action_index].unsqueeze(-1)
            probs_all = torch.where(
                valid_pass,
                (1.0 - self.pass_floor_epsilon) * probs_all + self.pass_floor_epsilon * one_hot_pass,
                probs_all,
            )

        dist = torch.distributions.Categorical(probs_all.clamp(min=1e-12))
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_all, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
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
        policy_loss = (pg_loss_elements * head_weight).mean()

        returns_target = returns_flat.view(-1)
        values_old = values_flat.detach()
        v_clipped = values_old + torch.clamp(
            values_flat - values_old, -self.value_clip_epsilon, self.value_clip_epsilon
        )
        critic_loss_unclipped = F.mse_loss(values_flat, returns_target)
        critic_loss_clipped = F.mse_loss(v_clipped, returns_target)
        critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)

        actor_loss = policy_loss - entropy_term + self.kl_coef * approx_kl_t
        return actor_loss, critic_loss, approx_kl_t, (pick_entropy, partner_entropy, bury_entropy, play_entropy)

    def update(self, epochs=6, batch_size=256):
        """Update actor and critic networks using PPO with recurrent unrolling.
        Includes performance optimisations and per-update timing logs.
        """
        t_update_start = time.time()
        if len(self.events) == 0:
            return {}

        # Compute advantages and returns
        advantages, returns = self.compute_gae()

        # Store statistics before normalization
        raw_advantages = advantages.copy() if advantages.size else np.array([0.0])
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)) if advantages.size else 0.0,
            'std': float(np.std(returns)) if advantages.size else 0.0,
            'min': float(np.min(returns)) if advantages.size else 0.0,
            'max': float(np.max(returns)) if advantages.size else 0.0,
        }

        # Normalize advantages and write back into events so the loss uses normalized values
        if advantages.size:
            adv_mean = advantages.mean()
            adv_std = advantages.std() + 1e-8
            for e in self.events:
                if e.get('kind') == 'action':
                    e['advantage'] = float((e['advantage'] - adv_mean) / adv_std)

        # Build static views and segments
        t_build_start = time.time()
        states, masks_t, kinds, pids = self._prepare_training_views()
        events_by_player = self._index_events_by_player(pids)
        segments = self._segments_from_events(events_by_player, kinds)

        # Precompute static index tensors once
        pick_idx_tensor_static = torch.tensor(self.action_groups['pick'], device=device)
        partner_idx_tensor_static = torch.tensor(self.action_groups['partner'], device=device)
        bury_idx_tensor_static = torch.tensor(self.action_groups['bury'], device=device)
        play_idx_tensor_static = torch.tensor(self.action_groups['play'], device=device)

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
        pick_adv_sum = 0.0
        pick_adv_count = 0
        pass_adv_sum = 0.0
        pass_adv_count = 0

        for _ in range(epochs):
            if not segments:
                break
            perm = torch.randperm(len(segments))
            for mb_start in range(0, len(segments), batch_size):
                batch_idxs = perm[mb_start:mb_start + batch_size].tolist()
                batch = [segments[i] for i in batch_idxs]
                if len(batch) == 0:
                    continue

                (
                    states_bt,
                    masks_bt,
                    is_action_bt,
                    actions_bt,
                    old_lp_bt,
                    returns_bt,
                    adv_bt,
                    lengths_bt,
                    win_bt,
                    final_ret_bt,
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                logits_bt, values_bt, win_logits_bt, ret_pred_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt, logits_bt, values_bt, actions_bt, old_lp_bt, returns_bt, adv_bt,
                    win_logits_bt, ret_pred_bt, win_bt, final_ret_bt, masks_bt
                )
                if flat is None:
                    continue
                (
                    logits_flat,
                    values_flat,
                    actions_flat,
                    old_lp_flat,
                    returns_flat,
                    adv_flat,
                    win_logits_flat,
                    ret_pred_flat,
                    win_labels_flat,
                    final_ret_labels_flat,
                    mask_flat,
                ) = flat

                # Record PICK/PASS advantages across minibatches
                with torch.no_grad():
                    pick_mask_specific = (actions_flat == self.pick_action_index)
                    if pick_mask_specific.any():
                        pick_adv_sum += adv_flat[pick_mask_specific].sum().item()
                        pick_adv_count += int(pick_mask_specific.sum().item())
                    pass_mask_specific = (actions_flat == self.pass_action_index)
                    if pass_mask_specific.any():
                        pass_adv_sum += adv_flat[pass_mask_specific].sum().item()
                        pass_adv_count += int(pass_mask_specific.sum().item())

                # Losses and metrics
                (
                    actor_loss,
                    critic_loss,
                    approx_kl_t,
                    (pick_entropy, partner_entropy, bury_entropy, play_entropy),
                ) = self._actor_critic_losses(
                    logits_flat,
                    mask_flat,
                    actions_flat,
                    old_lp_flat,
                    values_flat,
                    returns_flat,
                    adv_flat,
                    pick_idx_tensor_static,
                    partner_idx_tensor_static,
                    bury_idx_tensor_static,
                    play_idx_tensor_static,
                )
                last_approx_kl = float(approx_kl_t.item())

                # Entropy accumulation
                ent_pick_sum += pick_entropy.detach().item()
                ent_partner_sum += partner_entropy.detach().item()
                ent_bury_sum += bury_entropy.detach().item()
                ent_play_sum += play_entropy.detach().item()
                ent_batches += 1

                # Auxiliary losses (detached features used upstream)
                bce_loss = F.binary_cross_entropy_with_logits(win_logits_flat, win_labels_flat)
                return_loss = F.smooth_l1_loss(ret_pred_flat, final_ret_labels_flat)

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = actor_loss + self.value_loss_coeff * critic_loss + self.win_loss_coeff * bce_loss + self.return_loss_coeff * return_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.state_encoder.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                step_time += time.time() - t_step
                optimizer_steps += 1

                # Early stop further updates in this epoch if KL exceeds threshold
                if self.target_kl is not None and last_approx_kl > self.target_kl:
                    early_stop_triggered = True
                    break

            if early_stop_triggered:
                break

        transitions = sum(1 for e in self.events if e['kind'] == 'action')

        # Clear storage
        self.reset_storage()

        # Return training statistics
        t_end = time.time()
        timing = {
            'build_s': t_build_end - t_build_start,
            'forward_s': forward_time,
            'backward_s': backward_time,
            'step_s': step_time,
            'total_update_s': t_end - t_update_start,
            'optimizer_steps': optimizer_steps,
        }
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions,
            'approx_kl': last_approx_kl,
            'early_stop': early_stop_triggered,
            'timing': timing,
            'head_entropy': {
                'pick': (ent_pick_sum / ent_batches) if ent_batches > 0 else 0.0,
                'partner': (ent_partner_sum / ent_batches) if ent_batches > 0 else 0.0,
                'bury': (ent_bury_sum / ent_batches) if ent_batches > 0 else 0.0,
                'play': (ent_play_sum / ent_batches) if ent_batches > 0 else 0.0,
            },
            'pick_pass_adv': {
                'pick_mean': (pick_adv_sum / pick_adv_count) if pick_adv_count > 0 else 0.0,
                'pick_count': pick_adv_count,
                'pass_mean': (pass_adv_sum / pass_adv_count) if pass_adv_count > 0 else 0.0,
                'pass_count': pass_adv_count,
            },
        }

    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath, load_optimizers: bool = True):
        """Load model parameters.

        Parameters
        ----------
        filepath : str
            Path to checkpoint file
        load_optimizers : bool
            If True, also loads optimizer states. For population/inference agents,
            pass False to skip optimizer loading entirely.
        """
        checkpoint = torch.load(filepath, map_location=device)

        self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
