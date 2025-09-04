import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sheepshead import ACTION_IDS, BURY_ACTIONS, CALL_ACTIONS, UNDER_ACTIONS, PLAY_ACTIONS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


class PreNormResidual(nn.Module):
    """Pre-norm residual MLP block: y = x + Linear(LN(x) -> hidden -> act -> dropout -> dim)."""
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1, activation=swish):
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
      - enc_proj: Linear(state_size -> 512)
      - enc_blocks: 3 × PreNormResidual(512)
      - lstm: LSTM(512 -> 256)
      - trunk_blocks: 2 × PreNormResidual(256)
    """

    def __init__(self, state_size: int, activation: str = 'swish'):
        super().__init__()
        self.state_size = state_size

        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Encoder projection and residual blocks (512 width)
        self.enc_proj = nn.Linear(state_size, 512)
        self.enc_block1 = PreNormResidual(512, 512, dropout=0.1, activation=self.activation)
        self.enc_block2 = PreNormResidual(512, 512, dropout=0.1, activation=self.activation)
        self.enc_block3 = PreNormResidual(512, 512, dropout=0.1, activation=self.activation)

        # Recurrent core
        self.lstm = nn.LSTM(512, 256, batch_first=True)

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

        # Encoder
        x = self.activation(self.enc_proj(state))
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)

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
        # Encoder over (B, T, state_size)
        x = self.activation(self.enc_proj(states_bt))
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)

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

    def __init__(self, backbone: SharedRecurrentBackbone, action_size, action_groups, activation='swish'):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.backbone = backbone  # registered; owns shared params
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'partner', 'bury', 'play'

        # Activation selector
        if activation == 'swish':
            self.activation = swish
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
        self.partner_head = nn.Linear(256, len(action_groups['partner']))
        self.bury_head = nn.Linear(256, len(action_groups['bury']))
        self.play_head = nn.Linear(256, len(action_groups['play']))

        # Buffer to hold hidden states for each player id (1-5).  Populated on the fly.
        self._hidden_states = {}

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0.0)

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def reset_hidden(self):
        """Erase all stored hidden states (call at the start of every new game)."""
        self._hidden_states = {}

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, state, action_mask=None, player_id=None, hidden_in=None, return_hidden=False):
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

        # Ensure 2-D tensor (batch_first)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        batch_size = state.size(0)

        # Choose hidden state source (for shared LSTM)
        hidden = hidden_in
        if hidden is None and player_id is not None and batch_size == 1:
            hidden = self._hidden_states.get(player_id, None)

        # Shared backbone → features (256) + new hidden
        feat, new_hidden = self.backbone(state, hidden_in=hidden, return_hidden=True)

        # Actor adapter
        feat = self.actor_adapter(feat)

        # Persist hidden state for this player (only if single sample and not using explicit hidden)
        if hidden_in is None and player_id is not None and batch_size == 1:
            self._hidden_states[player_id] = (new_hidden[0].detach(), new_hidden[1].detach())

        # ---- Heads ----
        logits = torch.full((batch_size, self.action_size), -1e8, device=state.device)

        pick_logits = self.pick_head(feat)
        partner_logits = self.partner_head(feat)
        bury_logits = self.bury_head(feat)
        play_logits = self.play_head(feat)

        logits[:, self.action_groups['pick']] = pick_logits
        logits[:, self.action_groups['partner']] = partner_logits
        logits[:, self.action_groups['bury']] = bury_logits
        logits[:, self.action_groups['play']] = play_logits

        # Apply external action mask (invalid actions → large negative value)
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            logits = logits.masked_fill(~action_mask, -1e8)

        probs = F.softmax(logits, dim=-1)
        if return_hidden:
            return probs, new_hidden
        return probs


class RecurrentCriticNetwork(nn.Module):
    """Critic head using the shared backbone with a critic-specific adapter layer."""

    def __init__(self, backbone: SharedRecurrentBackbone, activation='swish'):
        super().__init__()
        # Store a non-registered reference to the shared backbone to avoid
        # duplicating parameters in the critic and to keep optimizer ownership
        object.__setattr__(self, "_backbone", backbone)

        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.critic_adapter = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU() if activation == 'relu' else nn.Identity()
        )
        self.value_head = nn.Linear(256, 1)

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

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        self.state_size = state_size
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
        self.backbone = SharedRecurrentBackbone(state_size, activation=activation).to(device)
        self.actor = MultiHeadRecurrentActorNetwork(
            self.backbone,
            action_size,
            self.action_groups,
            activation=activation,
        ).to(device)

        self.critic = RecurrentCriticNetwork(self.backbone, activation=activation).to(device)

        # Optimizers (single optimizer owns backbone via actor; critic excludes backbone)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
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
        self.clip_epsilon_partner = 0.15
        self.clip_epsilon_bury = 0.1
        self.clip_epsilon_play = 0.15
        self.value_clip_epsilon = 0.15

        # PPO early stopping target for approximate KL (per update)
        self.target_kl = None
        # KL regularization coefficient (added to actor loss)
        self.kl_coef = 0.2

        # Storage for trajectory data
        self.reset_storage()

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

    def act(self, state, valid_actions, player_id=None, deterministic=False):
        """Select action given state and valid actions"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mask = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)

        with torch.no_grad():
            # Use the actor's previous hidden state for the critic so value matches recurrent context
            prev_hidden = None
            if player_id is not None:
                prev_hidden = self.actor._hidden_states.get(player_id, None)
            action_probs = self.actor(state, action_mask, player_id)
            value = self.critic(state, hidden_in=prev_hidden)

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
        state : numpy array or Tensor
            Current observation for the player.
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

        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mask_t = action_mask.unsqueeze(0).to(device)

        with torch.no_grad():
            _ = self.actor(state_t, action_mask_t, player_id)
            _ = self.critic(state_t)

    def store_transition(self, state, action, reward, value, log_prob, done, valid_actions, player_id=None):
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
        states = [torch.FloatTensor(e['state']).to(device) for e in self.events]
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
        states_list = []
        masks_list = []
        is_action_list = []
        actions_list = []
        old_lp_list = []
        returns_list = []
        adv_list = []

        for pid, seg_start, seg_end in batch:
            ev_range = list(range(seg_start, seg_end + 1))
            lengths.append(len(ev_range))
            states_list.append(torch.stack([states[i] for i in ev_range], dim=0))
            masks_list.append(torch.stack([masks_t[i] for i in ev_range], dim=0))
            is_act = torch.tensor(
                [1 if (kinds[i] == 'action' and pids[i] == pid) else 0 for i in ev_range],
                dtype=torch.bool,
                device=device,
            )
            is_action_list.append(is_act)

            act_bt, olp_bt, ret_bt, adv_bt = [], [], [], []
            for i in ev_range:
                if kinds[i] == 'action' and pids[i] == pid:
                    act_bt.append(torch.tensor(self.events[i]['action'], dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(self.events[i]['log_prob'], dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(self.events[i]['return'], dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(self.events[i]['advantage'], dtype=torch.float32, device=device))
                else:
                    act_bt.append(torch.tensor(-1, dtype=torch.long, device=device))
                    olp_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    ret_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
                    adv_bt.append(torch.tensor(0.0, dtype=torch.float32, device=device))
            actions_list.append(torch.stack(act_bt, dim=0))
            old_lp_list.append(torch.stack(olp_bt, dim=0))
            returns_list.append(torch.stack(ret_bt, dim=0))
            adv_list.append(torch.stack(adv_bt, dim=0))

        states_bt, _ = self._pad_to_bt(states_list, lengths, 0.0)
        masks_bt, _ = self._pad_to_bt(masks_list, lengths, True)
        is_action_bt, _ = self._pad_to_bt(is_action_list, lengths, False)
        actions_bt, _ = self._pad_to_bt(actions_list, lengths, -1)
        old_lp_bt, _ = self._pad_to_bt(old_lp_list, lengths, 0.0)
        returns_bt, _ = self._pad_to_bt(returns_list, lengths, 0.0)
        adv_bt, _ = self._pad_to_bt(adv_list, lengths, 0.0)
        lengths_bt = torch.tensor(lengths, dtype=torch.long, device=device)

        return states_bt, masks_bt, is_action_bt, actions_bt, old_lp_bt, returns_bt, adv_bt, lengths_bt

    def _forward_vectorized(self, states_bt, masks_bt, lengths_bt):
        feat_bt = self.backbone.forward_sequence(states_bt, lengths_bt)
        actor_feat_bt = self.actor.actor_adapter(feat_bt)
        B, T = states_bt.size(0), states_bt.size(1)
        logits_bt = torch.full((B, T, self.action_size), -1e8, device=states_bt.device)
        logits_bt[:, :, self.action_groups['pick']] = self.actor.pick_head(actor_feat_bt)
        logits_bt[:, :, self.action_groups['partner']] = self.actor.partner_head(actor_feat_bt)
        logits_bt[:, :, self.action_groups['bury']] = self.actor.bury_head(actor_feat_bt)
        logits_bt[:, :, self.action_groups['play']] = self.actor.play_head(actor_feat_bt)
        logits_bt = logits_bt.masked_fill(~masks_bt, -1e8)
        probs_bt = F.softmax(logits_bt, dim=-1)
        critic_feat_bt = self.critic.critic_adapter(feat_bt)
        values_bt = self.critic.value_head(critic_feat_bt).squeeze(-1)
        return probs_bt, values_bt

    @staticmethod
    def _flatten_action_steps(is_action_bt, probs_bt, values_bt, actions_bt, old_lp_bt, returns_bt, adv_bt):
        flat_mask = is_action_bt.view(-1)
        if flat_mask.sum() == 0:
            return None
        return (
            probs_bt.view(-1, probs_bt.size(-1))[flat_mask],
            values_bt.view(-1)[flat_mask],
            actions_bt.view(-1)[flat_mask],
            old_lp_bt.view(-1)[flat_mask],
            returns_bt.view(-1)[flat_mask],
            adv_bt.view(-1)[flat_mask],
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
        probs_flat,
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
        dist = torch.distributions.Categorical(probs_flat)
        new_lp_flat = dist.log_prob(actions_flat)
        log_ratio = new_lp_flat - old_lp_flat
        approx_kl_t = (torch.exp(log_ratio) - 1 - log_ratio).mean()

        pick_entropy, partner_entropy, bury_entropy, play_entropy = self._head_entropies(
            probs_flat, pick_idx_t, partner_idx_t, bury_idx_t, play_idx_t
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
            # Choose minibatch size (number of segments)
            mb_size = max(1, batch_size // 8)
            for mb_start in range(0, len(segments), mb_size):
                batch_idxs = perm[mb_start:mb_start + mb_size].tolist()
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
                ) = self._build_minibatch_tensors(batch, states, masks_t, kinds, pids)

                # Vectorized forward
                t_fwd = time.time()
                probs_bt, values_bt = self._forward_vectorized(states_bt, masks_bt, lengths_bt)
                forward_time += time.time() - t_fwd

                flat = self._flatten_action_steps(
                    is_action_bt, probs_bt, values_bt, actions_bt, old_lp_bt, returns_bt, adv_bt
                )
                if flat is None:
                    continue
                probs_flat, values_flat, actions_flat, old_lp_flat, returns_flat, adv_flat = flat

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
                    probs_flat,
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

                # Backward + step per minibatch
                t_bwd = time.time()
                total_loss = actor_loss + self.value_loss_coeff * critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                backward_time += time.time() - t_bwd

                t_step = time.time()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
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
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=device)
        actor_sd = checkpoint['actor_state_dict']
        # Backward-compat: if checkpoint predates partner_head split, map old bury_head -> partner + bury
        needs_split = ('partner_head.weight' not in actor_sd)
        if needs_split and ('bury_head.weight' in actor_sd):
            try:
                old_w = actor_sd['bury_head.weight']  # (old_out, in)
                old_b = actor_sd['bury_head.bias']    # (old_out,)
                old_out, in_features = old_w.shape
                # Reconstruct the old merged list and new lists (old merged included UNDER in partner set)
                partner_actions = ["ALONE", "JD PARTNER"] + CALL_ACTIONS + UNDER_ACTIONS
                old_merged_indices = sorted({ACTION_IDS[a] - 1 for a in (BURY_ACTIONS + partner_actions)})
                new_partner_indices = list(self.action_groups['partner'])
                new_bury_indices = list(self.action_groups['bury'])

                # Sanity: old_out should equal len(old_merged_indices)
                if len(old_merged_indices) != old_out:
                    print("\n\n🚨 Backward-compat split ABORTED: row count mismatch")
                    print(f"   old_bury_head rows: {old_out}  vs expected merged indices: {len(old_merged_indices)}")
                    print("   Proceeding with non-strict load; new partner/bury heads will use random init.\n")
                else:
                    # Build lookup: global_idx -> row in new heads
                    partner_row_for_global = {g_idx: new_partner_indices.index(g_idx) for g_idx in new_partner_indices}
                    bury_row_for_global = {g_idx: new_bury_indices.index(g_idx) for g_idx in new_bury_indices}

                    # Allocate new tensors
                    new_partner_w = torch.zeros((len(new_partner_indices), in_features), dtype=old_w.dtype)
                    new_partner_b = torch.zeros((len(new_partner_indices),), dtype=old_b.dtype)
                    new_bury_w = torch.zeros((len(new_bury_indices), in_features), dtype=old_w.dtype)
                    new_bury_b = torch.zeros((len(new_bury_indices),), dtype=old_b.dtype)

                    # Scatter old rows into new heads based on global action index
                    for old_row, global_idx in enumerate(old_merged_indices):
                        if global_idx in partner_row_for_global:
                            new_row = partner_row_for_global[global_idx]
                            new_partner_w[new_row] = old_w[old_row]
                            new_partner_b[new_row] = old_b[old_row]
                        elif global_idx in bury_row_for_global:
                            new_row = bury_row_for_global[global_idx]
                            new_bury_w[new_row] = old_w[old_row]
                            new_bury_b[new_row] = old_b[old_row]
                        else:
                            print(f"⚠️  Unexpected global action index during split: {global_idx}")

                    # Inject into state dict
                    actor_sd['partner_head.weight'] = new_partner_w
                    actor_sd['partner_head.bias'] = new_partner_b
                    actor_sd['bury_head.weight'] = new_bury_w
                    actor_sd['bury_head.bias'] = new_bury_b
            except (KeyError, RuntimeError, ValueError, IndexError, TypeError) as e:
                print("\n\n🚨 Backward-compat split FAILED with expected error type")
                print(f"   Error: {repr(e)}")
                print("   Proceeding with non-strict load; new partner/bury heads will use random init.\n")

        # Load non-strictly to allow for keys that may not exist in old checkpoints
        self.actor.load_state_dict(actor_sd, strict=False)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'actor_optimizer' in checkpoint:
            try:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            except (RuntimeError, ValueError, KeyError) as e:
                print("\n\n🚨 Actor optimizer state load FAILED")
                print(f"   Error: {repr(e)}")
                print("   Continuing without optimizer state (will re-initialize optimizer moments).\n")
        if 'critic_optimizer' in checkpoint:
            try:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            except (RuntimeError, ValueError, KeyError) as e:
                print("\n\n🚨 Critic optimizer state load FAILED")
                print(f"   Error: {repr(e)}")
                print("   Continuing without optimizer state (will re-initialize optimizer moments).\n")
