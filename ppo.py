import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network with an LSTM core and separate linear heads for the pick / bury / play
    phases.  The three heads' logits are concatenated back into the full action
    space order so existing masking logic continues to work unchanged.
    """

    def __init__(self, backbone: SharedRecurrentBackbone, action_size, action_groups, activation='swish'):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.backbone = backbone  # registered; owns shared params
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'bury', 'play'

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
        bury_logits = self.bury_head(feat)
        play_logits = self.play_head(feat)

        logits[:, self.action_groups['pick']] = pick_logits
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

        # Bury-phase indices include:   (1) all explicit bury actions,
        #                               (2) calling going alone,
        #                               (3) all partner-calling actions (called ace and JD PARTNER).
        #                               (4) All "UNDER" actions.
        bury_phase_actions = BURY_ACTIONS + ["ALONE", "JD PARTNER"] + CALL_ACTIONS + UNDER_ACTIONS
        bury_indices = sorted({ACTION_IDS[a] - 1 for a in bury_phase_actions})

        play_indices = sorted({ACTION_IDS[a] - 1 for a in PLAY_ACTIONS})

        self.action_groups = {
            'pick': sorted(pick_indices),
            'bury': sorted(bury_indices),
            'play': sorted(play_indices),
        }

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
        self.gamma = 0.9
        self.gae_lambda = 0.95
        # Separate entropy coefficients per head
        self.entropy_coeff_pick = 0.02
        self.entropy_coeff_bury = 0.02
        self.entropy_coeff_play = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.5
        self.clip_epsilon_pick = 0.3
        self.clip_epsilon_bury = 0.25
        self.clip_epsilon_play = 0.2

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
            action_probs = self.actor(state, action_mask, player_id)
            value = self.critic(state)

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

    def store_transition(self, state, action, reward, value, log_prob, done, valid_actions):
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
        })

    def store_observation(self, state, valid_actions=None):
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
        })

    def compute_gae(self, next_value=0):
        """Compute GAE over action events only; write results back into events."""
        # Extract action events
        action_indices = [i for i, e in enumerate(self.events) if e['kind'] == 'action']
        if not action_indices:
            return np.array([]), np.array([])

        rewards = np.array([self.events[i]['reward'] for i in action_indices])
        values = np.array([self.events[i]['value'] for i in action_indices] + [next_value])
        dones = np.array([self.events[i]['done'] for i in action_indices] + [False])

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        # Write back
        for idx, adv, ret in zip(action_indices, advantages, returns):
            self.events[idx]['advantage'] = float(adv)
            self.events[idx]['return'] = float(ret)

        return advantages, returns

    def update(self, next_state=None, epochs=10, batch_size=64):
        """Update actor and critic networks using PPO with recurrent unrolling."""
        if len(self.events) == 0:
            return {}

        # Compute next value for GAE
        next_value = 0
        if next_state is not None:
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                next_value = self.critic(next_state).item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)

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

        # Normalize advantages
        if advantages.size:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        # Build tensor views of event fields
        states = [torch.FloatTensor(e['state']).to(device) for e in self.events]
        masks_t = [ (e['mask'].to(device) if isinstance(e['mask'], torch.Tensor) else torch.as_tensor(e['mask'], dtype=torch.bool, device=device)) for e in self.events]
        kinds = [e['kind'] for e in self.events]
        action_indices = [i for i,k in enumerate(kinds) if k=='action']
        # Map from global event index to its compact action position
        ev_to_act_pos = {ev_idx: pos for pos, ev_idx in enumerate(action_indices)}
        dones = torch.tensor([self.events[i]['done'] for i in action_indices], dtype=torch.bool, device=device)
        actions = torch.tensor([self.events[i]['action'] for i in action_indices], dtype=torch.long, device=device)
        old_log_probs = torch.tensor([self.events[i]['log_prob'] for i in action_indices], dtype=torch.float32, device=device)
        returns_t = torch.tensor([self.events[i]['return'] for i in action_indices], dtype=torch.float32, device=device) if advantages.size else torch.tensor([], dtype=torch.float32, device=device)
        adv_t = torch.tensor([self.events[i]['advantage'] for i in action_indices], dtype=torch.float32, device=device) if advantages.size else torch.tensor([], dtype=torch.float32, device=device)

        # Build contiguous sequences delimited by done=True
        # Build contiguous segments over event indices, ending when an action with done=True occurs
        segments = []  # list of (start_event_idx, end_event_idx)
        start = 0
        action_cursor = 0
        for ev_idx, kind in enumerate(kinds):
            if kind == 'action':
                if dones[action_cursor].item():
                    segments.append((start, ev_idx))
                    start = ev_idx + 1
                action_cursor += 1
        if start < len(self.events):
            segments.append((start, len(self.events) - 1))

        # Training epochs – sample by segments, keep temporal order inside
        for _ in range(epochs):
            perm = torch.randperm(len(segments))
            for seg_idx in perm.tolist():
                seg_start, seg_end = segments[seg_idx]

                hidden_actor = None
                hidden_critic = None

                new_log_probs_list = []
                values_list = []
                probs_for_entropy = []
                seg_global_positions = []

                # Unroll sequence
                for ev_idx in range(seg_start, seg_end + 1):
                    s_t = states[ev_idx].unsqueeze(0)
                    m_t = masks_t[ev_idx].unsqueeze(0)

                    # Always step to update hidden state
                    action_probs_t, hidden_actor = self.actor(s_t, m_t, hidden_in=hidden_actor, return_hidden=True)
                    value_t, hidden_critic = self.critic(s_t, hidden_in=hidden_critic, return_hidden=True)

                    if kinds[ev_idx] == 'action':
                        gpos = ev_to_act_pos[ev_idx]
                        seg_global_positions.append(gpos)
                        dist_t = torch.distributions.Categorical(action_probs_t)
                        new_log_prob_t = dist_t.log_prob(actions[gpos].unsqueeze(0))
                        new_log_probs_list.append(new_log_prob_t.squeeze(0))
                        values_list.append(value_t.squeeze(0).squeeze(-1))
                        probs_for_entropy.append(action_probs_t.squeeze(0))

                # Skip segments with no action steps
                if len(seg_global_positions) == 0:
                    continue
                # Stack per-action-step tensors
                new_log_probs_seq = torch.stack(new_log_probs_list)  # (L,)
                values_seq = torch.stack(values_list)                # (L,)
                probs_seq = torch.stack(probs_for_entropy)           # (L, A)

                # Slice old data for this segment over action steps only
                selector = torch.tensor(seg_global_positions, dtype=torch.long, device=device)
                old_log_probs_seq = old_log_probs.index_select(0, selector)
                returns_seq = returns_t.index_select(0, selector) if adv_t.numel() else torch.tensor([], device=device)
                adv_seq = adv_t.index_select(0, selector) if adv_t.numel() else torch.tensor([], device=device)

                # Entropy by head (averaged over sequence)
                with torch.no_grad():
                    pick_idx = torch.tensor(self.action_groups['pick'], device=probs_seq.device)
                    bury_idx = torch.tensor(self.action_groups['bury'], device=probs_seq.device)
                    play_idx = torch.tensor(self.action_groups['play'], device=probs_seq.device)

                    probs_pick = probs_seq[:, pick_idx]
                    probs_bury = probs_seq[:, bury_idx]
                    probs_play = probs_seq[:, play_idx]

                    def entropy_from_probs(sub):
                        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
                        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

                    pick_entropy = entropy_from_probs(probs_pick)
                    bury_entropy = entropy_from_probs(probs_bury)
                    play_entropy = entropy_from_probs(probs_play)

                entropy_term = (self.entropy_coeff_pick * pick_entropy +
                                self.entropy_coeff_bury * bury_entropy +
                                self.entropy_coeff_play * play_entropy)

                # PPO ratio and clipping per step
                ratios = torch.exp(new_log_probs_seq - old_log_probs_seq)

                pick_idx_tensor = torch.tensor(self.action_groups['pick'], device=ratios.device)
                bury_idx_tensor = torch.tensor(self.action_groups['bury'], device=ratios.device)

                actions_seq = actions.index_select(0, selector)
                pick_mask_seq = (actions_seq.unsqueeze(1) == pick_idx_tensor).any(dim=1)
                bury_mask_seq = (actions_seq.unsqueeze(1) == bury_idx_tensor).any(dim=1)

                eps_seq = torch.full_like(ratios, self.clip_epsilon_play)
                eps_seq[pick_mask_seq] = self.clip_epsilon_pick
                eps_seq[bury_mask_seq] = self.clip_epsilon_bury

                surr1 = ratios * adv_seq
                clipped_ratios = torch.maximum(torch.minimum(ratios, 1 + eps_seq), 1 - eps_seq)
                surr2 = clipped_ratios * adv_seq

                actor_loss = -torch.min(surr1, surr2).mean() - entropy_term
                critic_loss = F.mse_loss(values_seq, returns_seq.view(-1))

                # Joint backward so backbone gets gradients from both losses; step backbone via actor optimizer only
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss = actor_loss + self.value_loss_coeff * critic_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        transitions = len(action_indices)

        # Clear storage
        self.reset_storage()

        # Return training statistics
        return {
            'advantage_stats': advantage_stats,
            'value_target_stats': value_target_stats,
            'num_transitions': transitions
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
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
