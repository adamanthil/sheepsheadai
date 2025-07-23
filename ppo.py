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


class MultiHeadRecurrentActorNetwork(nn.Module):
    """Actor network with an LSTM core and separate linear heads for the pick / bury / play
    phases.  The three heads' logits are concatenated back into the full action
    space order so existing masking logic continues to work unchanged.
    """

    def __init__(self, state_size, action_size, action_groups, hidden_size=256,
                 lstm_size=128, activation='swish'):
        super(MultiHeadRecurrentActorNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_groups = action_groups  # dict with keys 'pick', 'bury', 'play'

        # === Encoder ===
        self.enc_fc1 = nn.Linear(state_size, hidden_size)
        self.enc_fc2 = nn.Linear(hidden_size, hidden_size)
        self.enc_fc3 = nn.Linear(hidden_size, hidden_size)
        self.enc_ln1 = nn.LayerNorm(hidden_size)

        # Activation selector
        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # === Recurrent core ===
        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)

        # === Heads ===
        self.pick_head = nn.Linear(lstm_size, len(action_groups['pick']))
        self.bury_head = nn.Linear(lstm_size, len(action_groups['bury']))
        self.play_head = nn.Linear(lstm_size, len(action_groups['play']))

        # Buffer to hold hidden states for each player id (1-5).  Populated on the fly.
        self._hidden_states = {}

        # Init
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

    # ------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------
    def reset_hidden(self):
        """Erase all stored hidden states (call at the start of every new game)."""
        self._hidden_states = {}

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, state, action_mask=None, player_id=None):
        """Parameters
        ----------
        state : Tensor  (batch, state_size)  OR (state_size,) for single sample
        action_mask : Bool Tensor same length as action space or None
        player_id : int | None  – if provided, a persistent hidden state is kept
        for that id; otherwise hidden state is initialised to zeros (used during
        batched advantage computation)
        """

        # Ensure 2-D tensor (batch_first)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        batch_size = state.size(0)

        # ---- Encoder ----
        x = self.activation(self.enc_fc1(state))
        x = self.activation(self.enc_fc2(x))
        x = self.activation(self.enc_fc3(x))
        x = self.enc_ln1(x)  # LayerNorm before recurrent core

        # Add time dimension for LSTM (seq_len=1)
        x = x.unsqueeze(1)  # (batch, 1, hidden_size)

        # Get / init hidden state
        hidden = None
        if player_id is not None and batch_size == 1:
            hidden = self._hidden_states.get(player_id, None)

        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.squeeze(1)  # (batch, lstm_size)

        # Persist hidden state for this player (only if single sample)
        if player_id is not None and batch_size == 1:
            self._hidden_states[player_id] = (new_hidden[0].detach(), new_hidden[1].detach())

        # ---- Heads ----
        logits = torch.full((batch_size, self.action_size), -1e8, device=state.device)

        pick_logits = self.pick_head(lstm_out)
        bury_logits = self.bury_head(lstm_out)
        play_logits = self.play_head(lstm_out)

        logits[:, self.action_groups['pick']] = pick_logits
        logits[:, self.action_groups['bury']] = bury_logits
        logits[:, self.action_groups['play']] = play_logits

        # Apply external action mask (invalid actions → large negative value)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        return F.softmax(logits, dim=-1)


class RecurrentCriticNetwork(nn.Module):
    """Value network that mirrors the encoder + LSTM structure of the actor but
    ends with a single scalar output."""

    def __init__(self, state_size, hidden_size=256, lstm_size=128, activation='swish'):
        super(RecurrentCriticNetwork, self).__init__()
        self.state_size = state_size

        # Activation selector
        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.enc_fc1 = nn.Linear(state_size, hidden_size)
        self.enc_fc2 = nn.Linear(hidden_size, hidden_size)
        self.enc_fc3 = nn.Linear(hidden_size, hidden_size)
        self.enc_ln1 = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(hidden_size, lstm_size, batch_first=True)
        self.value_head = nn.Linear(lstm_size, 1)

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

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.activation(self.enc_fc1(state))
        x = self.activation(self.enc_fc2(x))
        x = self.activation(self.enc_fc3(x))
        x = self.enc_ln1(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.squeeze(1)
        return self.value_head(lstm_out)

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

        # Networks (recurrent + multi-head)
        self.actor = MultiHeadRecurrentActorNetwork(
            state_size,
            action_size,
            self.action_groups,
            activation=activation,
        ).to(device)

        self.critic = RecurrentCriticNetwork(state_size, activation=activation).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.99
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
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

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
        self.states.append(state)
        self.actions.append(action - 1)  # Convert to 0-indexed
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        action_mask = self.get_action_mask(valid_actions, self.action_size)
        self.action_masks.append(action_mask)

    def compute_gae(self, next_value=0):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones + [False])

        advantages = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, next_state=None, epochs=10, batch_size=64):
        """Update actor and critic networks using PPO"""
        if len(self.states) == 0:
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
        raw_advantages = advantages.copy()
        advantage_stats = {
            'mean': float(np.mean(raw_advantages)),
            'std': float(np.std(raw_advantages)),
            'min': float(np.min(raw_advantages)),
            'max': float(np.max(raw_advantages))
        }

        value_target_stats = {
            'mean': float(np.mean(returns)),
            'std': float(np.std(returns)),
            'min': float(np.min(returns)),
            'max': float(np.max(returns))
        }

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        action_masks = torch.stack(self.action_masks).to(device)

        # PPO update
        for _ in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_masks = action_masks[batch_indices]

                # Get current policy and value predictions
                action_probs = self.actor(batch_states, batch_masks)
                current_values = self.critic(batch_states).squeeze(-1)  # Only squeeze last dimension

                # Calculate new log probabilities
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                # --- Entropy broken down by head --------------------
                with torch.no_grad():
                    probs = action_probs  # (batch, action_size)

                    pick_idx = torch.tensor(self.action_groups['pick'], device=probs.device)
                    bury_idx = torch.tensor(self.action_groups['bury'], device=probs.device)
                    play_idx = torch.tensor(self.action_groups['play'], device=probs.device)

                    probs_pick = probs[:, pick_idx]
                    probs_bury = probs[:, bury_idx]
                    probs_play = probs[:, play_idx]

                    def entropy_from_probs(sub):
                        sub_norm = sub / (sub.sum(dim=1, keepdim=True) + 1e-8)
                        return -(sub_norm * torch.log(sub_norm + 1e-8)).sum(dim=1).mean()

                    pick_entropy = entropy_from_probs(probs_pick)
                    bury_entropy = entropy_from_probs(probs_bury)
                    play_entropy = entropy_from_probs(probs_play)

                entropy_term = (self.entropy_coeff_pick * pick_entropy +
                                self.entropy_coeff_bury * bury_entropy +
                                self.entropy_coeff_play * play_entropy)

                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Determine per-sample epsilon based on action type
                # Recreate index tensors on the correct device
                pick_idx_tensor = torch.tensor(self.action_groups['pick'], device=ratios.device)
                bury_idx_tensor = torch.tensor(self.action_groups['bury'], device=ratios.device)

                pick_mask_batch = (batch_actions.unsqueeze(1) == pick_idx_tensor).any(dim=1)
                bury_mask_batch = (batch_actions.unsqueeze(1) == bury_idx_tensor).any(dim=1)

                eps_tensor = torch.full_like(ratios, self.clip_epsilon_play)
                eps_tensor[pick_mask_batch] = self.clip_epsilon_pick
                eps_tensor[bury_mask_batch] = self.clip_epsilon_bury

                surr1 = ratios * batch_advantages
                clipped_ratios = torch.maximum(torch.minimum(ratios, 1 + eps_tensor), 1 - eps_tensor)
                surr2 = clipped_ratios * batch_advantages

                # Actor loss (includes entropy bonus for exploration)
                actor_loss = -torch.min(surr1, surr2).mean() - entropy_term

                # Critic loss - ensure both tensors have same shape
                critic_loss = F.mse_loss(current_values, batch_returns.view(-1))

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

        transitions = len(self.states)

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
