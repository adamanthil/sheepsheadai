import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def swish(x):
    """Swish activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)

class ActorNetwork(nn.Module):
    """Policy network that outputs action probabilities"""

    def __init__(self, state_size, action_size, hidden_size=256, activation='swish'):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, action_size)

        # Layer normalization (applied before final layer)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Set activation function
        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state, action_mask=None):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        logits = self.fc6(x)

        # Apply action mask for invalid actions
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)

        return F.softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    """Value network that estimates state values"""

    def __init__(self, state_size, hidden_size=256, activation='swish'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, 1)

        # Set activation function
        if activation == 'swish':
            self.activation = swish
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        return self.fc6(x)

class PPOAgent:
    def __init__(self, state_size, action_size, lr_actor=3e-4, lr_critic=3e-4, activation='swish'):
        self.state_size = state_size
        self.action_size = action_size

        # Networks
        self.actor = ActorNetwork(state_size, action_size, activation=activation).to(device)
        self.critic = CriticNetwork(state_size, activation=activation).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        self.value_loss_coeff = 0.5
        self.max_grad_norm = 0.5

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

    def act(self, state, valid_actions, deterministic=False):
        """Select action given state and valid actions"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_mask = self.get_action_mask(valid_actions, self.action_size).unsqueeze(0).to(device)

        with torch.no_grad():
            action_probs = self.actor(state, action_mask)
            value = self.critic(state)

        if deterministic:
            action = torch.argmax(action_probs, dim=1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

        log_prob = torch.log(action_probs.squeeze(0)[action.item()] + 1e-8)

        return action.item() + 1, log_prob.item(), value.item()  # Convert back to 1-indexed

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
                entropy = dist.entropy().mean()

                # Calculate ratios and surrogate losses
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss - ensure both tensors have same shape
                critic_loss = F.mse_loss(current_values, batch_returns.view(-1))

                # Total loss
                total_loss = (actor_loss +
                             self.value_loss_coeff * critic_loss -
                             self.entropy_coeff * entropy)

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
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
