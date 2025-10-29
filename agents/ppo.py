import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ============================================

# Actor-Critic networks for PPO

# ============================================

class Actor(nn.Module):
    """
    Maps states to action distributions (for continuous action spaces).
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim)) # learnable log std

    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    """
    Maps states to value estimates V(s).
    """
    def __init__(self, obs_dim, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        self.net = nn.Sequential(*layers)
        self.v_head = nn.Linear(last_size, 1)

    def forward(self, x):
        return self.v_head(self.net(x))
    

# ============================================

# PPO Agent

# ============================================

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-4, gamma=0.99, clip_eps=0.1,
        lam=0.97, K_epochs=20, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs


        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

        # rollout buffers
        self.clear_buffer()

    # -----------------------------------------
    # Rollout / interaction methods
    # -----------------------------------------
    @torch.inference_mode()
    def act(self, state, return_details=False):
        """Select action and optionally return logprob/value for training."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.tanh(action)
        logprob = dist.log_prob(action).sum(-1)
        logprob -= torch.log(1 - action_clipped.pow(2) + 1e-7).sum(-1)
        
        value = self.critic(state_tensor)

        if return_details:
            return action_clipped.detach().cpu().numpy()[0], logprob.detach(), value.detach()
        return action_clipped.detach().cpu().numpy()[0]

    def store_transition(self, state, action, logprob, value, reward, done):
        """Store transition data for PPO update."""
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.logprobs.append(logprob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_buffer(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    # -----------------------------------------
    # PPO advantage computation and update
    # -----------------------------------------
    def compute_advantages(self):
        rewards = self.rewards
        values = [v.item() for v in self.values]
        dones = self.dones

        # append bootstrap value
        values.append(0.0)

        advantages, gae = [], 0.0

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32, device=self.device), \
            torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self):
        # sanity check: all rollout lists should have equal length
        min_len = min(len(self.states), len(self.actions),
                    len(self.logprobs), len(self.values),
                    len(self.rewards), len(self.dones))
        self.states = self.states[:min_len]
        self.actions = self.actions[:min_len]
        self.logprobs = self.logprobs[:min_len]
        self.values = self.values[:min_len]
        self.rewards = self.rewards[:min_len]
        self.dones = self.dones[:min_len]

        if len(self.rewards) == 0:
            return  # skip empty episodes

        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_logprobs = torch.cat(self.logprobs).detach()

        for _ in range(self.K_epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            values_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values_pred, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
            )
            self.optimizer.step()

        self.clear_buffer()
    
    def save(self, filepath):
        """
        Save both actor and critic state dictionaries.
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, map_location=None):
        """
        Load model and optimizer state dictionaries.
        """
        checkpoint = torch.load(filepath, map_location=map_location)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
        