import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# ============================================
#  Actor-Critic networks for PPO
# ============================================
class Actor(nn.Module):
    """
    Maps states to action probabilities (for discrete) or means (for continuous).
    We'll assume continuous action space with Tanh squashing.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last_size, act_dim)   # mean of action
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable log std

    def forward(self, x):
        x = self.net(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std

class Critic(nn.Module):
    """
    Maps states to state-value estimates V(s)
    """
    def __init__(self, obs_dim, hidden_sizes=(128,128)):
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
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, 
                 lam=0.95, K_epochs=10, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # storage buffers
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    # select action given observation
    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mu, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.tanh(action)  # Tanh for bounded actions
        logprob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(state_tensor)

        # save to buffer
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)

        return action_clipped.detach().cpu().numpy()[0]

    def store_reward_done(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    # compute advantage estimates using GAE
    def compute_advantages(self):
        rewards = self.rewards + [0]  # append dummy for final bootstrap
        values = [v.item() for v in self.values] + [0]
        dones = self.dones + [0]

        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    # PPO update
    def update(self):
        advantages, returns = self.compute_advantages()
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_logprobs = torch.cat(self.logprobs)

        for _ in range(self.K_epochs):
            mu, std = self.actor(states)
            dist = torch.distributions.Normal(mu, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01*entropy.mean()  # entropy bonus

            values_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values_pred, returns)

            loss = actor_loss + 0.5*critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()

        # clear buffers
        self.states, self.actions, self.logprobs, self.rewards, self.dones, self.values = [], [], [], [], [], []

