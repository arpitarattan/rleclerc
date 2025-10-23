'''
Deep Q-Network Agent to race
'''

import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

# Define base Q-Net
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

# Agent
class DQNAgent:
    def __init__(self, obs_dim, n_actions):
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")

        self.net = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device
        self.target.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(self.net.parameters(), lr= 1e-3)
        self.replay = deque(maxlen= 1000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-5
        self.update_steps = 0

    def act(self, obs);
        