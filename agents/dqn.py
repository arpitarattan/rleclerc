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
    def __init__(self, obs_dim, n_actions, dict_path = None):
        if torch.cuda.is_available(): self.device = torch.device("cuda")
        else: self.device = torch.device("cpu")

        self.net = QNet(obs_dim, n_actions).to(self.device)
        self.target = QNet(obs_dim, n_actions).to(self.device)

        if dict_path is not None: 
            print('Using pretrained agent...')
            checkpoint = torch.load(dict_path, weights_only=False, map_location='cpu')
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint  # pure state_dict case
            self.net.load_state_dict(state_dict)
        self.target.load_state_dict(self.net.state_dict())
            
        self.optimizer = optim.Adam(self.net.parameters(), lr= 1e-4)
        self.replay = deque(maxlen= 10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-5
        self.update_steps = 0

    def act(self, obs):
        if random.random() < self.eps:
            return random.randrange(self.net.net[-1].out_features)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(obs_t)
        return int(q.argmax().item())
    
    def store(self, *args):
        self.replay.append(Transition(*args))

    def sample_batch(self):
        batch = random.sample(self.replay, self.batch_size)
        return Transition(*zip(*batch))

    def learn(self):
        if len(self.replay) < self.batch_size: return

        batch = self.sample_batch()
        # Split batch into constituent tensors
        s = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        a = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        q_vals = self.net(s).gather(1, a) 
        with torch.no_grad():
            q_next = self.target(ns).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1- done)
        
        loss = nn.MSELoss()(q_vals, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        self.eps = max(self.eps_min, self.eps - self.eps_decay)
        self.update_steps += 1
        if self.update_steps % 500 == 0:
            self.target.load_state_dict(self.net.state_dict())
