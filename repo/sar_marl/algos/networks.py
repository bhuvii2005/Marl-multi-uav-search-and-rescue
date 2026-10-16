import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Decentralized policy: local obs -> action distribution. Shared across agents."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = _mlp(obs_dim, act_dim, hidden)

    def forward(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

    def act(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, action):
        dist = self.forward(obs)
        return dist.log_prob(action), dist.entropy()


class CentralizedCritic(nn.Module):
    """Centralized value function: global state -> scalar value. One critic, shared."""

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = _mlp(state_dim, 1, hidden)

    def forward(self, state):
        return self.net(state).squeeze(-1)
