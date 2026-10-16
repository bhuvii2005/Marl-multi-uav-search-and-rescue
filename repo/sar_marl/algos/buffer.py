import numpy as np
import torch


class RolloutBuffer:
    """
    Stores one rollout of length T for N agents (parameter-shared policy,
    centralized critic). All agent transitions from a given timestep share
    the same global state (used only by the critic).
    """

    def __init__(self, T, n_agents, obs_dim, state_dim, device="cpu"):
        self.T, self.n_agents = T, n_agents
        self.obs = np.zeros((T, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, n_agents), dtype=np.int64)
        self.logprobs = np.zeros((T, n_agents), dtype=np.float32)
        self.rewards = np.zeros((T, n_agents), dtype=np.float32)
        self.dones = np.zeros((T, n_agents), dtype=np.float32)
        self.state = np.zeros((T, state_dim), dtype=np.float32)
        self.values = np.zeros((T,), dtype=np.float32)
        self.device = device
        self.ptr = 0

    def add(self, obs, actions, logprobs, rewards, dones, state, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = actions
        self.logprobs[i] = logprobs
        self.rewards[i] = rewards
        self.dones[i] = dones
        self.state[i] = state
        self.values[i] = value
        self.ptr += 1

    def full(self):
        return self.ptr >= self.T

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """
        Team-level GAE: rewards are averaged across agents to form a single
        team reward per timestep (fully cooperative task), matched against
        the centralized value function.
        """
        team_reward = self.rewards.mean(axis=1)          # (T,)
        team_done = self.dones.max(axis=1)                # (T,)
        adv = np.zeros(self.T, dtype=np.float32)
        last_gae = 0.0
        values_ext = np.append(self.values, last_value)
        for t in reversed(range(self.T)):
            mask = 1.0 - team_done[t]
            delta = team_reward[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            last_gae = delta + gamma * lam * mask * last_gae
            adv[t] = last_gae
        returns = adv + self.values
        return adv, returns

    def get_tensors(self, advantages, returns):
        d = self.device
        obs = torch.as_tensor(self.obs, device=d)                       # (T, N, obs_dim)
        actions = torch.as_tensor(self.actions, device=d)                # (T, N)
        logprobs = torch.as_tensor(self.logprobs, device=d)              # (T, N)
        state = torch.as_tensor(self.state, device=d)                    # (T, state_dim)
        adv = torch.as_tensor(advantages, device=d)                      # (T,)
        ret = torch.as_tensor(returns, device=d)                         # (T,)
        return obs, actions, logprobs, state, adv, ret
