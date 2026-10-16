import torch
import torch.nn as nn
import numpy as np

from .networks import Actor, CentralizedCritic


class MAPPO:
    def __init__(
        self,
        obs_dim,
        act_dim,
        state_dim,
        n_agents,
        lr=3e-4,
        clip_eps=0.2,
        ppo_epochs=10,
        minibatch_size=256,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        device="cpu",
    ):
        self.device = device
        self.n_agents = n_agents
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = CentralizedCritic(state_dim).to(device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    @torch.no_grad()
    def act(self, obs_np):
        """obs_np: (n_agents, obs_dim) -> actions, logprobs (numpy arrays)."""
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        action, logprob, _ = self.actor.act(obs)
        return action.cpu().numpy(), logprob.cpu().numpy()

    @torch.no_grad()
    def value(self, state_np):
        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device)
        return self.critic(state.unsqueeze(0)).item()

    def update(self, buffer, advantages, returns):
        obs, actions, old_logprobs, state, adv, ret = buffer.get_tensors(advantages, returns)
        T, N, obs_dim = obs.shape

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # flatten agent dim for the actor (parameter sharing); repeat adv/ret per agent isn't
        # needed since advantage/return are computed at the team (per-timestep) level.
        flat_obs = obs.reshape(T * N, obs_dim)
        flat_actions = actions.reshape(T * N)
        flat_old_logprobs = old_logprobs.reshape(T * N)
        # broadcast the per-timestep advantage to every agent at that timestep
        adv_per_agent = adv.unsqueeze(1).expand(T, N).reshape(T * N)

        idx_actor = np.arange(T * N)
        idx_critic = np.arange(T)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx_actor)
            np.random.shuffle(idx_critic)

            for start in range(0, T * N, self.minibatch_size):
                mb = idx_actor[start:start + self.minibatch_size]
                mb_obs = flat_obs[mb]
                mb_actions = flat_actions[mb]
                mb_old_logprobs = flat_old_logprobs[mb]
                mb_adv = adv_per_agent[mb]

                new_logprobs, entropy = self.actor.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()

                self.opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt.step()

            for start in range(0, T, self.minibatch_size):
                mb = idx_critic[start:start + self.minibatch_size]
                mb_state = state[mb]
                mb_ret = ret[mb]
                values = self.critic(mb_state)
                value_loss = self.value_coef * nn.functional.mse_loss(values, mb_ret)

                self.opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt.step()

        return {"actor_loss": actor_loss.item(), "value_loss": value_loss.item()}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
