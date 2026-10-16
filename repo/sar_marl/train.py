"""
Train MAPPO on the belief-driven multi-UAV SAR environment.

Usage:
    python train.py --run_name belief_main --obs_mode belief --victim_moves 1
    python train.py --run_name baseline_coverage --obs_mode coverage --victim_moves 1
    python train.py --run_name baseline_static_belief --obs_mode belief --victim_moves 0

These three commands reproduce the three conditions in the report's
Section 3.4 evaluation plan: proposed belief-conditioned MARL, the
coverage-only baseline, and the static-target belief-MARL baseline.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from envs.belief_sar_env import BeliefSAREnv
from algos.mappo import MAPPO
from algos.buffer import RolloutBuffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", type=str, default="belief_main")
    p.add_argument("--grid_size", type=int, default=30)
    p.add_argument("--n_drones", type=int, default=4)
    p.add_argument("--n_victims", type=int, default=2)
    p.add_argument("--max_cycles", type=int, default=200)
    p.add_argument("--obs_mode", type=str, default="belief", choices=["belief", "coverage"])
    p.add_argument("--victim_moves", type=int, default=1)  # 0 -> static-target baseline
    p.add_argument("--victim_speed", type=float, default=1.0)
    p.add_argument("--rollout_len", type=int, default=200, help="timesteps per PPO update (>= 1 episode)")
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--save_every", type=int, default=20)
    return p.parse_args()


def make_env(args):
    return BeliefSAREnv(
        grid_size=args.grid_size,
        n_drones=args.n_drones,
        n_victims=args.n_victims,
        max_cycles=args.max_cycles,
        obs_mode=args.obs_mode,
        victim_moves=bool(args.victim_moves),
        victim_speed=args.victim_speed,
        seed=args.seed,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.csv")
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["update", "timesteps", "mean_episode_reward", "mean_time_to_find_all",
                      "mean_redundant_hits", "actor_loss", "value_loss"])

    env = make_env(args)
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n
    state_dim = env.global_state_size()
    n_agents = args.n_drones

    algo = MAPPO(obs_dim, act_dim, state_dim, n_agents, lr=args.lr, device=args.device)
    buf = RolloutBuffer(args.rollout_len, n_agents, obs_dim, state_dim, device=args.device)

    obs_dict, _ = env.reset(seed=args.seed)
    episode_reward = 0.0
    episode_start_t = 0
    finished_episode_lengths = []
    finished_episode_rewards = []
    redundant_hits_ep = 0
    redundant_hits_log = []

    total_steps = 0
    update_idx = 0
    start_time = time.time()

    while total_steps < args.total_timesteps:
        buf.reset()
        for _ in range(args.rollout_len):
            agent_list = env.possible_agents  # fixed order for the shared buffer
            obs_arr = np.stack([obs_dict.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agent_list])
            actions_arr, logprobs_arr = algo.act(obs_arr)
            actions = {a: int(actions_arr[i]) for i, a in enumerate(agent_list) if a in obs_dict}

            if not actions:  # episode already ended mid-buffer; reset
                obs_dict, _ = env.reset()
                finished_episode_lengths.append(env.t - episode_start_t)
                finished_episode_rewards.append(episode_reward)
                redundant_hits_log.append(redundant_hits_ep)
                episode_reward, redundant_hits_ep, episode_start_t = 0.0, 0, env.t
                continue

            next_obs, rewards, terms, truncs, infos = env.step(actions)
            reward_arr = np.array([rewards.get(a, 0.0) for a in agent_list], dtype=np.float32)
            done_arr = np.array(
                [terms.get(a, True) or truncs.get(a, True) for a in agent_list], dtype=np.float32
            )
            state_vec = env.get_global_state() if env.agents else np.zeros(state_dim, dtype=np.float32)
            value = algo.value(state_vec)

            buf.add(obs_arr, actions_arr, logprobs_arr, reward_arr, done_arr, state_vec, value)

            episode_reward += reward_arr.mean()
            if infos:
                redundant_hits_ep += next(iter(infos.values())).get("redundant_hits", 0)
            total_steps += n_agents

            if not env.agents:  # episode ended
                finished_episode_lengths.append(env.t - episode_start_t)
                finished_episode_rewards.append(episode_reward)
                redundant_hits_log.append(redundant_hits_ep)
                obs_dict, _ = env.reset()
                episode_reward, redundant_hits_ep, episode_start_t = 0.0, 0, 0
            else:
                obs_dict = next_obs

        last_state = env.get_global_state() if env.agents else np.zeros(state_dim, dtype=np.float32)
        last_value = algo.value(last_state)
        advantages, returns = buf.compute_gae(last_value, gamma=args.gamma, lam=args.gae_lambda)
        stats = algo.update(buf, advantages, returns)

        update_idx += 1
        mean_r = float(np.mean(finished_episode_rewards[-10:])) if finished_episode_rewards else float("nan")
        mean_len = float(np.mean(finished_episode_lengths[-10:])) if finished_episode_lengths else float("nan")
        mean_redundant = float(np.mean(redundant_hits_log[-10:])) if redundant_hits_log else float("nan")

        logger.writerow([update_idx, total_steps, mean_r, mean_len, mean_redundant,
                          stats["actor_loss"], stats["value_loss"]])
        log_file.flush()

        elapsed = time.time() - start_time
        print(f"[{args.run_name}] update {update_idx:4d} | steps {total_steps:8d} | "
              f"mean_reward {mean_r:7.3f} | mean_time_to_find {mean_len:6.1f} | "
              f"redundant {mean_redundant:5.2f} | elapsed {elapsed:6.1f}s")

        if update_idx % args.save_every == 0:
            algo.save(os.path.join(run_dir, f"ckpt_{update_idx}.pt"))

    algo.save(os.path.join(run_dir, "ckpt_final.pt"))
    log_file.close()
    print(f"Done. Logs at {log_path}")


if __name__ == "__main__":
    main()
