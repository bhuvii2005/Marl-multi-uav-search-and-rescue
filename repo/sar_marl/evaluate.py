"""
Run a trained checkpoint for several episodes (no rendering) and report
time-to-find-all-victims and redundant-revisit counts for each, plus a
mean/std summary — a quick way to confirm the trained policy behaves
consistently rather than judging it off a single lucky/unlucky rollout.

Usage:
    python evaluate.py --ckpt runs/belief_main/ckpt_final.pt --episodes 4

    # compare against a baseline checkpoint the same way:
    python evaluate.py --ckpt runs/baseline_coverage/ckpt_final.pt --obs_mode coverage --episodes 4
"""
import argparse

import numpy as np
import torch

from envs.belief_sar_env import BeliefSAREnv
from algos.mappo import MAPPO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=4, help="number of test rollouts to run")
    p.add_argument("--grid_size", type=int, default=30)
    p.add_argument("--n_drones", type=int, default=4)
    p.add_argument("--n_victims", type=int, default=2)
    p.add_argument("--max_cycles", type=int, default=200)
    p.add_argument("--obs_mode", type=str, default="belief", choices=["belief", "coverage"])
    p.add_argument("--victim_moves", type=int, default=1)
    p.add_argument("--victim_speed", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", help="use argmax actions instead of sampling")
    p.add_argument("--seed", type=int, default=0, help="base seed; each episode uses seed+i")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def act(algo, obs_np, greedy=False):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=algo.device)
    dist = algo.actor(obs)
    action = dist.probs.argmax(dim=-1) if greedy else dist.sample()
    return action.cpu().numpy()


def run_episode(env, algo, obs_dim, agent_list, seed, greedy):
    obs_dict, _ = env.reset(seed=seed)
    total_redundant = 0
    while env.agents:
        obs_arr = np.stack([obs_dict.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agent_list])
        actions_arr = act(algo, obs_arr, greedy=greedy)
        actions = {a: int(actions_arr[i]) for i, a in enumerate(agent_list) if a in obs_dict}
        obs_dict, rewards, terms, truncs, infos = env.step(actions)
        if infos:
            total_redundant += next(iter(infos.values())).get("redundant_hits", 0)
    return env.t, int(env.found.sum()), total_redundant


def main():
    args = parse_args()

    env = BeliefSAREnv(
        grid_size=args.grid_size,
        n_drones=args.n_drones,
        n_victims=args.n_victims,
        max_cycles=args.max_cycles,
        obs_mode=args.obs_mode,
        victim_moves=bool(args.victim_moves),
        victim_speed=args.victim_speed,
        seed=args.seed,
    )
    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    act_dim = env.action_space(env.possible_agents[0]).n
    state_dim = env.global_state_size()
    agent_list = env.possible_agents

    algo = MAPPO(obs_dim, act_dim, state_dim, args.n_drones, device=args.device)
    algo.load(args.ckpt)
    algo.actor.eval()

    print(f"Running {args.episodes} test episodes with checkpoint: {args.ckpt}")
    print(f"{'Episode':<10}{'Seed':<8}{'Time-to-find':<15}{'Victims found':<16}{'Redundant hits':<15}")

    times, founds, redundants = [], [], []
    for i in range(args.episodes):
        seed = args.seed + i
        t, found, redundant = run_episode(env, algo, obs_dim, agent_list, seed, args.greedy)
        times.append(t)
        founds.append(found)
        redundants.append(redundant)
        print(f"{i + 1:<10}{seed:<8}{t:<15}{f'{found}/{args.n_victims}':<16}{redundant:<15}")

    print("-" * 64)
    print(f"Mean time-to-find:    {np.mean(times):.1f}  (std {np.std(times):.1f})")
    print(f"Mean victims found:   {np.mean(founds):.2f} / {args.n_victims}")
    print(f"Mean redundant hits:  {np.mean(redundants):.1f}  (std {np.std(redundants):.1f})")
    all_found = all(f == args.n_victims for f in founds)
    print(f"All episodes found every victim: {all_found}")


if __name__ == "__main__":
    main()
