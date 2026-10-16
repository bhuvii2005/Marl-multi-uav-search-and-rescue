"""
Render an animated rollout of a trained policy: drones, victims, and the
belief-map heatmap, evolving frame by frame.

Usage (local or Colab):
    python visualize.py --ckpt runs/belief_main/ckpt_final.pt --obs_mode belief \
        --out rollout.gif

    # deterministic (greedy) actions instead of sampled:
    python visualize.py --ckpt runs/belief_main/ckpt_final.pt --greedy --out rollout.mp4

In Colab, after running this, display the result inline with:
    from IPython.display import Image
    Image('rollout.gif')
or for mp4:
    from IPython.display import Video
    Video('rollout.mp4', embed=True)

Requires drone.png, victim.png, victim_found.png (sprite icons) to sit next
to this script — they ship inside sar_marl/ already.
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from envs.belief_sar_env import BeliefSAREnv
from algos.mappo import MAPPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- visual theme (matches the project's navy/amber deck palette) ---------
NAVY = "#151A33"
PANEL_BG = "#1E2761"
ICE = "#CADCFC"
AMBER = "#FFC857"
GRID_LINE = "#2A3466"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to a saved checkpoint (.pt)")
    p.add_argument("--grid_size", type=int, default=30)
    p.add_argument("--n_drones", type=int, default=4)
    p.add_argument("--n_victims", type=int, default=2)
    p.add_argument("--max_cycles", type=int, default=200)
    p.add_argument("--obs_mode", type=str, default="belief", choices=["belief", "coverage"])
    p.add_argument("--victim_moves", type=int, default=1)
    p.add_argument("--victim_speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="use argmax actions instead of sampling")
    p.add_argument("--out", type=str, default="rollout.gif")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


@torch.no_grad()
def act(algo, obs_np, greedy=False):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=algo.device)
    dist = algo.actor(obs)
    if greedy:
        action = dist.probs.argmax(dim=-1)
    else:
        action = dist.sample()
    return action.cpu().numpy()


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

    algo = MAPPO(obs_dim, act_dim, state_dim, args.n_drones, device=args.device)
    algo.load(args.ckpt)
    algo.actor.eval()

    obs_dict, _ = env.reset(seed=args.seed)
    agent_list = env.possible_agents

    frames = []  # each frame: (drone_pos copy, victim_pos copy, found copy, belief copy, t)
    frames.append((env.drone_pos.copy(), env.victim_pos.copy(), env.found.copy(), env.belief.copy(), env.t))

    for _ in range(args.max_cycles):
        if not env.agents:
            break
        obs_arr = np.stack([obs_dict.get(a, np.zeros(obs_dim, dtype=np.float32)) for a in agent_list])
        actions_arr = act(algo, obs_arr, greedy=args.greedy)
        actions = {a: int(actions_arr[i]) for i, a in enumerate(agent_list) if a in obs_dict}
        obs_dict, rewards, terms, truncs, infos = env.step(actions)
        frames.append((env.drone_pos.copy(), env.victim_pos.copy(), env.found.copy(), env.belief.copy(), env.t))

    print(f"Rollout length: {len(frames)} frames, victims found: "
          f"{frames[-1][2].sum()}/{args.n_victims}")

    # ---------------------------------------------------------------- plot
    grid = args.grid_size
    all_beliefs = np.stack([f[3] for f in frames])
    vmax = np.percentile(all_beliefs, 99.5) + 1e-9

    drone_img = mpimg.imread(os.path.join(SCRIPT_DIR, "drone.png"))
    victim_img = mpimg.imread(os.path.join(SCRIPT_DIR, "victim.png"))
    found_img = mpimg.imread(os.path.join(SCRIPT_DIR, "victim_found.png"))

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (ax_swarm, ax_belief) = plt.subplots(1, 2, figsize=(13.5, 6.6))
    fig.patch.set_facecolor(NAVY)

    belief_cmap = "magma"

    for ax, cap in ((ax_swarm, "Swarm Search View"), (ax_belief, "Belief Map — P(victim present)")):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(0, grid)
        ax.set_ylim(0, grid)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_LINE)
        ax.set_title(cap, color=ICE, fontsize=13, fontweight="bold", pad=10)

    im1 = ax_swarm.imshow(frames[0][3].T, origin="lower", cmap=belief_cmap, extent=[0, grid, 0, grid],
                           vmin=0, vmax=vmax, alpha=0.9)
    im2 = ax_belief.imshow(frames[0][3].T, origin="lower", cmap=belief_cmap, extent=[0, grid, 0, grid],
                            vmin=0, vmax=vmax)
    cbar = fig.colorbar(im2, ax=ax_belief, fraction=0.046, pad=0.04)
    cbar.set_label("Belief probability", color=ICE)
    cbar.ax.yaxis.set_tick_params(color=ICE)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=ICE)
    cbar.outline.set_edgecolor(GRID_LINE)

    title = fig.suptitle("", fontsize=15, color="white", fontweight="bold", y=0.99)

    # sprite artists, pooled and reused frame-to-frame
    def make_sprite_boxes(img, n, zoom):
        boxes = []
        for _ in range(n):
            box = AnnotationBbox(OffsetImage(img, zoom=zoom), (0, 0), xybox=(0, 0),
                                  xycoords="data", boxcoords="data", frameon=False, zorder=6, pad=0)
            box.set_visible(False)
            ax_swarm.add_artist(box)
            boxes.append(box)
        return boxes

    drone_zoom = 0.055 * (30.0 / grid)
    marker_zoom = 0.045 * (30.0 / grid)
    drone_boxes = make_sprite_boxes(drone_img, args.n_drones, drone_zoom)
    victim_boxes = make_sprite_boxes(victim_img, args.n_victims, marker_zoom)
    found_boxes = make_sprite_boxes(found_img, args.n_victims, marker_zoom)

    # a static legend built from the same sprites, drawn once
    legend_ax = fig.add_axes([0.085, 0.90, 0.30, 0.001])
    legend_ax.axis("off")
    legend_items = [(drone_img, "Drone"), (victim_img, "Victim"), (found_img, "Found")]
    lx = 0.06
    for img, label in legend_items:
        ib = OffsetImage(img, zoom=0.02)
        ab = AnnotationBbox(ib, (lx, 0.955), xycoords="figure fraction", frameon=False)
        fig.add_artist(ab)
        fig.text(lx + 0.018, 0.955, label, color=ICE, fontsize=9, va="center")
        lx += 0.10

    def update(i):
        drone_pos, victim_pos, found, belief, t = frames[i]
        im1.set_data(belief.T)
        im2.set_data(belief.T)

        for j, box in enumerate(drone_boxes):
            pos = (drone_pos[j][0] + 0.5, drone_pos[j][1] + 0.5)
            box.xy = pos
            box.xybox = pos
            box.set_visible(True)

        undetected_idx = np.where(~found)[0]
        detected_idx = np.where(found)[0]
        for j, box in enumerate(victim_boxes):
            if j < len(undetected_idx):
                vi = undetected_idx[j]
                pos = (victim_pos[vi][0] + 0.5, victim_pos[vi][1] + 0.5)
                box.xy = pos
                box.xybox = pos
                box.set_visible(True)
            else:
                box.set_visible(False)
        for j, box in enumerate(found_boxes):
            if j < len(detected_idx):
                vi = detected_idx[j]
                pos = (victim_pos[vi][0] + 0.5, victim_pos[vi][1] + 0.5)
                box.xy = pos
                box.xybox = pos
                box.set_visible(True)
            else:
                box.set_visible(False)

        title.set_text(f"t = {t}      victims found: {int(found.sum())}/{len(found)}")
        return [im1, im2, title] + drone_boxes + victim_boxes + found_boxes

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)

    save_kwargs = {"savefig_kwargs": {"facecolor": fig.get_facecolor()}}
    if args.out.endswith(".gif"):
        anim.save(args.out, writer=animation.PillowWriter(fps=args.fps), **save_kwargs)
    else:
        anim.save(args.out, writer=animation.FFMpegWriter(fps=args.fps), **save_kwargs)

    print(f"Saved animation to {args.out}")


if __name__ == "__main__":
    main()
