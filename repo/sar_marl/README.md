# Belief-Driven Multi-UAV Search Coordination — Starter Codebase

This implements the system described in your Project Review-I report
("Dynamic Belief-Driven Multi-UAV Search Coordination for Non-Stationary
Victim Localization Using MARL"). It's a working starting point, not a
finished, tuned result — you'll need to run real training and tune
hyperparameters for your actual review.

## On the literature/code search

None of the 20 papers in your reference list (including rows 13 and 17,
the closest belief-based ones — Zhang et al. 2024 and Chi 2025) have
public training code released. That's consistent with your own research-gap
finding in Section 2.2: nobody has published this specific combination, so
there's no existing repo to fork. What I built instead is a clean-room
implementation from scratch, using the **architecture your report already
specifies** (Section 3.2) and a **standard, well-established MAPPO
implementation pattern** (parameter-shared decentralized actor + centralized
critic, PPO-clip, GAE) — the same structural approach used in the reference
MAPPO paper (Yu et al., "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games") and its official `marlbenchmark/on-policy` repo, adapted
here to a lightweight custom implementation instead of that repo's heavier
SMAC/Hanabi-oriented codebase.

## Structure

```
sar_marl/
├── envs/
│   └── belief_sar_env.py   # PettingZoo ParallelEnv: grid, victim motion,
│                            # Bayesian belief map, obs_mode switch
├── algos/
│   ├── networks.py         # Actor (shared) + CentralizedCritic
│   ├── buffer.py           # rollout storage + team-level GAE
│   └── mappo.py            # PPO-clip update loop
├── train.py                 # training entrypoint, CLI flags
├── plot_results.py          # compare run logs (Section 3.4 metrics)
└── requirements.txt
```

## How each report objective maps to code

| Report objective (Section 1.3) | Where it lives |
|---|---|
| 2D grid multi-UAV sim, non-adversarial moving victim | `BeliefSAREnv.step()` — victim motion is a biased random walk toward `shelter_pos` |
| Per-cell Bayesian belief map, noisy detections | `BeliefSAREnv.step()` — log-odds Bayesian update + `_blur()` diffusion for staleness |
| MARL policy conditioned on belief entropy | `_get_obs()` — computes local belief-patch entropy when `obs_mode="belief"` |
| Coverage-only baseline | `obs_mode="coverage"` — same env, agents see a visited/unvisited patch instead |
| Static-target belief-MARL baseline | `victim_moves=False` — belief input stays on, victim freezes |
| Reward: minimize time-to-detect, penalize redundant re-visits | `rewards[a] -= redundant_revisit_penalty` when a drone revisits a near-zero-belief cell |

## Generating rollout animations and running repeat tests

Once a checkpoint exists (e.g. `runs/belief_main/ckpt_final.pt`):

```bash
# animated GIF/MP4 of a rollout: swarm view + belief-map heatmap side by side
python visualize.py --ckpt runs/belief_main/ckpt_final.pt --out rollout.gif

# run the policy for several test episodes to confirm consistent behavior
python evaluate.py --ckpt runs/belief_main/ckpt_final.pt --episodes 4
```

`visualize.py` needs `drone.png`, `victim.png`, and `victim_found.png` (shipped
alongside it) to render the sprite icons — keep them in the same folder.
`--greedy` on either script switches from sampled to deterministic actions,
useful for a clean, repeatable demo run.


```bash
pip install -r requirements.txt
```

## Running the three conditions from Section 3.4

```bash
# Proposed approach
python train.py --run_name belief_main --obs_mode belief --victim_moves 1

# Baseline 1: coverage-only MARL
python train.py --run_name baseline_coverage --obs_mode coverage --victim_moves 1

# Baseline 2: static-target belief-MARL
python train.py --run_name baseline_static --obs_mode belief --victim_moves 0
```

Each run writes `runs/<run_name>/log.csv` (per-update mean reward,
time-to-find-all-victims, redundant re-visits) and periodic checkpoints.

Then compare them:

```bash
python plot_results.py runs/belief_main runs/baseline_coverage runs/baseline_static
```

This produces `comparison.png` with the two plots your Section 3.5 expected
outcomes need (faster localisation, fewer redundant re-visits).

To test robustness to victim speed (Section 3.4, point 3), sweep
`--victim_speed` (e.g. 0.5, 1.0, 2.0) for the belief_main run and re-plot.

## What you still need to do

1. **Real training runs.** The defaults (`--total_timesteps 200_000`) are a
   reasonable starting point on CPU for a 30×30 grid / 4 drones, but you
   should watch `log.csv` and extend training until the reward curve
   plateaus — this could be 500k–2M steps depending on your grid size.
2. **Hyperparameter tuning.** `lr`, `rollout_len`, `entropy_coef` (in
   `MAPPO.__init__`) are the first things worth sweeping if learning
   stalls.
3. **QMIX comparison**, if you want it as a second algorithm per your
   Section 1.3 objective 3 ("MAPPO/QMIX") — this codebase only implements
   MAPPO. QMIX needs a value-decomposition network instead of a
   centralized critic; that's a separate algorithm file (`algos/qmix.py`)
   you'd add following the same `envs/belief_sar_env.py` interface.
4. **Visualisation module** (Section 3.2) — `BeliefSAREnv.render()` gives a
   plain-text grid for debugging; the report calls for a Matplotlib/Pygame
   heatmap overlay for the review demo, which isn't built yet.
5. Re-verify all reward/penalty magnitudes against what you argue for in
   your final report — the ones here (+10 detection, +2 team bonus, -0.02
   redundant-revisit, -0.01/step) are reasonable starting values, not
   values taken from any paper.
