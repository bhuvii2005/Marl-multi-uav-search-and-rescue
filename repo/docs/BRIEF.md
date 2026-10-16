# Belief-Driven Multi-UAV Search — Research Brief & Colab Training Plan

Oct 16, 2026 · @bhuvan

## 1. Read this first

Our belief-conditioned MAPPO prototype works and beat both baselines in a single seed, but the novelty claim, one reported metric and several implementation details are not yet safe, and a full multi-seed study fits inside one Colab day once the training loop is vectorised.

This file briefs a stronger research model that will go deeper before the team continues. It describes the project, how the model works and is trained, the nearest papers, the one paper to reason against, a Colab time budget, the supervisor's new requests, and the tasks that need deep reasoning (Section 13).

**Five things to know.**

1. **Status.** Environment, belief update, MAPPO training, three conditions, evaluation and visualisation exist and ran on Colab. One seed per condition: proposed 56.9 steps to find both victims (last 50 updates), static-target 127.0, coverage-only 142.8. About 14 minutes per 250,000-step run (Sections 4–5).
2. **Closest paper.** Su & Qian (2023) already pairs MAPPO with a Bayesian target-probability map and moving targets. Our remaining distinctiveness is the controlled belief-versus-coverage ablation, the time-to-find objective, and the route to a calibrated sensor and hardware (Sections 7–8).
3. **Problems found in our own code.** The "redundant re-visits" figures are not independent (they equal 4 × steps), the critic is fed the next-step state, the observation saturates, and the shelter is fixed so the policy may memorise a corner (Section 12).
4. **Compute.** About 52M environment steps cover the full study. That is roughly 6 to 15 hours of Colab time after vectorising, versus about 48 hours with the current loop. Estimates, to be benchmarked (Section 9).
5. **New requests.** The supervisor wants a 3D simulation, thermal images injected in place of a real camera, and later the lab drone with a Raspberry Pi. The plan is the simulation first (a virtual-camera pipeline in gym-pybullet-drones, with a defined message design between drones and base station), and hardware only afterwards if time allows (Section 11).

**How to read it.** Sections 2–5 are facts about our project and can be trusted. Sections 6–8 are limited to what was opened in one session; each row says whether it was verified. Section 9 and the 3D and hardware parts of Section 11 contain estimates, labelled as such. Section 12 is the bug and risk list. Section 13 is your task list, in priority order. Hardware is optional and comes last: do not plan around it or spend research effort on it until the simulation is done.

**Things not in this file.** The code itself (it is in `sar_marl_code.zip`), the full text of Su & Qian's Sections 5–6, and any verification of Chi (2025).

## 2. The project (from the Review-1 report)

The project trains a team of drones to find victims who move, by giving each drone a shared probability map of where the victim might be, instead of a visited/unvisited coverage map.

**Title.** Dynamic Belief-Driven Multi-UAV Search Coordination for Non-Stationary Victim Localization Using Multi-Agent Reinforcement Learning.

**Team.** Bhuvan Chandran S (23BRS1260) and Rithvik Nimal R (23BRS1377), B.Tech CSE (AI and Robotics), VIT Chennai, academic year 2026–2027. Guide: Dr. V.R. BalaSaraswathi.

**Problem.** Most multi-UAV search-and-rescue work treats search as static coverage: sweep the area, mark cells "searched", never revisit. Real victims drift with floods or currents, walk toward shelter, or wander. A cell marked "searched" can be re-entered, so the swarm's information goes stale as the mission runs.

**Core idea.** Keep a per-cell probability of victim presence (the *belief map*), update it from noisy detections, let it decay and diffuse over time, and make each drone's decision depend on the belief and its *entropy* (uncertainty) rather than on coverage history.

**Review-1 objectives.**

1. A 2D grid multi-UAV simulation with a non-adversarial, stochastically moving victim.
2. A per-cell Bayesian belief map updated every step from noisy UAV detections.
3. A MARL policy (MAPPO primary, QMIX comparison) that conditions on belief entropy.
4. Evaluation against a coverage-only MARL baseline and a static-target belief-MARL baseline, on time-to-find-all-victims and redundant re-visits.

**Claimed contribution (stated honestly in the report).** No new RL algorithm. The novelty is the state and reward design, a belief-conditioned MARL formulation, on an established backbone (MAPPO). The report identifies the gap as the combination of three things no reviewed paper had together: a shared belief distribution, a realistically moving (not evading) victim, and a policy whose state and reward use belief uncertainty.

**Scope at Review-1.**

| In scope | Out of scope at Review-1 |
| --- | --- |
| 2D grid, 3–5 UAVs | 3D flight dynamics and physics |
| 1–3 victims on random-walk motion | Adversarial or evading victims |
| Bayesian belief map updated every step | New RL algorithm from scratch |
| Off-the-shelf MAPPO / QMIX | Real hardware |
| Comparison with 2 baselines | Communication-loss and bandwidth modelling |

**What has changed since Review-1.** The supervisor has now asked for three things the report had ruled out: a proper 3D simulation, a thermal-camera-based detector (with images injected rather than a real camera, because thermal cameras are costly), and a hardware demonstration. The team will use the lab drone with a Raspberry Pi, after the simulation is finished. Section 11 covers these.

**Planned timeline in the report.** Eight weeks: environment and victim model (weeks 1–2), belief update (3), belief in the agent state and MAPPO pipeline (4), small-grid training and reward tuning (5), scale to up to 50×50 (6), baselines and comparison (7), analysis and demo (8).

## 3. How the system works (as implemented today)

The implementation is a custom PettingZoo ParallelEnv (`BeliefSAREnv`) where 3–4 drones move on a 30×30 grid, noisy detections update a shared belief map, and each drone sees only a local 5×5 slice of that map plus an uncertainty number. All numbers below are the defaults in our code.

### 3.1 World and sensing

| Item | Value | Meaning |
| --- | --- | --- |
| Grid | 30×30 cells, one drone per cell step | Discrete 2D arena |
| Drones | 4 (Colab runs); report allows 3–5 | Share one policy |
| Victims | 2 | Both must be found to end the episode |
| Actions | 5: stay, up, down, right, left | One cell per step |
| Episode cap | 200 steps | Hitting it counts as failure |
| Sensing window | radius 2 (5×5 cells) around the drone | Square footprint, not a circle |
| Detection probability | 0.90 per victim inside the window | 10% miss rate |
| False-positive probability | 0.02 per drone-step | Raises belief in the window with no victim |
| Communication radius | 8 cells | Teammate positions beyond this are zeroed in the observation |

### 3.2 Victim motion

Each unfound victim moves with probability 0.6 per step. The move is 0.6 × (unit vector toward a fixed shelter at the far corner, cell (29,29)) + 0.4 × Gaussian noise, scaled by the victim speed (1.0). Positions are continuous and clipped to the grid. A victim stops moving once found. This is a biased random walk, non-adversarial, as the report required.

### 3.3 Belief map

The belief is a 30×30 array of per-cell probabilities, initialised uniform (1/900 each) and renormalised to sum to 1 after every step. Each step does three things:

1. **Evidence.** For each drone, every cell in its 5×5 window gets +4.0 log-odds if that drone detected a victim, +1.0 if a false positive fired (probability 0.02), otherwise −0.5.
2. **Conversion.** Belief is turned into log-odds, the evidence is added, and the result is turned back into probabilities (clipped to ±20 log-odds).
3. **Staleness.** The map is blended 98% with itself and 2% with its 3×3 box-blurred version, then renormalised. This is the "decay/diffusion" that lets a cleared area regain probability.

The report's "Bayesian filter" is therefore a simplified, window-level log-odds update with fixed evidence constants, not a likelihood derived from the 0.9/0.02 sensor model. Section 12 lists what to fix.

### 3.4 What each drone observes

Each drone gets a 34-number vector (4 drones, patch radius 2), clipped to \[−1, 1\]:

- Its own position divided by the grid size (2 numbers).
- One uncertainty number: in belief mode, the entropy of its local 5×5 patch (normalised); in coverage mode, the mean coverage of the patch instead.
- The 5×5 patch itself (25 numbers): belief × 900 in belief mode; a recency-decayed "visited" map (decay 0.98 per step) in coverage mode.
- Relative positions of each teammate within 8 cells (2 per teammate, zero if out of range).

The three experimental conditions differ only here and in victim motion:

| Condition | Observation | Victim |
| --- | --- | --- |
| Proposed (belief) | Belief patch + entropy | Moves toward shelter |
| Coverage-only baseline | Visit-recency patch + coverage mean | Moves toward shelter |
| Static-target baseline (as coded) | Belief patch + entropy | Does not move at all |

### 3.5 Policy and critic

This is MAPPO with parameter sharing and a centralised critic (CTDE).

- **Actor (shared by all drones).** MLP 34 → 128 → 128 → 5, tanh activations, outputs a categorical distribution over the 5 moves. At test time it sees only the local observation.
- **Critic (training only).** MLP 908 → 256 → 256 → 1. Its input is the global state: all drone positions (8 numbers) plus the full 900-cell belief map. It outputs one team value.

### 3.6 Reward

Per step, per drone: −0.01. When a drone detects a victim: +10 for that drone and +2 for every drone. A drone standing on a cell whose belief is below 0.01 gets −0.02 (the "redundant revisit" penalty). The learning signal uses the **team mean** reward, so one detection is worth (10 + 2×4) / 4 = 4.5 team reward.

Section 12 explains why the −0.02 penalty and the "redundant re-visits" metric are currently broken.

## 4. How the model is trained today

Training is a plain single-environment MAPPO loop: collect 200 steps, compute team-level advantages, run 10 PPO epochs, repeat 1,250 times. One run of 250,000 environment steps took about 14 minutes on a Colab GPU, measured from our own logs.

### 4.1 The loop, step by step

1. Reset one environment. Each update collects exactly 200 environment steps (rollout length = episode cap). Finished episodes auto-reset inside the rollout.
2. At each step, run the shared actor on all drones' observations, sample 5-way actions, step the env, and record: observations, actions, log-probs, per-drone rewards, done flags, the global state, and the critic's value.
3. Form the team reward (mean over drones) and team done (any drone done). Compute GAE (γ = 0.99, λ = 0.95) against the centralised critic.
4. Normalise advantages to zero mean and unit variance across the rollout, and broadcast the same advantage to every drone at that step.
5. Run 10 PPO epochs. Actor: shuffled minibatches of 256 over the 800 drone-samples, clipped surrogate (ε = 0.2) minus 0.01 × entropy. Critic: MSE against the GAE return, weight 0.5. Gradient clip 0.5 on each.
6. Log, checkpoint every 20 updates, repeat.

### 4.2 Hyperparameters used

| Parameter | Value |
| --- | --- |
| Optimiser | Adam, one optimiser over actor and critic, lr 3e-4, no decay |
| Discount γ / GAE λ | 0.99 / 0.95 |
| PPO clip ε / epochs | 0.2 / 10 |
| Minibatch | 256 |
| Entropy coefficient | 0.01, constant |
| Value coefficient / grad clip | 0.5 / 0.5 |
| Rollout | 200 env steps, 1 environment, no parallel workers |
| Samples per update | 800 drone-samples (200 steps × 4 drones) |
| Updates / total | 1,250 updates = 1,000,000 drone-steps = 250,000 env steps |
| Seeds | one (seed 0) per condition |
| Normalisation | none on observations or values |

### 4.3 Measured Colab cost

| Run | Updates | Wall-clock (Colab GPU) |
| --- | --- | --- |
| belief\_main | 1,250 | 835 s |
| baseline\_coverage | 1,250 | 832 s |
| baseline\_static | 1,250 | 806 s |

That is about 300 environment steps per second. The GPU is almost idle. The bottleneck is Python stepping of one environment plus one tiny GPU call per step, so running on CPU may be about as fast.

### 4.4 What exists in the codebase

- `envs/belief_sar_env.py`: environment, victim model, belief update, three observation conditions.
- `algos/networks.py`, `buffer.py`, `mappo.py`: actor, centralised critic, rollout buffer with team GAE, PPO update.
- `train.py`: CLI training with CSV logs and checkpoints. Flags select the condition (`--obs_mode`, `--victim_moves`) and victim speed.
- `evaluate.py`: runs a checkpoint for N test episodes and prints time-to-find per episode.
- `visualize.py`: two-panel animation (swarm view with drone and victim icons, belief heatmap with colourbar).
- `plot_results.py`: plots the logged curves.
- Not built: QMIX (named in the report), 3D simulation, thermal detector, hardware interface.

### 4.5 What was run

Three conditions, one seed each, 1,250 updates each: proposed (belief, moving victim), coverage-only (moving victim), and static-target (belief, frozen victim). Results are in the next section.

## 5. Results so far

The belief-conditioned policy finds both victims in about 57 steps on average at the end of training, against 127 for the static-target variant and 143 for coverage-only, but this is one seed per condition and several confounds (Section 12) mean it is a promising signal, not yet a result.

All numbers come from the logs printed in our Colab notebooks. "Steps to find" is the episode length, capped at 200, so a failed episode counts as 200.

| Condition | Mean steps to find both victims (last 50 updates) | Last logged value | Mean reward (last 50 updates) |
| --- | --- | --- | --- |
| Proposed (belief, moving victim) | 56.9 | 31.3 | 6.60 |
| Static-target (belief, frozen victim) | 127.0 | 137.8 | 2.86 |
| Coverage-only (moving victim) | 142.8 | 134.8 | 0.30 |

&#91;embedded content: From the Colab training logs (project\_1 notebook, training cells) · 25-update moving average of the logged mean episode length, sampled every 20 updates · one seed per condition\]

Coverage-only dips to about 73 near update 40, then climbs above 130 and stays there. The proposed curve bumps to about 112 near updates 200–240, then sits between about 44 and 68 from update 280 on.

**Single demonstration rollouts.** Proposed found 2/2 victims at step 53 (the polished rollout in the second notebook found 2/2 in 54 frames). Static-target found 2/2 at step 182. Coverage-only found 0/2 in 200 steps, with the drones bunched in a few strips. One rollout each says little statistically.

**What the curves show.** The proposed run falls from 200 to about 60 by update 60, bumps back to about 112 near updates 200–240, then stays between about 44 and 68 from update 280 on. The coverage-only run dips early to about 73, then climbs back to 136–193 and does not recover, which looks like a collapsed or stuck policy rather than slow learning. The static-target run stays at about 130–190 until update 640, then drops to about 89–125 and ends at 114 after a late rise.

**Read these results with care.**

- One seed per condition, so no variance estimate. The two uploaded notebooks contain identical training logs, so they are one run carried over, not two replications.
- The logged value is the mean of the last 10 finished episodes. It swings a lot (31.3 at the final update against 56.9 averaged over the last 50), so "peak" is not yet established.
- The "redundant re-visits" figures shown in the earlier review slides (226, 507, 571) are not a valid second metric. They are almost exactly 4 × steps-to-find (4 drones × episode length) because the penalty fires on nearly every drone-step. Details in Section 12. Do not present them as independent evidence.
- The static-target condition as coded freezes the victim in the environment. The report defines it as a policy that *assumes* a frozen victim and is tested on a moving one. These are different experiments.
- Victims drift toward a fixed corner shelter and the observation includes absolute position, so a policy may be exploiting geography ("go to the corner") instead of using belief. Both found victims in the proposed rollout were in that corner.

Section 10 lists the figures to produce for the final study.

## 6. Nearest papers and repos

The closest work is Su & Qian (2023), which already combines MAPPO, a Bayesian target-probability map with information decay, and randomly moving targets. That is more overlap than the Review-1 table admitted (it listed this paper as having no belief state).

Below, "verified" means I opened the abstract or paper page in this session. Rows from the Review-1 list that I did not re-open are in the note under the table.

| Paper / repo | Targets | Belief or map used | Algorithm | Public code | Relevance |
| --- | --- | --- | --- | --- | --- |
| [Su & Qian, Appl. Sci. 13(21):11905, 2023](https://www.mdpi.com/2076-3417/13/21/11905) | Wander randomly, 2D | Per-UAV target-probability map (start 0.5), log-odds Bayes update from P\_D and P\_F, neighbour fusion with a decay factor, plus a last-observed-time map and a UAV-location map | MAPPO, CTDE | None found | **Closest. See Section 8** |
| [Liao et al., ETS–MAPPO, Sensors 24(21):6859, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548590/) | Escape; static obstacles | Target existence probability map | MAPPO + STEE network (+ GCLA variant) | None found | Moving-target search with a probability map; reports re-search counts |
| [Choi & Choi, BSR-MAPPO, 2026](https://github.com/choiaidrone/belief-aware-graph-token-mappo) | Moving, multiple classes | Dempster–Shafer spatial map with temporal decay, IMM-Kalman target tracks | MAPPO, CTDE, graph tokens | **Yes, MIT** | Belief-aware MAPPO with ablation and seed scripts; heavier than ours |
| [quialga, Multi-UAV-ISR (repo)](https://github.com/quialga/Multi-UAV-ISR) | Evading | Log-odds belief map | PPO, CTDE, GNN | **Yes, Apache-2.0** | Same env API (PettingZoo); good training reference |
| [AM-MAPPO, Drones 8(8):378, 2024](https://www.mdpi.com/2504-446X/8/8/378) | Moving, **3D** | Uncertainty of the search area is the objective | Action-masked MAPPO, FOV state encoding | None found | Best reference for the 3D simulation request |
| [Hong et al., MPRS-MAPPO, Drones 9(11):770, 2025](https://doi.org/10.3390/drones9110770) | Dynamic | Probability map feeds three potential fields for reward shaping | MAPPO with warm-up | None found | Addresses sparse reward and slow convergence, our "not fully trained" problem |
| [Searching and tracking an unknown number of targets, Sensors 21(4):1076, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7915622/) | Moving, unknown count | Four observation maps with map merging | Learning-based (DRL) | None found | Fairness and exploration metrics |
| [Zhang et al., CNN-SEMU, Drones 2024](https://doaj.org/article/ed599f24f5d14e39a3454c2dc6477dfb) | **Static** | Belief maps used for communication and fusion | CNN + semantic episodic memory | None found | Belief maps, limited comms; static targets |
| [Yehoshua et al., arXiv 2103.09520, 2021](https://arxiv.org/abs/2103.09520) | **Static** | None (raw observations) | Fully decentralised policy gradient | None found | Realistic drone simulator reference |
| [Li et al., Profit-driven moving-target search, Sensors 19(7):1545, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480202) | Moving | Local and shared information | Non-RL heuristic (observation profit) | None found | A heuristic baseline to beat |
| [Multi-agent cooperative target search, Sensors 14(6):9408, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4118349/) | Mobile | Bayesian probability map with neighbour fusion | Non-RL distributed control | None found | Theory behind the belief update and fusion |

Two more recent MAPPO variants appeared in search: [AS-MAPPO (attention plus shared observations for dynamic targets, 2025)](https://link.springer.com/article/10.1007/s44163-025-00411-9) and [a center-sub-critics method for searching and tracking a USV swarm with a confidence probability map (Drones 10:123)](https://mdpi-res.com/d_attachment/drones/drones-10-00123/article_deploy/drones-10-00123.pdf). I only saw abstracts and snippets for these.

**Not re-opened in this session** (taken from the Review-1 table): Chi (2025, belief entropy with VDN), Zhang et al. (2024, double-critic DDPG with belief maps), Wei et al. (2024, MAPPO with optimal control), Xu et al. (2025, CA-MAPPO), and the rest of the 20. Section 13 asks the research model to verify the ones that matter, especially Chi 2025, because it is the only reviewed paper that drives decisions with belief entropy.

## 7. How our work differs from the nearby papers

What is still distinctive is the controlled comparison (belief and entropy versus coverage-only, same task) and the time-to-find-all objective, not the existence of a belief map; every nearby paper above also uses some probability map, so the Review-1 claim that no work combines belief, moving targets and MARL no longer holds as written.

| Dimension | Ours | Su & Qian 2023 | ETS–MAPPO 2024 | BSR-MAPPO 2026 |
| --- | --- | --- | --- | --- |
| Belief map | One shared 30×30 map, ad hoc log-odds constants | Per-UAV maps, exact log-likelihood-ratio update, decay factor on fusion | Target existence probability map | Dempster–Shafer map with temporal decay |
| Target motion | Biased walk toward a shelter | Random wander | Escaping | Class-specific models tracked by IMM Kalman filters |
| Uncertainty used by the policy | Local belief entropy as an input | Not seen in the sections read | Not seen | Covariance and track-age tokens, uncertainty–belief priority map |
| Objective | Time to find all victims, episode ends on detection | Average observation rate and exploration rate (continuous) | Searches and re-searches, coverage | Search plus persistent tracking |
| Communication model | Teammate positions limited to 8 cells, but the belief itself is globally shared | Neighbour fusion within a communication range | Not seen | Not seen |
| Algorithm | MAPPO, MLP, parameter sharing | MAPPO | MAPPO + STEE network | MAPPO + graph tokens |
| Dimensionality | 2D grid | 2D | 2D with obstacles | 2D |
| Ablation of the belief signal | Yes: coverage-only and frozen-victim conditions | Compared with other baseline algorithms | Compared with five MARL algorithms | Ablations on tokens and fusion |
| Public code | Ours only | None found | None found | Yes |

**Where we are genuinely different.**

1. A deliberate ablation that isolates the belief-plus-entropy state from coverage information, on a task with a moving victim. Su & Qian feed both a probability map and a search-status map together, so they cannot say which one helped.
2. A clean, single-number task objective (time to find all victims) that is easy to explain to a panel and compare against heuristics.
3. Realistic path to deployment: a sensor model that can be calibrated from a real thermal detector, a 3D simulation, and a hardware-in-the-loop demo. None of the papers above show this chain.

**Where we are weaker, and a panelist can say so.**

- Our belief update is not derived from the sensor model. Su & Qian use the exact log-likelihood-ratio increments, ln(P\_F/P\_D) for a detection and ln((1−P\_F)/(1−P\_D)) for a miss. Ours are fixed constants (+4, +1, −0.5). Their Q is the negative of the log-odds of presence, so in presence log-odds P\_D = 0.9 and P\_F = 0.02 give about +3.8 for a detection and −2.3 for a miss. Ours give +4 and −0.5.
- A shared global belief means perfect, instant map fusion between drones, while teammate positions are range-limited. That is an inconsistent communication assumption.
- Entropy is our headline feature but is computed on a local 5×5 patch after clipping (Section 12), so its effect has not been shown in isolation. An entropy-removed ablation is missing.
- No heuristic baseline (greedy-to-highest-belief, lawnmower sweep) yet.

**Suggested contribution statement.** "We study when belief-and-uncertainty-conditioned MARL helps over coverage-conditioned MARL for non-stationary victim search, with a calibrated sensor model and a path to hardware." Avoid "first" or "no prior work" wording until the research model has checked Chi (2025) and the other unverified rows.

## 8. The one paper to reason against: Su & Qian (2023)

We should argue against Su & Qian, "Multi-UAV Cooperative Searching and Tracking for Moving Targets Based on Multi-Agent Reinforcement Learning" ([Applied Sciences 13(21):11905](https://www.mdpi.com/2076-3417/13/21/11905), open access), because it shares our task, our algorithm family and our core data structure, so every panel question about novelty reduces to how we differ from it.

**Why it is the closest.**

- Same problem family: cooperative search for moving targets with limited sensing range and limited communication, in a discretised 2D grid.
- Same algorithm family: MAPPO with centralised training and decentralised execution.
- Same sensor model: a per-cell detection probability P\_D and false-alarm probability P\_F.
- Same core idea: a per-cell target probability map, started at 0.5, updated by Bayes' rule and decayed over time.
- Same moving-target assumption: targets wander randomly (ours drift toward a shelter).

**Point-by-point comparison.**

| Aspect | Su & Qian | Ours | What it means for us |
| --- | --- | --- | --- |
| Belief update | Exact Bayes, rewritten as a linear log-likelihood-ratio update using P\_D and P\_F | Fixed evidence constants, renormalised to sum to 1 | Adopt their update; it is derived, citable, and fixes our weakest technical point |
| Where the belief lives | Each UAV keeps its own map, fuses with neighbours in range, decay factor η on fusion | One global map seen by all drones | Per-drone beliefs with range-limited fusion is more realistic and matches a real swarm |
| Other state | Last-observed-time map and UAV-location map beside the probability map | Probability patch + entropy, or visit-recency patch (baseline) | Their policy sees belief and coverage-style information together. Our coverage-only baseline is close to their time map alone |
| Objective | Average observation rate of targets and exploration rate of the area | Time to find all victims | Different goal: theirs is persistent tracking, ours is first detection. Not directly comparable numerically |
| Action set | 8 neighbours or stay (fixed-wing, 2D) | 4 neighbours or stay | Minor |
| Uncertainty feature | Not seen in the parts read | Local belief entropy | Our possible edge. Needs an entropy on/off ablation to claim it |
| Reported hyperparameter | Learning rate 5×10⁻⁴ (from a search snippet; not confirmed in the paper text) | 3×10⁻⁴ | Check Section 6 of the paper |

**What to borrow.**

1. The log-likelihood-ratio belief update, with P\_D and P\_F taken from our real thermal-detector confusion matrix.
2. The decay factor idea for fusion, in addition to our spatial diffusion.
3. Per-drone belief maps with neighbour fusion, replacing the single global map.
4. Their two metrics as extra columns: observation rate and exploration rate, alongside time-to-find-all.
5. Their baseline set as a template for ours.

**How to answer "this already exists".** Agree that belief maps exist. State that (a) the contribution is the controlled belief-versus-coverage study with an entropy ablation on a non-stationary victim; (b) the objective is time-to-find rather than persistent observation; (c) the sensor model is calibrated from a real thermal detector and the policy is deployed through a 3D simulation to a drone.

**Limits of this comparison.** I read Sections 1–4 of the paper (system model, probability map, search-status map, location map). Sections 5–6 (observation encoding, reward, network sizes, training budget, baselines, results) were not read, because the page's mathematics made the text too long to pull in full. Section 13 asks the research model to read them.

## 9. Colab training budget: under 24 hours, to peak performance

The full study (3 conditions × 3 seeds × 5M environment steps, plus a pilot and a robustness run, about 52M steps) fits in roughly 6 to 15 hours of Colab time once the training loop is vectorised, but would take about 48 hours with the current single-environment loop. Vectorising is therefore the first task.

### 9.1 What we measured

Our current loop ran 250,000 environment steps in 805–835 s, about 300 steps per second (Section 4.3). The bottleneck is one Python environment plus one small GPU call per step. The two numbers below for faster loops are **estimates to be confirmed by a 10-minute benchmark**, not measurements.

| Throughput (env steps/s) | Basis | Pilot 2M | Full study \~52M | Full study on 2 parallel processes |
| --- | --- | --- | --- | --- |
| 300 | Measured, current code | 1.9 h | 48 h | 24 h |
| 1,000 | Estimate, 8 envs, batched inference | 0.6 h | 14.4 h | 7.2 h |
| 2,500 | Estimate, 16 envs, vectorised belief update | 0.2 h | 5.8 h | 2.9 h |

The "52M steps" is: pilot 2M, main study 45M (3 conditions × 3 seeds × 5M), one randomised-training run 5M. Evaluation adds under 30 minutes. Colab usually offers 2 CPU cores, so two processes in parallel is realistic; the GPU can stay idle for MLP policies, and a CPU runtime may do as well. The 2-parallel column assumes that.

**Fallback if throughput stays near 1,000 steps/s:** 3 conditions × 2 seeds × 3M steps = 18M steps, about 5 hours.

### 9.2 What "peak" means and how to know we reached it

- Peak is not a fixed step count. Run the pilot first and plot the rolling mean of time-to-find over episodes.
- Declare convergence when the 100-episode rolling mean of time-to-find improves by less than 2% over the last 1M steps, in at least 2 of 3 seeds.
- Judge the final policy with a separate evaluation of 100 fresh episodes at fixed seeds, not with the training log (the training log mixes exploration noise and is noisy, Section 5).
- Keep the best checkpoint by held-out evaluation, not the last one.

### 9.3 Schedule (about 14 hours of work inside a 24-hour window)

| Hours | Work | Output |
| --- | --- | --- |
| 0–2 | Fix the bugs in Section 12, vectorise, benchmark throughput | Measured steps/s, unit tests pass |
| 2–3 | Pilot: proposed condition, 1 seed, 2M steps | Learning curve, convergence estimate |
| 3–4 | Pick learning rate and rollout size from the pilot; freeze all settings | Frozen config file |
| 4–10 | Main study, 3 conditions × 3 seeds, in 2 parallel processes | 9 checkpoints with logs |
| 10–12 | Randomised-training run and zero-shot victim-speed sweep | Robustness table |
| 12–14 | Held-out evaluation, heuristic baselines, figures and tables | Final results pack |
| 14–24 | Slack for Colab disconnects, reruns, and the 3D and thermal-injection integration | Buffer |

### 9.4 Rules for staying inside the day

1. Checkpoint model, optimiser, random-number state and log every 20 updates, to Google Drive, so any disconnect resumes instead of restarting.
2. Train only in the fast 2D grid. The 3D simulation, the thermal-image pipeline and the drone are used to deploy and test the trained policy, never to train it; physics simulation is far slower per step than this grid.
3. Freeze settings after the pilot. No tuning on the main-study seeds.
4. Colab's free-tier session length and GPU availability change; verify the current limits before planning, and design each job to be restartable.
5. If a condition has not converged inside its budget, report it as not converged rather than extending only that condition.

## 10. Evaluation protocol and the figures to produce

Clean results need three things we do not have yet: several seeds, a held-out evaluation, and baselines that are defined the way the report defines them. The protocol below fixes each.

### 10.1 Conditions to run

| Condition | Definition | Replaces |
| --- | --- | --- |
| Proposed | Belief patch + entropy, per-drone or shared belief, moving victim | Same |
| Coverage-only | Visit-recency patch, same moving victim | Same |
| Static-assumption belief | Belief update with **no diffusion**, trained and tested on the **moving** victim | The current frozen-victim run, which is a different experiment |
| Proposed without entropy | Belief patch only | New: isolates the headline feature |
| Greedy-to-belief heuristic | Each drone moves toward the highest-belief cell in its sensing reach | New: non-learning baseline |
| Lawnmower sweep | Fixed partitioned sweep | New: classical baseline |
| Randomised-shelter proposed | Shelter position and victim start drawn each episode | New: checks it is not memorising the corner |

### 10.2 Metrics

- **Success rate:** fraction of episodes where both victims are found within 200 steps.
- **Time to find all:** mean and median over successful episodes, with the success rate reported beside it. Do not average failures as 200 without saying so.
- **Time to first victim.**
- **Redundant re-visit rate (corrected):** fraction of drone-steps spent on a cell that was observed within the last K steps and whose belief is below a threshold set relative to the uniform prior (for example 0.1 × 1/900), not the absolute 0.01 used now.
- **Belief quality:** negative log-likelihood or Brier score of the belief at each victim's true cell over time; this checks the update is calibrated, not just that the policy wins.
- **Robustness:** all of the above at victim speeds 0.5×, 1×, 2× without retraining.

### 10.3 Statistics

1. Three training seeds per condition. Each trained policy is one statistical unit.
2. Evaluate each policy on the same 100 fixed-seed episodes.
3. Report the mean across seeds with a bootstrapped 95% interval, and per-seed values in an appendix table.
4. Compare conditions with a paired test across matched evaluation seeds. Report the effect size, not only significance.
5. Keep the evaluation script separate from training and run it only on frozen checkpoints.

### 10.4 Figures and tables for the review

1. Training curves, three conditions, mean line with seed band.
2. Bar chart of time to find all with intervals, plus the success rate.
3. Line chart of success rate against victim speed.
4. Entropy-on versus entropy-off ablation bar.
5. Belief calibration plot over time.
6. One side-by-side rollout GIF per condition from the same random seed (drone icons, victim markers, belief panel with colourbar).
7. One summary table: conditions × metrics, mean ± interval.

The existing `plot_results.py`, `evaluate.py` and `visualize.py` cover items 1, 2 and 6 in a basic form and need extending for the rest.

## 11. Supervisor's requests: 3D simulation, injected thermal images, lab drone with Raspberry Pi

The three requests are one pipeline: a virtual thermal camera feeds a detector, the detector's hits update the belief map, the trained policy picks a target cell, and a 3D simulator (and later the lab drone) fly there. The policy is still trained in the fast 2D grid; the pipeline is used to test and demonstrate it.

&#91;embedded content: pipeline · 8 stages, trained once in the 2D grid\]

The policy is trained once in the 2D grid. At test time its target cells become waypoints for the simulated drones, and later for the lab drone.

### 11.1 Thermal camera by image injection (confirmed by the supervisor)

The supervisor suggested using thermal images instead of buying a thermal camera. We read this as hardware-in-the-loop with a **virtual camera**: for each drone, take the true victim position and the drone's position, render the frame the camera would see, and feed it to the detector. Confirmed by the supervisor in a comment: in the code, inject a thermal image containing a person and show the detector's output; in the report and presentation, describe it as the thermal camera capturing the frames. The minimum code deliverable is therefore a script that loads an injected thermal frame, runs the detector, shows the box and score, and shows the resulting belief update. The per-drone frame generator below is the fuller version of the same idea.

1. Build frames by pasting real thermal person crops onto thermal background frames at the pixel position the drone's footprint implies. Candidate sources: an aerial infrared dataset such as HIT-UAV, or FLIR ADAS (from memory; verify licence and availability).
2. Run a small detector (for example YOLOv8-nano, fine-tuned on that data) on each frame.
3. Measure the detector's true-positive and false-positive rates on held-out frames, by altitude band.
4. Put those measured rates into the environment as P\_D and P\_F, replacing the invented 0.9 and 0.02, and use the matching log-likelihood-ratio belief update (Section 8).
5. Train with the rates randomised around the measured values so the policy tolerates detector error.

Edge cases to handle: person appears smaller at higher altitude, hot ground or vehicles causing false alarms, partial visibility at the footprint edge, two victims in one frame, motion blur, and day/night contrast changes.

### 11.2 3D simulation

- **Simulator.** gym-pybullet-drones, [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones). It supports several drones, PID flight control and a Gymnasium API. Gazebo with PX4 is more realistic and much heavier; use it only if time remains.
- **Hierarchy.** The learned policy runs at about 1–2 Hz and outputs a target grid cell for each drone. A PID controller flies the drone there. No retraining in 3D is needed.
- **Altitude.** Make the sensing footprint a function of height: radius = h × tan(FOV/2). This gives `sensing_radius` a physical meaning. Letting altitude be an action is a stretch goal.
- **Mapping.** Define the metres-per-cell scale and the arena size once, and keep it identical in the grid, the 3D simulator and the real arena.
- **Reference.** [AM-MAPPO](https://www.mdpi.com/2504-446X/8/8/378) searches for moving targets in 3D with MAPPO and uses action masks for collision avoidance.
- **Edge cases.** Inter-drone collisions, boundary clipping, wind and drift, controller lag against the 1–2 Hz policy, and the camera footprint changing as the drone tilts.

### 11.3 What the drones and the base station should share

Share evidence, not maps: each drone sends a few tens of bytes per step about what it just sensed, and the base station keeps the global belief and sends back only a small summary. Evidence adds up exactly in log-odds, so a lost or repeated packet does not corrupt anyone's belief. These are proposals to test in the simulation, not final numbers.

| Message | From → to | Content | Rate | Why |
| --- | --- | --- | --- | --- |
| Evidence packet | Drone → neighbours in range and base | Drone id, sequence number, step, own cell, and the sensing result: window centre and radius with hit or miss | Every step, tens of bytes | Log-odds evidence is additive, so each packet is applied once (dedupe by id and sequence number) and order does not matter |
| Position and intent | Drone → neighbours and base | Own cell and the target cell it is heading to | At each policy decision | Teammates avoid choosing the same target; the base can monitor |
| Victim alert | Drone → base, acknowledged | Cell, detector confidence, frame reference | On detection | The only message that must be retransmitted until acknowledged |
| Belief summary | Base → drones | Top-K hottest cells with probabilities, per-region entropy, found-victim list, timestamp | Every few steps, a few hundred bytes | Gives drones out of range the global picture without sending a 900-cell map |
| Safety command | Base → drones | Return, hold, geofence change | On event | Operator control |
| Not sent | None | Full belief maps, raw thermal frames, network weights | Never in flight | Too large, and the evidence packets already carry the information |

**Fusion rule.** Each drone keeps its own belief and applies every evidence packet once, whether it sensed the cells itself or heard about them. A packet that arrives late is first diffused by its age, so it is not treated as fresh. The base station applies the same rule to all packets, so it holds the global belief and fills gaps for drones that were out of range. The actor reads a patch of the drone's own belief, not a global one; the training critic may still see the global state.

**How to check it in the simulation.** Model a communication range, packet loss and a delay of a few steps, then compare four cases on the same episodes: full sharing, this evidence-only design, neighbour-only sharing with no base station, and no sharing. Keep the evidence-only design if it stays close to full sharing at far lower bandwidth. The research model should settle the exact late-packet rule and the message sizes (Section 13, tasks 4 and 5).

### 11.4 Lab drone with a Raspberry Pi (after the simulation)

Hardware is optional and comes last: do not plan around it, or let it delay the simulation, until the simulation results in Section 10 are done.

When the time comes, the Raspberry Pi on the lab drone would run the small policy network, the detector on the injected thermal frame, and its own belief map. It would exchange the messages above with the base station and send waypoints to the drone's flight controller. Check then, not now: which Pi model and detector size give a usable frame rate, what the lab drone's controller accepts for waypoints (its telemetry is limited), and what position source works indoors. The simulator-only demo stays the guaranteed deliverable.

## 12. Known problems in the current implementation, risks and reality checks

The current code runs and the belief-conditioned policy clearly outperforms the other two conditions, but at least four implementation flaws could change that result and one reported metric is invalid; they should be fixed before the final training run. These come from reading our own code and logs; the first is confirmed by the numbers in our logs; the rest come from reading the code and each needs a quick test to confirm.

### 12.1 Implementation issues

| # | Issue | Evidence | Effect | Fix | Severity |
| --- | --- | --- | --- | --- | --- |
| 1 | Redundant re-visit penalty and metric are degenerate | Threshold is an absolute 0.01, but the belief sums to 1 over 900 cells, so a typical cell is about 0.0011. Logged "redundant" is 800 at update 1 (200 steps × 4 drones) and, at the end, almost exactly 4 × steps-to-find: 226 against 4 × 56.9 = 228, 507 against 4 × 127.0 = 508, 571 against 4 × 142.8 = 571 | The −0.02 penalty is a constant extra step cost, not a targeted one. The metric restates time-to-find, so it is not independent evidence. The earlier review slides that used it should be corrected | Use a threshold relative to the uniform prior and count only cells observed in the last K steps | High |
| 2 | Critic input is one step ahead | In `train.py` the global state and value are taken after `env.step`, but stored beside the pre-step observation and action | Values and advantages are misaligned by one step, which adds noise to learning | Compute state and value before stepping; bootstrap properly at truncation | High |
| 3 | Observation saturates | Belief patch is multiplied by 900 and then clipped to \[−1, 1\]. Any cell at or above the uniform prior reads 1.0 | The policy cannot see strong hot spots, only "cleared" versus "not cleared". This weakens the claim that it uses belief, and makes belief close to a coverage map | Normalise by a fixed or running scale without clipping, or use log-belief | High |
| 4 | Fixed shelter plus absolute position | Victims always drift toward cell (29,29); absolute own position is in the observation; both victims in the demo were found in that corner | A policy can learn "go to the corner" without using belief. Coverage-only may have failed for other reasons | Randomise shelter and victim starts each episode; test with absolute position removed | High |
| 5 | Belief update is not a proper filter | Fixed constants, window-level rather than cell-level evidence, one false-positive draw per drone-step, found victims stay in the belief, two victims share one map that sums to 1 | Belief is not calibrated to P\_D and P\_F; the map is ambiguous with 2 victims | Log-likelihood-ratio update from P\_D and P\_F (Section 8); one map per victim or sum to the expected count; remove found victims | Medium |
| 6 | "Static-target" baseline is a different experiment | `--victim_moves 0` freezes the victim in the environment | Not the report's definition (a policy that assumes a frozen victim, tested on a moving one). It is also an easier task | Train with no diffusion, test on the moving victim (Section 10.1) | Medium |
| 7 | Small, un-normalised training batches | One environment, 800 samples per update, constant learning rate, no observation or value normalisation, truncation treated as terminal | Noisy training; "peak" is hard to reach | 16–32 parallel environments, larger batches, observation and value normalisation, learning-rate decay | Medium |
| 8 | No resumable checkpoints | Only network weights are saved; no optimiser or random-number state | A Colab disconnect loses runs; we already lost checkpoints once | Save full training state and copy to Drive | Medium |
| 9 | Single seed, one run | Both uploaded notebooks show identical logs | No variance estimate | 3 seeds per condition (Section 10.3) | Medium |
| 10 | Time-to-find is censored | Episodes cap at 200 and a failure is logged as 200 | Means mix success and failure | Report success rate and time given success | Low |
| 11 | Observation size tied to 4 drones | Observation vector grows with teammates | The hardware plan uses 3 drones, so the policy must be retrained | Train at 3 drones, or use a permutation-invariant teammate encoding | Medium |
| 12 | QMIX not implemented | Report names MAPPO/QMIX | Panel may ask | Either implement as a comparison or state it was dropped for time | Low |

### 12.2 Project risks

- **Novelty claim.** The Review-1 gap statement is too strong (Section 7). Reword it before the next review.
- **Scope growth.** Training fixes, 3D simulation, thermal detector and a drone is a lot for one term. The schedule in Section 9.3 keeps training under a day, but integration time is separate. Agree a minimum demo with the supervisor.
- **Hardware.** The lab drone's flight controller may not accept waypoints from the Raspberry Pi, and indoor position is unresolved. Hardware comes last; the simulator-only demo is the guaranteed deliverable.
- **Sim-to-real.** The thermal detector's real error rates, wind, latency and marker-tracking jitter will all differ from the grid. Domain randomisation reduces the gap but does not remove it.
- **Overstated results.** The earlier review deck cites the invalid redundant-visit numbers and one-rollout demonstrations. Present only the corrected, multi-seed results next time.

## 13. Handoff: tasks that need deep reasoning or deep research

These 13 tasks are the ones I could not settle with a quick search or a read of our code, and each changes what we build; the research model should do P0 items before anything else because later items depend on their answers.

Everything in Sections 3–5 and 12 describes our own code and logs and can be taken as given. Everything in Sections 6–8 about other papers is limited to what was opened in one session (Section 6 says which). Treat unverified rows as leads, not facts.

| # | Priority | Task | Why it needs deep work | Deliverable |
| --- | --- | --- | --- | --- |
| 1 | P0 | Read Su & Qian (2023) Sections 5–6 in full and extract observation encoding, reward, network sizes, training budget, baselines and results | The paper is our closest competitor; its design may be directly adoptable or may expose a weakness in ours. The page's mathematics made it too long to pull in here | A side-by-side design table and a list of what to adopt or reject, each with reasons |
| 2 | P0 | Verify the Review-1 list of 20 papers, especially Chi (2025), Zhang (2024 DDPG belief maps), Wei (2024), Xu (2025 CA-MAPPO), Hong (2025), and search for 2025–2026 additions and any with public code | I could not find Chi (2025) at all in search. Row 8 was already wrong in the Review-1 table, so other rows may be too. The novelty claim depends on this | A corrected table of the 20 papers, each claim checked against the paper, plus the safest wording for our contribution |
| 3 | P0 | Review `belief_sar_env.py` and `train.py`, confirm or refute issues 1–5 and 7 in Section 12.1, and find any more | I read my own code; an independent, careful read is needed. Two issues (critic misalignment, observation saturation) need a test, not just argument | A confirmed bug list with minimal tests, then a corrected, vectorised MAPPO and environment |
| 4 | P0 | Derive the correct belief model: cell-level log-likelihood-ratio update from P\_D and P\_F, handling of two victims, removal of found victims, and a prediction step based on the victim's motion model instead of a 3×3 blur | Needs probabilistic reasoning. A motion-aware prediction step gives the policy knowledge of the shelter drift. Decide how much of that is fair, and give the baselines equal information | The update equations, a decision on motion knowledge, and a numerical test showing calibration |
| 5 | P0 | Redesign the observation: remove saturation, decide on absolute position, shared versus per-drone belief with range-limited fusion (start from the message design in Section 11.3 and verify it in simulation), and a teammate encoding that works for 3 or 4 drones | Interacts with fairness of the baselines and with the hardware plan. Wrong choices could make the policy memorise geography | Observation spec, an ablation plan, and a recommendation backed by small experiments |
| 6 | P1 | Reward design: replace the broken redundant-visit penalty; evaluate information-gain (entropy-reduction) reward and potential-based shaping from MPRS-MAPPO; check for reward hacking | Reward shaping can change the optimal policy unless done carefully | A reward spec with proof or test of policy invariance, and a comparison on the pilot |
| 7 | P1 | Explain and fix the coverage-only baseline collapse: bug, weak observation, or a genuine result? Make it a fair, strong baseline (for example add a time-since-observed map as Su & Qian do) | A weak baseline makes our result look better than it is. Needs controlled experiments | A diagnosis, a strengthened baseline, and a rerun |
| 8 | P1 | Throughput engineering: a vectorised environment and batched inference that reaches the steps-per-second assumed in Section 9; confirm current Colab session and GPU limits | The time budget is an estimate until measured. Needs profiling and code | Benchmark numbers, the code, and a revised budget table |
| 9 | P1 | The supervisor has confirmed the meaning (inject a thermal image with a person in the code and show the output; describe it as a real camera in the report), so build that minimum demonstration first; pick a thermal dataset and detector; design how detector error rates map to P\_D and P\_F by altitude | Ambiguous requirement; wrong guess wastes weeks. Also dataset licence and domain-gap questions | A recommended interpretation, a short note for the supervisor to confirm, and a detector calibration plan |
| 10 | P3, later | Lab drone with Raspberry Pi (optional, only after the simulation is finished): what the flight controller accepts, Pi model and detector speed, and the message design in Section 11.3 | Hardware facts are unverified; spend no effort here until the simulation is done | A short feasibility note, nothing more |
| 11 | P2 | 3D simulation design in gym-pybullet-drones (and whether Gazebo is worth it): policy-to-waypoint interface, footprint from altitude, collision handling, frame rates | Integration design with several moving parts | Architecture and a runnable skeleton |
| 12 | P2 | Experimental design: number of seeds and episodes, tests, convergence rule, and the entropy and no-diffusion ablations | Statistical choices affect how defensible the claims are | A fixed evaluation plan and analysis script |
| 13 | P2 | Review messaging: the corrected novelty statement, a Q&A sheet (especially "Su & Qian already does this"), corrected slides replacing the invalid redundant-visit numbers, and the QMIX decision | Needs the answers to tasks 1–2 and 7 first | Updated deck outline and a one-page Q&A |

**Suggested order:** 1, 2 and 3 in parallel, then 4 and 5, then rerun the pilot (Section 9.3) before doing 6, 7 and 12. Tasks 9 and 11 and the message design in Section 11.3 can run alongside once the observation and belief design is fixed; task 10 (hardware) waits until the simulation is finished.

## 14. Sources

Pages opened in full this session: Su & Qian (Sections 1–4 only), the BSR-MAPPO repository README, and the Multi-UAV-ISR repository README. Everything else below was read as an abstract or search snippet.

**Papers**

- [Su & Qian (2023), Multi-UAV Cooperative Searching and Tracking for Moving Targets, Applied Sciences 13(21):11905](https://www.mdpi.com/2076-3417/13/21/11905)
- [Liao et al. (2024), Multi-UAV Escape Target Search (ETS–MAPPO), Sensors 24(21):6859](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548590/)
- [Reinforcement-Learning-Based Multi-UAV Cooperative Search for Moving Targets in 3D Scenarios (AM-MAPPO), Drones 8(8):378, 2024](https://www.mdpi.com/2504-446X/8/8/378)
- [Hong et al. (2025), Multi-UAV Dynamic Target Search Based on Multi-Potential-Field Fusion Reward Shaping MAPPO, Drones 9(11):770](https://doi.org/10.3390/drones9110770)
- [Searching and Tracking an Unknown Number of Targets: A Learning-Based Method Enhanced with Maps Merging, Sensors 21(4):1076, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7915622/)
- [Zhang et al. (2024), Multi-UAV Reconnaissance and Search with Limited Communication Range Using Semantic Episodic Memory, Drones](https://doaj.org/article/ed599f24f5d14e39a3454c2dc6477dfb)
- [Yehoshua et al. (2021), Decentralized Reinforcement Learning for Multi-Target Search and Detection by a Team of Drones, arXiv 2103.09520](https://arxiv.org/abs/2103.09520)
- [Li et al. (2019), Profit-Driven Adaptive Moving Targets Search with UAV Swarms, Sensors 19(7):1545](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480202)
- [Multi-Agent Cooperative Target Search, Sensors 14(6):9408, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4118349/)
- [AS-MAPPO, attention with shared observations for dynamic targets (2025)](https://link.springer.com/article/10.1007/s44163-025-00411-9)
- [Multi-UAVs Searching and Tracking for USV Swarm: A Center-Sub-Critics Reinforcement Learning Approach, Drones 10:123](https://mdpi-res.com/d_attachment/drones/drones-10-00123/article_deploy/drones-10-00123.pdf)

**Code and tools**

- [BSR-MAPPO (choiaidrone/belief-aware-graph-token-mappo), MIT](https://github.com/choiaidrone/belief-aware-graph-token-mappo)
- [Multi-UAV-ISR (quialga), Apache-2.0](https://github.com/quialga/Multi-UAV-ISR)
- [gym-pybullet-drones (utiasDSL)](https://github.com/utiasDSL/gym-pybullet-drones)

**Our own materials (not linked)**

- Project Review-I report (docx), `sar_marl_code.zip` (environment, MAPPO, training, evaluation, visualisation), two Colab notebooks with training logs, and the earlier review decks. Numbers in Sections 4 and 5 come from the notebook logs.

Figures marked "estimate" (Section 9) and any dataset or product detail marked "from memory" (Section 11) are not sourced.
