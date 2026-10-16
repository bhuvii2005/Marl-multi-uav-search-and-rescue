"""
BeliefSAREnv — a 2D grid multi-UAV search-and-rescue environment.

Implements, from Project_Review1_Report.docx:
  - 2D grid world (default 30x30), N drones, 1-3 non-adversarial moving victims
  - Victim motion: stochastic drift / biased random walk toward a "shelter" point
  - Belief map: per-cell probability of victim presence, updated with a Bayesian
    filter from noisy drone detections, with decay/diffusion each step
  - Two observation modes:
      obs_mode="belief"   -> agents see a local patch of the belief map + its
                              entropy (the proposed approach)
      obs_mode="coverage" -> agents see a local patch of a visited/unvisited
                              coverage mask instead (the coverage-only baseline)
  - A `victim_moves=False` flag freezes victim motion, reproducing the
    "static-target belief-MARL" baseline while keeping obs_mode="belief".

This is a PettingZoo ParallelEnv (all agents act simultaneously), which is
the API MAPPO training loops expect.
"""
from __future__ import annotations

import functools
import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

# Discrete movement actions
ACTIONS = {
    0: (0, 0),   # stay
    1: (0, 1),   # up
    2: (0, -1),  # down
    3: (1, 0),   # right
    4: (-1, 0),  # left
}


class BeliefSAREnv(ParallelEnv):
    metadata = {"name": "belief_sar_v0", "render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 30,
        n_drones: int = 4,
        n_victims: int = 2,
        sensing_radius: int = 2,
        comm_radius: int = 8,
        detect_prob: float = 0.9,
        false_positive_prob: float = 0.02,
        belief_patch_radius: int = 2,
        diffusion_rate: float = 0.02,
        victim_step_prob: float = 0.6,
        victim_speed: float = 1.0,
        shelter_pos: tuple | None = None,
        max_cycles: int = 200,
        obs_mode: str = "belief",          # "belief" | "coverage"
        victim_moves: bool = True,          # False -> static-target baseline
        redundant_revisit_penalty: float = 0.02,
        revisit_belief_threshold: float = 0.01,
        seed: int | None = None,
    ):
        assert obs_mode in ("belief", "coverage")
        self.grid_size = grid_size
        self.n_drones = n_drones
        self.n_victims = n_victims
        self.sensing_radius = sensing_radius
        self.comm_radius = comm_radius
        self.detect_prob = detect_prob
        self.false_positive_prob = false_positive_prob
        self.belief_patch_radius = belief_patch_radius
        self.diffusion_rate = diffusion_rate
        self.victim_step_prob = victim_step_prob
        self.victim_speed = victim_speed
        self.shelter_pos = shelter_pos or (grid_size - 1, grid_size - 1)
        self.max_cycles = max_cycles
        self.obs_mode = obs_mode
        self.victim_moves = victim_moves
        self.redundant_revisit_penalty = redundant_revisit_penalty
        self.revisit_belief_threshold = revisit_belief_threshold

        self.possible_agents = [f"drone_{i}" for i in range(n_drones)]
        self.agents = list(self.possible_agents)

        patch_dim = (2 * belief_patch_radius + 1) ** 2
        # own_pos(2) + entropy(1) + patch(patch_dim) + other-agent rel positions (2 * (n_drones-1))
        self._obs_dim = 2 + 1 + patch_dim + 2 * (n_drones - 1)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # PettingZoo required space accessors
    # ------------------------------------------------------------------ #
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(ACTIONS))

    def global_state_size(self):
        """Size of the centralized-critic global state vector."""
        return self.n_drones * 2 + self.grid_size * self.grid_size

    # ------------------------------------------------------------------ #
    # Core RL API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0

        self.drone_pos = self._rng.integers(0, self.grid_size, size=(self.n_drones, 2))
        self.victim_pos = self._rng.integers(0, self.grid_size, size=(self.n_victims, 2)).astype(float)
        self.found = np.zeros(self.n_victims, dtype=bool)

        # Uniform prior belief over "any victim present in this cell"
        self.belief = np.full((self.grid_size, self.grid_size), 1.0 / (self.grid_size ** 2), dtype=np.float32)
        self.coverage = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)  # visit recency, 1=fresh

        obs = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: dict):
        self.t += 1

        # --- move drones -------------------------------------------------
        for i, a in enumerate(self.agents):
            dx, dy = ACTIONS[int(actions[a])]
            self.drone_pos[i] = np.clip(self.drone_pos[i] + np.array([dx, dy]), 0, self.grid_size - 1)

        # --- move victims (stochastic drift toward shelter) --------------
        if self.victim_moves:
            for v in range(self.n_victims):
                if self.found[v]:
                    continue
                if self._rng.random() < self.victim_step_prob:
                    direction = np.array(self.shelter_pos) - self.victim_pos[v]
                    norm = np.linalg.norm(direction) + 1e-6
                    bias = direction / norm
                    noise = self._rng.normal(0, 1.0, size=2)
                    move = self.victim_speed * (0.6 * bias + 0.4 * noise)
                    self.victim_pos[v] = np.clip(
                        self.victim_pos[v] + move, 0, self.grid_size - 1
                    )

        # --- coverage recency decay ---------------------------------------
        self.coverage *= 0.98

        # --- noisy detections + Bayesian belief update --------------------
        rewards = {a: -0.01 for a in self.agents}  # small per-step cost -> minimize time-to-find
        redundant_hits = 0
        detections = np.zeros_like(self.belief)  # log-likelihood-ish accumulator

        for i, a in enumerate(self.agents):
            px, py = self.drone_pos[i]
            self.coverage[px, py] = 1.0
            if self.coverage[px, py] < self.revisit_belief_threshold:
                pass  # unreachable branch kept for clarity of "coverage" semantics

            # redundant-search penalty: revisiting a near-zero-belief cell
            if self.belief[px, py] < self.revisit_belief_threshold:
                redundant_hits += 1
                rewards[a] -= self.redundant_revisit_penalty

            lo = np.maximum([px - self.sensing_radius, py - self.sensing_radius], 0)
            hi = np.minimum(
                [px + self.sensing_radius + 1, py + self.sensing_radius + 1],
                [self.grid_size, self.grid_size],
            )
            window = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))

            victim_here = False
            for v in range(self.n_victims):
                if self.found[v]:
                    continue
                vx, vy = int(round(self.victim_pos[v][0])), int(round(self.victim_pos[v][1]))
                if lo[0] <= vx < hi[0] and lo[1] <= vy < hi[1]:
                    if self._rng.random() < self.detect_prob:
                        victim_here = True
                        self.found[v] = True
                        rewards[a] += 10.0
                        for a2 in self.agents:
                            rewards[a2] += 2.0  # shared team bonus

            if victim_here:
                detections[window] += 4.0  # strong positive evidence
            else:
                # small negative evidence in the sensed window (unless false positive)
                if self._rng.random() < self.false_positive_prob:
                    detections[window] += 1.0
                else:
                    detections[window] -= 0.5

        # Bayesian-style update in log-odds space, then renormalize
        log_odds = np.log(self.belief / (1 - self.belief + 1e-9) + 1e-9)
        log_odds += detections
        odds = np.exp(np.clip(log_odds, -20, 20))
        self.belief = (odds / (1 + odds)).astype(np.float32)

        # diffusion / decay: information about unvisited regions goes stale
        self.belief = (1 - self.diffusion_rate) * self.belief + self.diffusion_rate * self._blur(self.belief)
        self.belief /= self.belief.sum() + 1e-9

        # --- termination ---------------------------------------------------
        all_found = bool(self.found.all())
        truncate = self.t >= self.max_cycles
        terminations = {a: all_found for a in self.agents}
        truncations = {a: truncate for a in self.agents}

        obs = {a: self._get_obs(i) for i, a in enumerate(self.agents)}
        infos = {a: {"redundant_hits": redundant_hits, "found": int(self.found.sum())} for a in self.agents}

        if all_found or truncate:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _blur(self, mat):
        """Cheap 3x3 box blur used to model belief diffusing to neighbouring cells."""
        padded = np.pad(mat, 1, mode="edge")
        out = np.zeros_like(mat)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                out += padded[1 + dx:1 + dx + mat.shape[0], 1 + dy:1 + dy + mat.shape[1]]
        return out / 9.0

    def _local_entropy(self, px, py):
        r = self.belief_patch_radius
        lo = np.maximum([px - r, py - r], 0)
        hi = np.minimum([px + r + 1, py + r + 1], [self.grid_size, self.grid_size])
        patch = self.belief[lo[0]:hi[0], lo[1]:hi[1]]
        p = patch / (patch.sum() + 1e-9)
        ent = -(p * np.log(p + 1e-9)).sum()
        return ent, patch

    def _get_obs(self, i):
        px, py = self.drone_pos[i]
        r = self.belief_patch_radius
        source_map = self.belief if self.obs_mode == "belief" else self.coverage

        lo = np.maximum([px - r, py - r], 0)
        hi = np.minimum([px + r + 1, py + r + 1], [self.grid_size, self.grid_size])
        patch = source_map[lo[0]:hi[0], lo[1]:hi[1]]
        full_patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
        off = (r - (px - lo[0]), r - (py - lo[1]))
        full_patch[off[0]:off[0] + patch.shape[0], off[1]:off[1] + patch.shape[1]] = patch

        if self.obs_mode == "belief":
            ent, _ = self._local_entropy(px, py)
            ent_feat = np.array([ent / np.log((2 * r + 1) ** 2 + 1e-9 + 1)], dtype=np.float32)
            patch_feat = full_patch.flatten() * self.grid_size ** 2  # rescale ~O(1)
        else:
            ent_feat = np.array([full_patch.mean()], dtype=np.float32)  # "coverage density" in place of entropy
            patch_feat = full_patch.flatten()

        own_pos = np.array([px, py], dtype=np.float32) / self.grid_size

        others = []
        for j in range(self.n_drones):
            if j == i:
                continue
            rel = (self.drone_pos[j] - self.drone_pos[i]).astype(np.float32) / self.grid_size
            dist = np.linalg.norm(self.drone_pos[j] - self.drone_pos[i])
            if dist > self.comm_radius:
                rel = np.zeros(2, dtype=np.float32)  # out of comm range -> no info
            others.append(rel)
        others_feat = np.concatenate(others) if others else np.zeros(0, dtype=np.float32)

        obs = np.concatenate([own_pos, ent_feat, patch_feat, others_feat]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)

    def get_global_state(self):
        """Flattened global state for the centralized MAPPO critic."""
        return np.concatenate(
            [self.drone_pos.flatten().astype(np.float32) / self.grid_size, self.belief.flatten()]
        )

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype="<U1")
        for (x, y) in self.drone_pos:
            grid[x, y] = "D"
        for v, (x, y) in enumerate(self.victim_pos):
            if not self.found[v]:
                grid[int(x), int(y)] = "V"
        print("\n".join("".join(row) for row in grid))
        print(f"t={self.t} found={self.found.sum()}/{self.n_victims}")
