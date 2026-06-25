# File: policies.py
# Baseline and heuristic scheduling policies for the Sat-QKD revision study.
#
# All policies share a small stateful interface so they can be evaluated
# identically by the same environment loop:
#     p.reset()            -> called once at the start of each episode
#     p.act(obs, env)      -> returns a discrete action (int)
#
# Crucially, every policy is scored by the SAME environment reward
# (env.step reward = secure key bits minus operational penalties). This is the
# fair-comparison guarantee that addresses Reviewer 2's central concern.

import numpy as np
from qkd_env import MIN_ELEVATION_DEGREES


def _decode_links(obs, env):
    """Return a list of (elevation_deg, ogs_index) for OGS that are both
    geometrically visible and (in dynamic/realistic scenarios) cloud-free.
    Mirrors exactly the observation layout produced by SatelliteQKDEnv._get_obs."""
    base_obs_dim = 4 + 3 * env.num_ogs
    candidates = []
    for i in range(env.num_ogs):
        # Cloud gate (only present for dynamic / realistic scenarios).
        if env.is_dynamic_weather and obs[base_obs_dim + i] > 0:
            continue
        elev_norm = obs[4 + 3 * i]
        elev_deg = elev_norm * 90.0
        if elev_deg < MIN_ELEVATION_DEGREES:
            continue
        candidates.append((elev_deg, i))
    return candidates


def _elevation_of(obs, idx):
    """Normalized-to-degrees elevation of a single OGS index (negative if hidden)."""
    return obs[4 + 3 * idx] * 90.0


class RandomPolicy:
    """Uniform random action. Serves as a sanity-check lower bound: it shows the
    task is non-trivial (random scores negative once penalties exist)."""

    name = "Random"

    def reset(self):
        pass

    def act(self, obs, env):
        return env.action_space.sample()


class GreedyPolicy:
    """Strong, domain-aware but myopic heuristic: always connect to the highest
    available (visible, cloud-free) OGS. Identical logic to the original paper's
    get_greedy_action. It is blind to switching cost, so in the Realistic scenario
    it pays the 2-minute setup penalty every time a different OGS becomes higher."""

    name = "Greedy"

    def reset(self):
        pass

    def act(self, obs, env):
        candidates = _decode_links(obs, env)
        if not candidates:
            return env.num_ogs  # idle
        # max by elevation; tie-break on lowest index for determinism
        best = max(candidates, key=lambda c: (c[0], -c[1]))
        return best[1]


class HybridPolicy:
    """Cost-aware heuristic: greedy selection with switching hysteresis.

    It keeps the current link unless a candidate OGS exceeds the current link's
    elevation by at least `margin_deg`. The margin encodes the intuition that a
    switch is only worthwhile if the elevation (hence key-rate) gain is large
    enough to amortize the fixed setup cost during which throughput is zero.

    This is the 'learning + heuristic' style baseline requested by Reviewer 1
    (point 4). With margin_deg = 0 it degenerates exactly to GreedyPolicy."""

    def __init__(self, margin_deg=15.0):
        self.margin_deg = float(margin_deg)
        self.current = None
        self.name = f"Hybrid(m={int(self.margin_deg)})"

    def reset(self):
        self.current = None

    def act(self, obs, env):
        candidates = _decode_links(obs, env)
        if not candidates:
            self.current = None
            return env.num_ogs  # idle

        best_elev, best_idx = max(candidates, key=lambda c: (c[0], -c[1]))

        # Is the currently held link still usable this step?
        if self.current is not None:
            valid_idxs = {i for _, i in candidates}
            if self.current in valid_idxs:
                cur_elev = _elevation_of(obs, self.current)
                # Switch only if the gain clears the hysteresis margin.
                if best_elev - cur_elev > self.margin_deg:
                    self.current = best_idx
                # else: hold current link, avoid paying the setup cost
            else:
                # Current link dropped (set below horizon or clouded) -> must move.
                self.current = best_idx
        else:
            self.current = best_idx

        return self.current


class DRLPolicy:
    """Wraps a Stable-Baselines3 PPO model. Imported lazily so the heuristic
    policies and the environment can be smoke-tested without torch/SB3.

    The released models were saved under numpy 2.x (their pickled spaces reference
    numpy._core), while SB3 still requires numpy<2 at runtime. To load them under a
    numpy-1 stack we pass `custom_objects` built from a reference environment, so
    SB3 skips deserializing the numpy-2 pickled observation/action spaces and
    learning-rate / clip-range schedules. This is version-proof: it does not depend
    on the exact numpy/SB3 version the model was trained with."""

    name = "DRL"

    def __init__(self, model_path, ref_env, device="cpu"):
        from stable_baselines3 import PPO  # local import on purpose
        custom_objects = {
            "observation_space": ref_env.observation_space,
            "action_space": ref_env.action_space,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        self.model = PPO.load(model_path, device=device, custom_objects=custom_objects)

    def reset(self):
        pass

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
