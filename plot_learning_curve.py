# File: plot_learning_curve.py
# Learning curve for the Realistic headline agent (Figure: DRL reward vs training
# steps). Globs the actual checkpoints saved by train_revision.py, evaluates each
# over several seeds with the corrected env, and plots the progression against the
# Greedy baseline. Saves a vector PDF for the manuscript.

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qkd_env import SatelliteQKDEnv
from policies import GreedyPolicy, RandomPolicy, DRLPolicy
from evaluate_revision import run_episode

SCENARIO = "realistic"
MODEL_DIR = "models/realistic_3M"
SEEDS = list(range(1000, 1020))  # 20 seeds, matching the headline evaluation
# Sub-sample checkpoints (steps in millions*1e6) to keep the run fast yet smooth.
KEEP_STEPS = {200_000, 600_000, 1_000_000, 1_400_000, 1_800_000,
              2_200_000, 2_600_000, 3_000_000}


def eval_model_path(path, seeds):
    env = SatelliteQKDEnv(num_ogs=5, scenario=SCENARIO)
    pol = DRLPolicy(path, env, device="cpu")
    rewards = [run_episode(env, pol, s) for s in seeds]
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    # Collect checkpoints: ppo_realistic_3M_<step>_steps.zip plus final_model.zip.
    pts = []
    for f in glob.glob(os.path.join(MODEL_DIR, "ppo_realistic_3M_*_steps.zip")):
        m = re.search(r"_(\d+)_steps\.zip$", f)
        if m:
            pts.append((int(m.group(1)), f))
    pts.sort()
    pts = [(s, f) for (s, f) in pts if s in KEEP_STEPS]
    final = os.path.join(MODEL_DIR, "final_model.zip")
    if os.path.exists(final):
        last_step = pts[-1][0] if pts else 3_000_000
        pts.append((max(last_step, 3_000_000), final))

    steps, means, stds = [], [], []
    for step, path in pts:
        mu, sd = eval_model_path(path, SEEDS)
        steps.append(step); means.append(mu); stds.append(sd)
        print(f"step={step:>9,}  DRL mean={mu:9,.1f}  std={sd:8,.1f}")

    # Baselines for context.
    env = SatelliteQKDEnv(num_ogs=5, scenario=SCENARIO)
    g = GreedyPolicy(); r = RandomPolicy()
    greedy = np.mean([run_episode(env, g, s) for s in SEEDS])
    rand = np.mean([run_episode(env, r, s) for s in SEEDS])
    env.close()
    print(f"Greedy={greedy:,.1f}  Random={rand:,.1f}")

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    steps_m = np.array(steps) / 1e6
    ax.errorbar(steps_m, means, yerr=stds, marker="o", markersize=4, capsize=2,
                linewidth=1.3, color="royalblue", label="DRL")
    ax.axhline(greedy, color="darkorange", linestyle="--", linewidth=1.3, label="Greedy")
    ax.axhline(rand, color="forestgreen", linestyle=":", linewidth=1.3, label="Random")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Total secure key (bits)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "learning_curve.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
