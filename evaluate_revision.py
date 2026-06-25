# File: evaluate_revision.py
# Multi-seed evaluation for the revised Sat-QKD study.
#
# Differences vs the original evaluate_final.py:
#   1. Evaluates over MULTIPLE seeds and reports mean +/- std (the original used a
#      single seed=123). This removes single-seed luck and matches transaction-grade
#      rigor.
#   2. Adds the cost-aware Hybrid (greedy + switching hysteresis) baseline.
#   3. Every policy is scored by the SAME environment reward, made explicit here so
#      the fairness of the comparison is auditable (Reviewer 2).
#
# Pure-numpy core: it imports torch/SB3 only if a DRL model path is given, so the
# heuristic comparison can be smoke-tested in a minimal environment.

import os
import argparse
import numpy as np

from qkd_env import SatelliteQKDEnv
from policies import RandomPolicy, GreedyPolicy, HybridPolicy, DRLPolicy


def run_episode(env, policy, seed):
    """Run one full 24h episode and return the total accumulated reward
    (total secure key bits minus operational penalties)."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.act(obs, env)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    return total_reward


def evaluate_policy(scenario, policy, seeds, switching_cost_minutes=2):
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario,
                          switching_cost_minutes=switching_cost_minutes)
    rewards = np.array([run_episode(env, policy, s) for s in seeds], dtype=np.float64)
    env.close()
    return rewards


def build_policies(model_path=None, device="cpu", hybrid_margin=15.0):
    policies = [GreedyPolicy(), HybridPolicy(margin_deg=hybrid_margin), RandomPolicy()]
    if model_path and os.path.exists(model_path):
        policies.insert(0, DRLPolicy(model_path, device=device))
    elif model_path:
        print(f"WARNING: DRL model not found at {model_path}; skipping DRL.")
    return policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="realistic",
                    choices=["static", "dynamic", "realistic"])
    ap.add_argument("--model", default="models/realistic_3M/final_model.zip",
                    help="DRL model path; ignored if missing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=20, help="number of seeds")
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--switching-cost", type=int, default=2)
    ap.add_argument("--hybrid-margin", type=float, default=15.0)
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    policies = build_policies(args.model, args.device, args.hybrid_margin)

    print(f"\nScenario={args.scenario}  switching_cost={args.switching_cost} min  "
          f"seeds={args.seeds}")
    print("-" * 64)
    results = {}
    for p in policies:
        r = evaluate_policy(args.scenario, p, seeds, args.switching_cost)
        results[p.name] = r
        print(f"{p.name:16s}  mean={r.mean():12,.1f}  std={r.std():10,.1f}  "
              f"min={r.min():12,.1f}  max={r.max():12,.1f}")

    # Relative performance normalized to Greedy (matches the paper's convention).
    if "Greedy" in results:
        base = results["Greedy"].mean()
        print("-" * 64)
        for name, r in results.items():
            rel = 100.0 * r.mean() / base if base != 0 else float("nan")
            print(f"{name:16s}  relative-to-Greedy = {rel:6.1f}%")

    # Save raw rewards for reproducible figures / tables.
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"eval_{args.scenario}_cost{args.switching_cost}.csv")
    with open(out, "w") as f:
        f.write("policy,seed,reward\n")
        for name, r in results.items():
            for s, val in zip(seeds, r):
                f.write(f"{name},{s},{val}\n")
    print(f"\nSaved raw rewards to {out}")


if __name__ == "__main__":
    main()
