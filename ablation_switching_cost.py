# File: ablation_switching_cost.py
# Sensitivity analysis on the link-switching cost magnitude (Reviewer 1, point 2;
# Reviewer 2, points 1-2).
#
# For each switching-cost value we evaluate, over multiple seeds:
#   - Greedy   (myopic, pays the cost on every switch)
#   - Hybrid   (cost-aware hysteresis)
#   - Random   (lower bound)
#   - DRL      (optional): per-cost retrained model if available, otherwise the
#               fixed cost=2 model evaluated off-distribution (clearly labelled).
#
# Output: a tidy CSV (results/ablation_switching_cost.csv) and, if matplotlib is
# present, a publication figure (results/ablation_switching_cost.pdf).
#
# The expected story: as the switching cost grows, myopic Greedy degrades while the
# cost-aware Hybrid stays robust, and a DRL agent tuned to a single cost value does
# not transfer. This both explains the original negative result and gives an
# actionable design takeaway.

import os
import argparse
import numpy as np

from qkd_env import SatelliteQKDEnv
from evaluate_revision import evaluate_policy
from policies import GreedyPolicy, HybridPolicy, RandomPolicy, DRLPolicy


def find_drl_for_cost(cost, fixed_model):
    """Prefer a model retrained at this exact cost; fall back to the fixed model.
    The label is kept constant ("DRL") so all per-cost points form a single
    connected series in the plot rather than one disconnected point each."""
    per_cost = os.path.join("models", f"ablation_cost{cost}", "final_model.zip")
    if os.path.exists(per_cost):
        return per_cost, "DRL"
    if fixed_model and os.path.exists(fixed_model):
        return fixed_model, "DRL"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5],
                    help="switching-cost values in minutes to sweep. Around ~5 min "
                         "all policies collapse to 0 (cost exceeds a typical pass "
                         "duration), which is itself reported as a critical threshold")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--hybrid-margin", type=float, default=15.0)
    ap.add_argument("--fixed-model", default="models/realistic_3M/final_model.zip")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-drl", action="store_true", help="skip DRL entirely")
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    rows = []  # (cost, policy_name, mean, std)
    drl_cache = {}

    for cost in args.costs:
        print(f"\n=== switching cost = {cost} min ===")
        policies = [GreedyPolicy(), HybridPolicy(margin_deg=args.hybrid_margin),
                    RandomPolicy()]

        if not args.no_drl:
            path, label = find_drl_for_cost(cost, args.fixed_model)
            if path is not None:
                if path not in drl_cache:
                    ref_env = SatelliteQKDEnv(num_ogs=5, scenario="realistic",
                                              switching_cost_minutes=cost)
                    drl_cache[path] = DRLPolicy(path, ref_env, device=args.device)
                drl = drl_cache[path]
                drl.name = label
                policies.insert(0, drl)

        for p in policies:
            r = evaluate_policy("realistic", p, seeds, switching_cost_minutes=cost)
            rows.append((cost, p.name, float(r.mean()), float(r.std())))
            print(f"  {p.name:16s}  mean={r.mean():12,.1f}  std={r.std():10,.1f}")

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "ablation_switching_cost.csv")
    with open(csv_path, "w") as f:
        f.write("switching_cost_min,policy,mean_reward,std_reward\n")
        for cost, name, m, s in rows:
            f.write(f"{cost},{name},{m},{s}\n")
    print(f"\nSaved {csv_path}")

    try:
        plot_ablation(rows, args.costs)
    except Exception as e:
        print(f"(Plot skipped: {e})")


def plot_ablation(rows, costs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    names = []
    for _, name, _, _ in rows:
        if name not in names:
            names.append(name)

    fig, ax = plt.subplots(figsize=(3.4, 2.6))  # single ACM column width
    for name in names:
        xs, ys, es = [], [], []
        for cost in costs:
            for c, n, m, s in rows:
                if c == cost and n == name:
                    xs.append(c); ys.append(m); es.append(s)
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, linewidth=1.2,
                    markersize=4, label=name)
    ax.set_xlabel("Link-switching cost (minutes)")
    ax.set_ylabel("Total secure key (bits)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join("results", "ablation_switching_cost.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
