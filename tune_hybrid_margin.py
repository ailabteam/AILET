# File: tune_hybrid_margin.py
# Sweeps the Hybrid policy's hysteresis margin so the cost-aware baseline is
# reported at a properly chosen operating point rather than as a strawman
# (supports the Reviewer 1, point 4 response). Heuristic-only, no GPU.

import os
import argparse
import numpy as np
from evaluate_revision import evaluate_policy
from policies import HybridPolicy, GreedyPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", type=float, nargs="+",
                    default=[0, 5, 10, 15, 20, 30, 45])
    ap.add_argument("--switching-cost", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=1000)
    args = ap.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    g = evaluate_policy("realistic", GreedyPolicy(), seeds, args.switching_cost)
    print(f"Greedy baseline: mean={g.mean():,.1f}  (cost={args.switching_cost} min)")
    print("-" * 50)
    rows = []
    for m in args.margins:
        r = evaluate_policy("realistic", HybridPolicy(margin_deg=m), seeds,
                            args.switching_cost)
        rows.append((m, float(r.mean()), float(r.std())))
        delta = 100.0 * (r.mean() - g.mean()) / g.mean() if g.mean() else float("nan")
        print(f"margin={m:5.1f} deg  mean={r.mean():12,.1f}  std={r.std():9,.1f}  "
              f"vs Greedy={delta:+5.1f}%")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"hybrid_margin_cost{args.switching_cost}.csv")
    with open(out, "w") as f:
        f.write("margin_deg,mean_reward,std_reward\n")
        for m, mean, std in rows:
            f.write(f"{m},{mean},{std}\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
