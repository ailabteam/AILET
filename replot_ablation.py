# File: replot_ablation.py
# Regenerate the switching-cost ablation figure from results/ablation_switching_cost.csv.
# It consolidates any per-cost DRL labels (e.g. "DRL(c=0)", "DRL(fixed c=2)") into a
# single "DRL" series so the agent forms one connected line instead of disconnected
# points. Pure post-processing; no model evaluation, so it runs in seconds.

import os
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def canon(name):
    return "DRL" if name.startswith("DRL") else name


def main():
    path = os.path.join("results", "ablation_switching_cost.csv")
    rows = list(csv.DictReader(open(path)))
    costs = sorted({int(r["switching_cost_min"]) for r in rows})

    # series[name][cost] = (mean, std); DRL labels are merged.
    series = defaultdict(dict)
    for r in rows:
        c = int(r["switching_cost_min"])
        series[canon(r["policy"])][c] = (float(r["mean_reward"]), float(r["std_reward"]))

    # Order legend sensibly.
    order = [n for n in ["DRL", "Greedy", "Hybrid(m=15)", "Random"] if n in series]
    order += [n for n in series if n not in order]

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name in order:
        xs = [c for c in costs if c in series[name]]
        ys = [series[name][c][0] for c in xs]
        es = [series[name][c][1] for c in xs]
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
    # Print collapse threshold (first cost where Greedy mean rounds to 0).
    for c in costs:
        if "Greedy" in series and c in series["Greedy"] and round(series["Greedy"][c][0]) == 0:
            print(f"Collapse threshold (Greedy ~ 0) at C = {c} min")
            break


if __name__ == "__main__":
    main()
