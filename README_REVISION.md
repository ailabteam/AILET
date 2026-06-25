# Sat-QKD DRL — Revision Experiments

New code added for the AILET resubmission, addressing the reviewer comments. The
original training/eval scripts are unchanged; everything here is additive.

## What is new and why

| File | Purpose | Reviewer point |
|------|---------|----------------|
| `qkd_env.py` (edited) | `switching_cost_minutes` is now a constructor argument | enables the cost ablation |
| `policies.py` | unified policy interface; adds **Hybrid** (greedy + switching hysteresis) | R1-4 (hybrid baseline) |
| `evaluate_revision.py` | **multi-seed** evaluation, mean ± std, fair shared-reward comparison | R2-1/2 (fair metric), rigor |
| `ablation_switching_cost.py` | sweeps the switching-cost magnitude | R1-2, R2-1/2 (sensitivity) |
| `tune_hybrid_margin.py` | sweeps the hybrid hysteresis margin (so it is not a strawman) | R1-4 |
| `train_ablation.py` | trains one PPO agent per switching cost (GPU-heavy) | R1-2 (effect on training) |
| `smoke_test_revision.py` | no-GPU correctness checks | — |

**Honest note on the expected outcome.** In this single-LEO / 5-OGS setting most
link switches are *forced by orbital geometry* (the held OGS sets below the
horizon), not discretionary. The cost-aware Hybrid therefore does **not** reliably
beat plain Greedy; and beyond a switching cost of roughly one pass duration (~6-8
min) every policy collapses to zero. These are reported as findings, not hidden.
Also, multi-seed evaluation will shift the original single-seed headline numbers
(Greedy ~4334 was a single seed); report the new mean ± std honestly.

## Setup (conda + tmux on the server)

```bash
conda env create -f environment.yml
conda activate qkd-revision
# GPU server: install the CUDA torch build to match your driver
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python verify_gpu.py        # confirm CUDA is visible
```

## Run order

```bash
# 0. correctness (seconds, no GPU)
python smoke_test_revision.py

# 1. multi-seed headline re-evaluation with the EXISTING 3M model (minutes, CPU ok)
python evaluate_revision.py --scenario realistic --seeds 20
python evaluate_revision.py --scenario static    --model "" --seeds 20
python evaluate_revision.py --scenario dynamic   --model "" --seeds 20

# 2. hybrid margin operating point (minutes, CPU ok)
python tune_hybrid_margin.py --switching-cost 2 --seeds 20

# 3a. quick ablation with heuristics + the fixed 3M model (minutes, CPU ok)
tmux new -s ablation
python ablation_switching_cost.py --seeds 20

# 3b. RIGOROUS ablation: retrain one PPO per cost, then re-run 3a (GPU, hours)
tmux new -s train
python train_ablation.py --costs 0 1 4 6 --timesteps 1000000
#   then re-run:
python ablation_switching_cost.py --seeds 20
```

Detach tmux with `Ctrl-b d`, reattach with `tmux attach -t train`.

## Outputs

- `results/eval_<scenario>_cost<C>.csv` — per-seed rewards for tables.
- `results/ablation_switching_cost.csv` + `.pdf` — the sensitivity figure.
- `results/hybrid_margin_cost<C>.csv` — margin sensitivity.

All policies are scored by the identical `env.step` reward (total secure key bits
minus operational penalties), so the comparison is auditable.
