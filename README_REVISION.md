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
beat plain Greedy; and beyond a switching cost of roughly one pass duration (~5
min) every policy collapses to zero. These are reported as findings, not hidden.
Also, multi-seed evaluation shifts the original single-seed headline numbers
(Greedy ~4334 was a single seed); report the new mean ± std honestly.

**Corrected switching-cost dynamics.** `qkd_env.step` previously decremented the
setup timer before computing the reward, so a nominal C-minute switching cost only
forfeited C-1 minutes (cost=1 was identical to cost=0). This is fixed: a C-minute
cost now forfeits exactly C minutes. All models must therefore be retrained under
the corrected env via `train_revision.py` (the released static/dynamic models are
also stale: they used an older 25-dim observation and no longer load).

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

# 1. RETRAIN everything under the corrected env (GPU, several hours) — run in tmux.
#    Trains static, dynamic, the cost=2 headline (3M), and per-cost ablation models.
tmux new -s train
python train_revision.py            # adjust --headline-steps / --ablation-steps to budget
#    (detach: Ctrl-b d ; reattach: tmux attach -t train ; watch GPU: watch -n2 nvidia-smi)

# 2. multi-seed evaluation of the headline scenarios (minutes, CPU ok)
python evaluate_revision.py --scenario realistic --seeds 20
python evaluate_revision.py --scenario static    --seeds 20
python evaluate_revision.py --scenario dynamic   --seeds 20

# 3. hybrid margin operating point (minutes, CPU ok)
python tune_hybrid_margin.py --switching-cost 2 --seeds 20

# 4. ablation across switching cost, now with per-cost-optimal DRL (minutes, CPU ok)
python ablation_switching_cost.py --seeds 20

# 5. learning curve for the headline model (Fig 2a)
python plot_checkpoint_performance.py
```

If GPU time is tight, lower `--headline-steps` (e.g. 1000000) and `--ablation-steps`
(e.g. 500000); the qualitative conclusions are unchanged.

## Outputs

- `results/eval_<scenario>_cost<C>.csv` — per-seed rewards for tables.
- `results/ablation_switching_cost.csv` + `.pdf` — the sensitivity figure.
- `results/hybrid_margin_cost<C>.csv` — margin sensitivity.

All policies are scored by the identical `env.step` reward (total secure key bits
minus operational penalties), so the comparison is auditable.
