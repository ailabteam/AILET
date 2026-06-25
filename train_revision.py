# File: train_revision.py
# One-shot training orchestration for the revised study, run under the CORRECTED
# environment (switching-cost off-by-one fixed). Trains, in sequence:
#
#   1. static   -> models/static/final_model.zip          (Table 1 escalation)
#   2. dynamic  -> models/dynamic/final_model.zip          (Table 1 escalation)
#   3. realistic headline (cost=2, long run)
#               -> models/realistic_3M/final_model.zip     (headline DRL + learning curve)
#   4. realistic per-cost ablation
#               -> models/ablation_cost{C}/final_model.zip (per-cost DRL line)
#
# Why retrain everything: the released static/dynamic models were saved with a
# 25-dim observation (an older env) and no longer match the current variable-dim
# env; and the off-by-one fix changes the realistic dynamics. Retraining makes every
# model consistent with the corrected env.
#
# Launch under tmux. Adjust step counts to your GPU budget.

import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qkd_env import SatelliteQKDEnv


def train(scenario, total_timesteps, model_dir, switching_cost_minutes=2,
          seed=0, ckpt_freq=200_000):
    log_dir = os.path.join("logs", os.path.basename(model_dir))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n{'='*64}\n--- TRAIN {model_dir}  scenario={scenario}  "
          f"cost={switching_cost_minutes}  steps={total_timesteps:,}  seed={seed} ---\n{'='*64}")

    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario,
                          switching_cost_minutes=switching_cost_minutes)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO("MlpPolicy", env, verbose=1, seed=seed,
                tensorboard_log=log_dir, device=device)
    ckpt = CheckpointCallback(save_freq=ckpt_freq, save_path=model_dir,
                              name_prefix=f"ppo_{os.path.basename(model_dir)}")
    model.learn(total_timesteps=total_timesteps, callback=ckpt,
                tb_log_name="PPO")
    final = os.path.join(model_dir, "final_model.zip")
    model.save(final)
    print(f"Saved {final}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-steps", type=int, default=300_000)
    ap.add_argument("--dynamic-steps", type=int, default=300_000)
    ap.add_argument("--headline-steps", type=int, default=3_000_000)
    ap.add_argument("--ablation-steps", type=int, default=1_000_000)
    ap.add_argument("--ablation-costs", type=int, nargs="+", default=[0, 1, 3, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="phases to skip: static dynamic headline ablation")
    args = ap.parse_args()

    if "static" not in args.skip:
        train("static", args.static_steps, "models/static", seed=args.seed)
    if "dynamic" not in args.skip:
        train("dynamic", args.dynamic_steps, "models/dynamic", seed=args.seed)
    if "headline" not in args.skip:
        # cost=2 long run; doubles as the headline realistic model.
        train("realistic", args.headline_steps, "models/realistic_3M",
              switching_cost_minutes=2, seed=args.seed)
    if "ablation" not in args.skip:
        for c in args.ablation_costs:
            train("realistic", args.ablation_steps, f"models/ablation_cost{c}",
                  switching_cost_minutes=c, seed=args.seed)

    print("\nAll requested training phases complete.")


if __name__ == "__main__":
    main()
