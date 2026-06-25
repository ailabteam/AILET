# File: train_ablation.py
# Trains one PPO agent per switching-cost value so the ablation can report a DRL
# agent that was actually optimized for each cost (the rigorous version of
# Reviewer 1's request about how cost affects exploration / convergence / final
# performance).
#
# Models are saved to models/ablation_cost{C}/final_model.zip, which
# ablation_switching_cost.py picks up automatically.
#
# This is the GPU-heavy step. Defaults to 1M timesteps per cost (lighter than the
# 3M headline run) so the full sweep is feasible; raise --timesteps if you want to
# match the 3M setting for cost=2. Launch under tmux on the server.

import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qkd_env import SatelliteQKDEnv


def train_one(cost, timesteps, seed):
    suffix = f"ablation_cost{cost}"
    log_dir = os.path.join("logs", suffix)
    model_dir = os.path.join("models", suffix)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'='*60}\n--- Training PPO  switching_cost={cost} min  "
          f"steps={timesteps:,}  seed={seed} ---\n{'='*60}")

    env = SatelliteQKDEnv(num_ogs=5, scenario="realistic",
                          switching_cost_minutes=cost)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO("MlpPolicy", env, verbose=1, seed=seed,
                tensorboard_log=log_dir, device=device)

    ckpt = CheckpointCallback(save_freq=200_000, save_path=model_dir,
                              name_prefix=f"ppo_{suffix}")
    model.learn(total_timesteps=timesteps, callback=ckpt,
                tb_log_name=f"PPO_{suffix}")
    final_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_path)
    print(f"Saved {final_path}")
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", type=int, nargs="+", default=[0, 1, 4, 8, 16],
                    help="costs to train (cost=2 already exists as realistic_3M)")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for cost in args.costs:
        train_one(cost, args.timesteps, args.seed)


if __name__ == "__main__":
    main()
