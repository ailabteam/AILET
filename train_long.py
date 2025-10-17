# File: train_long.py
# A dedicated script for the long training run of the 'realistic' scenario.

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qkd_env import SatelliteQKDEnv

def main():
    """
    Main function to run a long training session for the realistic scenario.
    """
    # --- Configuration ---
    # The environment uses the 'realistic' configuration
    env_scenario_name = "realistic"
    
    # We save the model and logs under a new name to avoid overwriting previous results
    model_name_suffix = "realistic_3M"
    
    # The total number of timesteps for this long run
    total_timesteps = 3_000_000

    print(f"\n{'='*60}")
    print(f"--- Starting LONG Training for '{model_name_suffix.upper()}' ---")
    print(f"    Environment: '{env_scenario_name.upper()}'")
    print(f"    Total Timesteps: {total_timesteps:,}")
    print(f"{'='*60}")

    # --- 1. Define Paths ---
    log_dir = os.path.join("logs", model_name_suffix)
    model_dir = os.path.join("models", model_name_suffix)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # --- 2. Create and Wrap the Environment ---
    print(f"Creating SatelliteQKDEnv with scenario='{env_scenario_name}'")
    env = SatelliteQKDEnv(num_ogs=5, scenario=env_scenario_name)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # --- 3. Instantiate the PPO Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )

    # --- 4. Setup Callbacks ---
    # Save checkpoints more frequently in a long run
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=model_dir,
        name_prefix=f"ppo_qkd_{model_name_suffix}"
    )

    # --- 5. Start Training ---
    print(f"\nTraining started. This will take a significant amount of time...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO_3M_Run" # A specific name for TensorBoard
    )
    print(f"--- Long Training Finished for {model_name_suffix.upper()} ---")

    # --- 6. Save the Final Model ---
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model for {model_name_suffix.upper()} saved to {final_model_path}")
    
    env.close()

if __name__ == '__main__':
    main()
