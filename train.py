# File: train.py
# Version 3.0 - Handles training for all three scenarios (static, dynamic, realistic).
# Final, carefully implemented version.

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qkd_env import SatelliteQKDEnv

def train_scenario(scenario: str, total_timesteps: int):
    """
    A helper function to encapsulate the training process for a single scenario.

    :param scenario: The environment scenario ("static", "dynamic", or "realistic").
    :param total_timesteps: The total number of steps to train the agent.
    """
    print(f"\n{'='*60}")
    print(f"--- Starting Training for {scenario.upper()} Scenario ---")
    print(f"{'='*60}")

    # --- 1. Define Paths ---
    # Create separate directories for logs and models for each scenario
    log_dir = os.path.join("logs", scenario)
    model_dir = os.path.join("models", scenario)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # --- 2. Create and Wrap the Environment ---
    print(f"Creating SatelliteQKDEnv with scenario='{scenario}'")
    # Instantiate the custom environment
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
    
    # Wrap the environment with Monitor and DummyVecEnv for compatibility
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # --- 3. Instantiate the PPO Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # PPO is a robust on-policy algorithm suitable for this problem.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        # Hyperparameters can be adjusted, but SB3 defaults are strong.
        # Example: Increase n_steps for more data per update
        # n_steps=4096 
    )

    # --- 4. Setup Callbacks ---
    # Saves a checkpoint of the model periodically.
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000, # Adjust frequency based on total_timesteps
        save_path=model_dir,
        name_prefix=f"ppo_qkd_{scenario}"
    )

    # --- 5. Start Training ---
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO"
    )
    print(f"--- Training Finished for {scenario.upper()} Scenario ---")

    # --- 6. Save the Final Model ---
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model for {scenario.upper()} scenario saved to {final_model_path}")
    
    env.close()

def main():
    """
    Main function to run the entire training pipeline for all three scenarios.
    """
    # Define the training plan
    training_plan = {
        #"static": 100_000,
        #"dynamic": 200_000,
        "realistic": 3_000_000,
    }

    for scenario_name, timesteps in training_plan.items():
        train_scenario(scenario=scenario_name, total_timesteps=timesteps)
    
    print("\n\n" + "="*60)
    print("      All training sessions have been completed.      ")
    print("="*60)


if __name__ == '__main__':
    main()
