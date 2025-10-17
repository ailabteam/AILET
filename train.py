# File: train.py
# Version 2.0 - Handles training for both static and dynamic scenarios.
# Implemented carefully to separate models and logs.

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from qkd_env import SatelliteQKDEnv

def train_scenario(dynamic_weather: bool, model_name_suffix: str, total_timesteps: int):
    """
    A helper function to encapsulate the training process for a single scenario.

    :param dynamic_weather: Boolean flag to set the environment mode.
    :param model_name_suffix: String to append to folder names (e.g., "static" or "dynamic").
    :param total_timesteps: The total number of steps to train the agent.
    """
    scenario_name = "Dynamic" if dynamic_weather else "Static"
    print(f"\n{'='*50}")
    print(f"--- Starting Training for {scenario_name} Scenario ---")
    print(f"{'='*50}")

    # --- 1. Define Paths ---
    # Create separate directories for logs and models for each scenario
    log_dir = os.path.join("logs", model_name_suffix)
    model_dir = os.path.join("models", model_name_suffix)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # --- 2. Create and Wrap the Environment ---
    print(f"Creating SatelliteQKDEnv with dynamic_weather={dynamic_weather}")
    # Instantiate the custom environment
    env = SatelliteQKDEnv(num_ogs=5, dynamic_weather=dynamic_weather)
    
    # It's good practice to wrap the environment with Monitor and DummyVecEnv
    # Monitor keeps track of episode statistics (reward, length)
    env = Monitor(env)
    # DummyVecEnv is a wrapper for single, non-vectorized environments
    env = DummyVecEnv([lambda: env])
    
    # --- 3. Instantiate the PPO Model ---
    # Check for available GPU device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # PPO is a robust on-policy algorithm suitable for this problem.
    # We use "MlpPolicy" as our observation space is a flat vector.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,            # Print training progress to the console
        tensorboard_log=log_dir, # Save logs for TensorBoard
        device=device,
        # Hyperparameters can be tuned, but defaults are often a good start
        # n_steps=2048,
        # batch_size=64,
        # n_epochs=10,
        # gamma=0.99,
    )

    # --- 4. Setup Callbacks ---
    # The CheckpointCallback saves the model periodically during training.
    # This is useful for long training runs to avoid losing progress.
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000, # Save a checkpoint every 20,000 steps
        save_path=model_dir,
        name_prefix="ppo_qkd_checkpoint"
    )

    # --- 5. Start Training ---
    print(f"\nTraining for {total_timesteps} timesteps...")
    # The learn() method starts the training loop
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        # Provide a name for the TensorBoard run
        tb_log_name="PPO"
    )
    print(f"--- Training Finished for {scenario_name} Scenario ---")

    # --- 6. Save the Final Model ---
    # After training is complete, save the final version of the model.
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model for {scenario_name} scenario saved to {final_model_path}")
    
    # Close the environment
    env.close()

def main():
    """
    Main function to run the entire training pipeline for both scenarios.
    """
    # --- Scenario 1: Static Environment ---
    # This agent learns in a predictable world without long-term disruptions.
    # 100,000 timesteps is a reasonable starting point.
    train_scenario(
        dynamic_weather=False, 
        model_name_suffix="static", 
        total_timesteps=100_000
    )
    
    # --- Scenario 2: Dynamic Environment ---
    # This agent must learn a more complex policy to handle unpredictable,
    # long-lasting cloud cover. It requires more training to learn these
    # long-term dependencies. We use 200,000 timesteps.
    train_scenario(
        dynamic_weather=True, 
        model_name_suffix="dynamic", 
        total_timesteps=200_000
    )
    print("\nAll training sessions complete.")

if __name__ == '__main__':
    main()
