# File: plot_checkpoint_performance.py
# Evaluates model checkpoints at different stages of training to show performance progression.

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv

# --- You can copy these helper functions from evaluate_final.py ---
def get_greedy_action(obs: np.ndarray, env: SatelliteQKDEnv) -> int:
    best_action = env.num_ogs; max_elevation = -1.0
    base_obs_dim = 4 + 3 * env.num_ogs
    for i in range(env.num_ogs):
        if env.is_dynamic_weather and obs[base_obs_dim + i] > 0: continue
        elevation_normalized = obs[4 + 3 * i]
        # Use the raw value from the env constant for accuracy
        if elevation_normalized * 90.0 < env.metadata.get('min_elevation', 20.0): continue
        if elevation_normalized > max_elevation: max_elevation = elevation_normalized; best_action = i
    return best_action

def evaluate_agent(env: gym.Env, model=None, policy_type="drl") -> float:
    obs, info = env.reset(seed=123); terminated = False; total_reward = 0.0
    while not terminated:
        if policy_type == "drl": action, _ = model.predict(obs, deterministic=True)
        elif policy_type == "random": action = env.action_space.sample()
        elif policy_type == "greedy": action = get_greedy_action(obs, env)
        else: raise ValueError(f"Unknown policy type: {policy_type}")
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated: break
    return total_reward
# --- End of copied functions ---

def main():
    print("--- Evaluating Checkpoint Performance Progression ---")
    
    model_dir = "models/realistic_3M"
    scenario_to_eval = "realistic"
    
    # --- 1. Define Checkpoints to Evaluate ---
    # We select key milestones from the training process.
    checkpoints = [
        300000, 600000, 1000000, 1500000, 2000000, 2500000, 3000000
    ]
    # Also add the original 300k model for comparison
    original_300k_model_path = "models/realistic/final_model.zip"

    training_steps = []
    drl_rewards = []

    # --- 2. Evaluate the original 300k model ---
    if os.path.exists(original_300k_model_path):
        print(f"\nEvaluating original model (300k steps)...")
        env = SatelliteQKDEnv(num_ogs=5, scenario=scenario_to_eval)
        model = PPO.load(original_300k_model_path, device='cpu')
        reward = evaluate_agent(env, model, policy_type="drl")
        print(f"  -> Reward: {reward:,.2f}")
        training_steps.append(300000)
        drl_rewards.append(reward)
        env.close()

    # --- 3. Loop Through and Evaluate Checkpoints ---
    for step in checkpoints:
        model_filename = f"ppo_qkd_realistic_3M_{step}_steps.zip"
        model_path = os.path.join(model_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"WARNING: Checkpoint not found at {model_path}. Skipping.")
            continue
            
        print(f"\nEvaluating checkpoint at {step:,} steps...")
        env = SatelliteQKDEnv(num_ogs=5, scenario=scenario_to_eval)
        model = PPO.load(model_path, device='cpu')
        
        reward = evaluate_agent(env, model, policy_type="drl")
        print(f"  -> Reward: {reward:,.2f}")
        
        training_steps.append(step)
        drl_rewards.append(reward)
        env.close()

    # --- 4. Evaluate Baselines for Context ---
    print("\nEvaluating baselines (Greedy and Random)...")
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario_to_eval)
    greedy_reward = evaluate_agent(env, policy_type="greedy")
    random_reward = evaluate_agent(env, policy_type="random")
    print(f"  -> Greedy Reward: {greedy_reward:,.2f}")
    print(f"  -> Random Reward: {random_reward:,.2f}")
    env.close()

    # --- 5. Generate the Plot ---
    print("\n--- Generating High-Quality Plot ---")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14, 'font.family': 'serif'})
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot DRL agent's performance progression
    ax.plot(training_steps, drl_rewards, marker='o', linestyle='-', color='royalblue', label='DRL Agent Performance', linewidth=2.5, markersize=8)

    # Plot baseline performances as horizontal lines
    ax.axhline(greedy_reward, color='darkorange', linestyle='--', linewidth=2, label=f'Greedy Agent ({greedy_reward:,.0f})')
    ax.axhline(random_reward, color='forestgreen', linestyle=':', linewidth=2, label=f'Random Agent ({random_reward:,.0f})')

    # Aesthetics
    ax.set_title("DRL Performance vs. Training Steps in Realistic Scenario", pad=20)
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Total Reward (Secure Key Bits)")
    ax.legend(frameon=True)
    
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0'))
    
    fig.tight_layout()
    
    output_filename = "checkpoint_performance_plot.png"
    fig.savefig(output_filename, dpi=600)
    print(f"Saved checkpoint performance plot to {output_filename}")
    plt.close(fig)

if __name__ == '__main__':
    main()
