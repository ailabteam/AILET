# File: evaluate.py
# Version 1.1 - Professional, High-DPI Plotting

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn
import os
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv, MIN_ELEVATION_DEGREES

# --- Helper Function for Greedy Agent ---
def get_greedy_action(obs, num_ogs):
    best_action = num_ogs
    max_elevation = -1.0
    for i in range(num_ogs):
        elevation = obs[4 + 3 * i] * 90.0
        if elevation > MIN_ELEVATION_DEGREES and elevation > max_elevation:
            max_elevation = elevation
            best_action = i
    return best_action

# --- Main Evaluation Function ---
def evaluate_agent(env, model=None, policy_type="drl"):
    obs, info = env.reset(seed=123)
    terminated = False
    total_reward = 0
    rewards_history = []
    actions_history = []

    while not terminated:
        if policy_type == "drl":
            action, _states = model.predict(obs, deterministic=True)
        elif policy_type == "random":
            action = env.action_space.sample()
        elif policy_type == "greedy":
            action = get_greedy_action(obs, env.num_ogs)
        else:
            raise ValueError("Unknown policy type")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards_history.append(reward)
        actions_history.append(action)
        if terminated or truncated:
            break
    return total_reward, rewards_history, actions_history

def main():
    MODEL_DIR = "models"
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_qkd_model_final.zip")
    NUM_OGS = 5

    # --- 1. Create the Environment ---
    print("Creating the environment for evaluation...")
    env = SatelliteQKDEnv(num_ogs=NUM_OGS)

    # --- 2. Evaluate DRL Agent ---
    print("\n--- Evaluating DRL Agent ---")
    drl_reward, drl_rewards_hist, drl_actions_hist = (None, None, None)
    if os.path.exists(FINAL_MODEL_PATH):
        model = PPO.load(FINAL_MODEL_PATH)
        drl_reward, drl_rewards_hist, drl_actions_hist = evaluate_agent(env, model, policy_type="drl")
        print(f"DRL Agent Total Reward: {drl_reward:.2f}")
    else:
        print(f"Model not found at {FINAL_MODEL_PATH}. Please train the model first.")

    # --- 3. Evaluate Random Agent ---
    print("\n--- Evaluating Random Agent ---")
    random_reward, random_rewards_hist, random_actions_hist = evaluate_agent(env, policy_type="random")
    print(f"Random Agent Total Reward: {random_reward:.2f}")

    # --- 4. Evaluate Greedy Agent ---
    print("\n--- Evaluating Greedy Agent ---")
    greedy_reward, greedy_rewards_hist, greedy_actions_hist = evaluate_agent(env, policy_type="greedy")
    print(f"Greedy Agent Total Reward: {greedy_reward:.2f}")
    
    env.close()

    # --- 5. Plotting Results ---
    print("\n--- Generating High-Quality Plots ---")

    # === PLOTTING STYLE SETUP ===
    sns.set_theme(style="whitegrid") # Set a nice theme
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 20
    # Use a colorblind-friendly palette
    color_palette = sns.color_palette("colorblind", 3)

    # === Bar chart for total rewards ===
    policies = []
    rewards = []
    if drl_reward is not None:
        policies.append("DRL Agent")
        rewards.append(drl_reward)
    policies.extend(["Greedy Agent", "Random Agent"])
    rewards.extend([greedy_reward, random_reward])
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bars = ax1.bar(policies, rewards, color=color_palette, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel("Total Secure Key Generated (bits)")
    ax1.set_title("Comparison of Scheduling Policies", pad=20)
    
    # Add values on top of bars
    ax1.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=12)
    
    fig1.tight_layout()
    fig1.savefig("policy_comparison.png", dpi=600)
    print("Saved high-DPI policy comparison chart to policy_comparison.png")
    plt.close(fig1)

    # === Action distribution histogram ===
    if drl_reward is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        actions_data = [drl_actions_hist, greedy_actions_hist, random_actions_hist]
        labels = ['DRL', 'Greedy', 'Random']
        
        # Use seaborn for a nicer histogram
        sns.histplot(data=actions_data, multiple="dodge", shrink=0.8,
                     bins=np.arange(NUM_OGS + 2) - 0.5,
                     ax=ax2, palette=color_palette)

        ax2.set_xticks(range(NUM_OGS + 1))
        ax2.set_xticklabels([f'OGS {i}' for i in range(NUM_OGS)] + ['Idle'])
        ax2.set_title('Action Distribution Comparison', pad=20)
        ax2.set_xlabel('Action')
        ax2.set_ylabel('Action Count')
        
        # Manually create legend as seaborn's default might not be perfect here
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_palette[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
        ax2.legend(handles=legend_elements)

        fig2.tight_layout()
        fig2.savefig("action_distribution.png", dpi=600)
        print("Saved high-DPI action distribution chart to action_distribution.png")
        plt.close(fig2)


if __name__ == '__main__':
    main()
