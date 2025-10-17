# File: evaluate.py
# Version 2.0 - Evaluates both scenarios and generates a final comparison plot.
# Implemented carefully with high-quality plotting.

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv, MIN_ELEVATION_DEGREES

def get_greedy_action(obs: np.ndarray, num_ogs: int, dynamic_weather: bool) -> int:
    """
    Selects an action based on the Greedy policy.
    In the dynamic scenario, it must also check for cloud cover.
    """
    best_action = num_ogs  # Default to "idle"
    max_elevation = -1.0
    
    # Base observation dimension before cloud info
    base_obs_dim = 4 + 3 * num_ogs

    for i in range(num_ogs):
        # In the dynamic scenario, first check for clouds
        if dynamic_weather:
            # Cloud status starts at index base_obs_dim
            cloud_status = obs[base_obs_dim + i]
            if cloud_status > 0:
                continue # Skip this OGS if it's cloudy
        
        # OGS elevation is at index 4 + 3*i
        elevation_normalized = obs[4 + 3 * i]
        
        if elevation_normalized > (MIN_ELEVATION_DEGREES / 90.0) and elevation_normalized > max_elevation:
            max_elevation = elevation_normalized
            best_action = i
            
    return best_action

def evaluate_agent(env: gym.Env, model=None, policy_type="drl") -> float:
    """
    Runs a full episode and returns the total cumulative reward.
    Uses a fixed seed for fair comparison.
    """
    # Use a consistent seed for all evaluations to ensure comparability
    obs, info = env.reset(seed=123)
    terminated = False
    total_reward = 0.0

    while not terminated:
        if policy_type == "drl":
            action, _ = model.predict(obs, deterministic=True)
        elif policy_type == "random":
            action = env.action_space.sample()
        elif policy_type == "greedy":
            action = get_greedy_action(obs, env.num_ogs, env.dynamic_weather)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break
            
    return total_reward

def run_evaluation_for_scenario(dynamic_weather: bool, model_name_suffix: str) -> dict:
    """
    Evaluates all three policies (DRL, Greedy, Random) for a single scenario.
    """
    scenario_name = "Dynamic" if dynamic_weather else "Static"
    print(f"\n--- Running Evaluation for {scenario_name} Scenario ---")
    
    model_path = os.path.join("models", model_name_suffix, "final_model.zip")
    num_ogs = 5

    # Create the correct environment for the scenario
    env = SatelliteQKDEnv(num_ogs=num_ogs, dynamic_weather=dynamic_weather)
    
    # --- 1. DRL Agent Evaluation ---
    drl_reward = 0.0
    if os.path.exists(model_path):
        try:
            model = PPO.load(model_path, device='cpu') # Load to CPU for evaluation
            drl_reward = evaluate_agent(env, model, policy_type="drl")
        except Exception as e:
            print(f"Could not load or run DRL model: {e}")
    else:
        print(f"WARNING: Model not found at {model_path}. DRL reward will be 0.")
        
    # --- 2. Greedy Agent Evaluation ---
    greedy_reward = evaluate_agent(env, policy_type="greedy")

    # --- 3. Random Agent Evaluation ---
    random_reward = evaluate_agent(env, policy_type="random")
    
    print(f"Results for {scenario_name} Scenario:")
    print(f"  DRL Agent:    {drl_reward:,.2f}")
    print(f"  Greedy Agent: {greedy_reward:,.2f}")
    print(f"  Random Agent: {random_reward:,.2f}")
    
    env.close()
    
    return {"DRL": drl_reward, "Greedy": greedy_reward, "Random": random_reward}

def plot_results(results_static: dict, results_dynamic: dict):
    """
    Generates a high-quality, publication-ready bar chart comparing all results.
    """
    print("\n--- Generating High-Quality Plot ---")
    
    # --- Plotting Style Setup ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 12, 'legend.fontsize': 14,
        'figure.titlesize': 20, 'font.family': 'serif'
    })
    color_palette = sns.color_palette("colorblind", 3)

    # --- Data Preparation ---
    # Convert results dictionaries to a Pandas DataFrame for easier plotting
    data = []
    for policy, reward in results_static.items():
        data.append({"Scenario": "Static Environment", "Policy": policy, "Reward": reward})
    for policy, reward in results_dynamic.items():
        data.append({"Scenario": "Dynamic Environment", "Policy": policy, "Reward": reward})
    df = pd.DataFrame(data)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use seaborn's barplot for grouped bars
    sns.barplot(
        x="Scenario", y="Reward", hue="Policy",
        data=df,
        palette=color_palette,
        edgecolor='black',
        linewidth=1.2,
        ax=ax
    )
    
    # --- Aesthetics and Labels ---
    ax.set_title("Performance Comparison Across Scenarios", pad=20)
    ax.set_ylabel("Total Secure Key Generated (bits)")
    ax.set_xlabel("") # Scenarios are self-explanatory
    
    # Format y-axis to be more readable
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ','))
    )
    
    # Improve legend
    ax.legend(title='Policy', frameon=True, facecolor='white', framealpha=0.8)
    
    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', padding=3, fontsize=10)
        
    fig.tight_layout()
    
    # --- Saving ---
    output_filename = "policy_comparison_dual_scenario.png"
    fig.savefig(output_filename, dpi=600)
    print(f"Saved high-DPI comparison chart to {output_filename}")
    plt.close(fig)


def main():
    """
    Main function to run the entire evaluation pipeline.
    """
    results_static = run_evaluation_for_scenario(
        dynamic_weather=False, model_name_suffix="static"
    )
    results_dynamic = run_evaluation_for_scenario(
        dynamic_weather=True, model_name_suffix="dynamic"
    )
    
    plot_results(results_static, results_dynamic)
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
