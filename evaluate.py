# File: evaluate.py
# Version 3.0 - Evaluates all three scenarios and generates the final comparison plot.
# Final, carefully implemented version.

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv, MIN_ELEVATION_DEGREES

def get_greedy_action(obs: np.ndarray, env: SatelliteQKDEnv) -> int:
    """
    Selects an action based on the Greedy policy.
    This version is aware of the environment's scenario to act optimally
    under the given constraints (clouds, switching costs).
    """
    best_action = env.num_ogs  # Default to "idle"
    max_elevation = -1.0
    
    # The greedy agent's observation space is simpler than the DRL agent's
    base_obs_dim = 4 + 3 * env.num_ogs

    for i in range(env.num_ogs):
        # --- Conditional Logic for Complex Scenarios ---
        # 1. Avoid cloudy OGSs if the weather is dynamic
        if env.is_dynamic_weather:
            cloud_status = obs[base_obs_dim + i]
            if cloud_status > 0:
                continue  # Skip this OGS, it's cloudy
        
        # 2. Check for geometric viability
        elevation_normalized = obs[4 + 3 * i]
        if elevation_normalized * 90.0 < MIN_ELEVATION_DEGREES:
            continue # Skip this OGS, it's below the horizon
        
        # Among all valid OGSs, find the one with the highest elevation
        if elevation_normalized > max_elevation:
            max_elevation = elevation_normalized
            best_action = i
            
    # The greedy agent is "myopic", so it doesn't reason about switching costs.
    # It will always try to switch if it sees a better option, which is its flaw.
    return best_action

def evaluate_agent(env: gym.Env, model=None, policy_type="drl") -> float:
    """
    Runs a full episode and returns the total cumulative reward.
    Uses a fixed seed for fair comparison across all policies.
    """
    obs, info = env.reset(seed=123)
    terminated = False
    total_reward = 0.0

    while not terminated:
        if policy_type == "drl":
            action, _ = model.predict(obs, deterministic=True)
        elif policy_type == "random":
            action = env.action_space.sample()
        elif policy_type == "greedy":
            action = get_greedy_action(obs, env)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break
            
    return total_reward

def run_evaluation_for_scenario(scenario: str) -> dict:
    """
    Evaluates all three policies (DRL, Greedy, Random) for a single scenario.
    """
    print(f"\n--- Running Evaluation for {scenario.upper()} Scenario ---")
    
    model_path = os.path.join("models", scenario, "final_model.zip")
    
    # Create the correct environment for the evaluation
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
    
    # --- 1. DRL Agent Evaluation ---
    drl_reward = 0.0
    if os.path.exists(model_path):
        try:
            # Load the trained model to CPU for faster inference
            model = PPO.load(model_path, device='cpu')
            drl_reward = evaluate_agent(env, model, policy_type="drl")
        except Exception as e:
            print(f"Could not load or run DRL model for '{scenario}': {e}")
    else:
        print(f"WARNING: Model not found at {model_path}. DRL reward will be 0.")
        
    # --- 2. Greedy Agent Evaluation ---
    greedy_reward = evaluate_agent(env, policy_type="greedy")

    # --- 3. Random Agent Evaluation ---
    random_reward = evaluate_agent(env, policy_type="random")
    
    print(f"Results for {scenario.upper()} Scenario:")
    print(f"  DRL Agent:    {drl_reward:,.2f}")
    print(f"  Greedy Agent: {greedy_reward:,.2f}")
    print(f"  Random Agent: {random_reward:,.2f}")
    
    env.close()
    
    return {"DRL": drl_reward, "Greedy": greedy_reward, "Random": random_reward}

def plot_results(all_results: dict):
    """
    Generates a high-quality, publication-ready bar chart comparing all results.
    """
    print("\n--- Generating High-Quality Final Plot ---")
    
    # --- Plotting Style Setup ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 12, 'legend.fontsize': 14,
        'figure.titlesize': 20, 'font.family': 'serif'
    })
    color_palette = sns.color_palette("colorblind", 3)

    # --- Data Preparation ---
    # Convert the results dictionary to a Pandas DataFrame for easy plotting
    data = []
    for scenario, results in all_results.items():
        for policy, reward in results.items():
            data.append({"Scenario": scenario, "Policy": policy, "Reward": reward})
    df = pd.DataFrame(data)
    
    # Capitalize scenario names for the plot
    df['Scenario'] = df['Scenario'].str.replace('_', ' ').str.title()
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(x="Scenario", y="Reward", hue="Policy", data=df,
                palette=color_palette, edgecolor='black', linewidth=1.2, ax=ax)
    
    # --- Aesthetics and Labels ---
    ax.set_title("Policy Performance Comparison Across Simulation Scenarios", pad=20)
    ax.set_ylabel("Total Secure Key Generated (bits)")
    ax.set_xlabel("Environment Scenario")
    
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--') # Add a zero line
    
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0') # Format as '1,234k'
    )
    
    ax.legend(title='Policy', frameon=True, facecolor='white', framealpha=0.8)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', padding=3, fontsize=10, rotation=90)
        
    fig.tight_layout()
    
    # --- Saving ---
    output_filename = "final_comparison_plot.png"
    fig.savefig(output_filename, dpi=600)
    print(f"Saved final comparison chart to {output_filename}")
    plt.close(fig)

def main():
    """
    Main function to run the entire evaluation pipeline for all three scenarios.
    """
    all_results = {}
    
    scenarios_to_run = ["static", "dynamic", "realistic"]
    for scenario in scenarios_to_run:
        all_results[scenario] = run_evaluation_for_scenario(scenario=scenario)
    
    plot_results(all_results)
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
