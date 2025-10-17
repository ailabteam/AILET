# File: evaluate_final.py
# Compares the original 3 agents with the new agent trained for 3M steps.

import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv, MIN_ELEVATION_DEGREES

# --- get_greedy_action and evaluate_agent functions are the same as evaluate.py v3.0 ---
def get_greedy_action(obs: np.ndarray, env: SatelliteQKDEnv) -> int:
    best_action = env.num_ogs
    max_elevation = -1.0
    base_obs_dim = 4 + 3 * env.num_ogs
    for i in range(env.num_ogs):
        if env.is_dynamic_weather:
            if obs[base_obs_dim + i] > 0: continue
        elevation_normalized = obs[4 + 3 * i]
        if elevation_normalized * 90.0 < MIN_ELEVATION_DEGREES: continue
        if elevation_normalized > max_elevation:
            max_elevation = elevation_normalized
            best_action = i
    return best_action

def evaluate_agent(env: gym.Env, model=None, policy_type="drl") -> float:
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
        if terminated or truncated: break
    return total_reward
# --- End of copied functions ---

def run_evaluation(scenario: str, model_suffix: str) -> dict:
    """A more generic evaluation function."""
    print(f"\n--- Evaluating '{model_suffix.upper()}' on '{scenario.upper()}' environment ---")
    model_path = os.path.join("models", model_suffix, "final_model.zip")
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
    
    drl_reward = 0.0
    if os.path.exists(model_path):
        model = PPO.load(model_path, device='cpu')
        drl_reward = evaluate_agent(env, model, policy_type="drl")
    else:
        print(f"WARNING: Model for '{model_suffix}' not found.")
        
    greedy_reward = evaluate_agent(env, policy_type="greedy")
    random_reward = evaluate_agent(env, policy_type="random")
    
    print(f"Results: DRL={drl_reward:,.2f}, Greedy={greedy_reward:,.2f}, Random={random_reward:,.2f}")
    env.close()
    return {"DRL": drl_reward, "Greedy": greedy_reward, "Random": random_reward}

def main():
    """Main function to run all evaluations and plot."""
    all_results = {}
    
    # 1. Evaluate the original models on their respective environments
    all_results['Static'] = run_evaluation(scenario='static', model_suffix='static')
    all_results['Dynamic'] = run_evaluation(scenario='dynamic', model_suffix='dynamic')
    
    # 2. Evaluate original realistic model vs the new one
    print("\n--- Comparing original 'Realistic' model with the new 'Realistic 3M' model ---")
    original_realistic_results = run_evaluation(scenario='realistic', model_suffix='realistic')
    long_train_realistic_results = run_evaluation(scenario='realistic', model_suffix='realistic_3M')
    
    # Let's create a DataFrame for plotting
    data = []
    
    # Scenario 1 data
    for policy, reward in all_results['Static'].items():
        data.append({"Scenario": "Static", "Policy": policy, "Reward": reward})
    # Scenario 2 data
    for policy, reward in all_results['Dynamic'].items():
        data.append({"Scenario": "Dynamic", "Policy": policy, "Reward": reward})
    # Scenario 3: Compare the two DRL agents and the baselines
    data.append({"Scenario": "Realistic", "Policy": "DRL (300k steps)", "Reward": original_realistic_results["DRL"]})
    data.append({"Scenario": "Realistic", "Policy": "DRL (3M steps)", "Reward": long_train_realistic_results["DRL"]})
    data.append({"Scenario": "Realistic", "Policy": "Greedy", "Reward": long_train_realistic_results["Greedy"]})
    data.append({"Scenario": "Realistic", "Policy": "Random", "Reward": long_train_realistic_results["Random"]})

    df = pd.DataFrame(data)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'font.family': 'serif'})

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x="Scenario", y="Reward", hue="Policy", data=df,
                palette="viridis", edgecolor='black', linewidth=1.0, ax=ax)

    ax.set_title("Policy Performance Comparison Across All Scenarios", pad=20)
    ax.set_ylabel("Total Secure Key Generated (bits)")
    ax.set_xlabel("Environment Scenario")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0'))
    ax.legend(title='Policy')
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', padding=3, fontsize=9, rotation=90)
    fig.tight_layout()
    
    output_filename = "final_comparison_with_3M_model.png"
    fig.savefig(output_filename, dpi=600)
    print(f"\nSaved final comparison chart to {output_filename}")
    plt.close(fig)

    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
