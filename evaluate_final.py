#on 3.1 - Generates final plots AND a LaTeX table for the paper.
import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
# ... (các import khác giữ nguyên) ...
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from qkd_env import SatelliteQKDEnv, MIN_ELEVATION_DEGREES
import numpy as np

# --- get_greedy_action và evaluate_agent giữ nguyên như v3.0 ---
def get_greedy_action(obs: np.ndarray, env: SatelliteQKDEnv) -> int:
    best_action = env.num_ogs; max_elevation = -1.0
    base_obs_dim = 4 + 3 * env.num_ogs
    for i in range(env.num_ogs):
        if env.is_dynamic_weather and obs[base_obs_dim + i] > 0: continue
        elevation_normalized = obs[4 + 3 * i]
        if elevation_normalized * 90.0 < MIN_ELEVATION_DEGREES: continue
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

def run_evaluation(scenario: str, model_suffix: str) -> dict:
    print(f"\n--- Evaluating '{model_suffix.upper()}' on '{scenario.upper()}' environment ---")
    model_path = os.path.join("models", model_suffix, "final_model.zip")
    env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
    drl_reward = 0.0
    if os.path.exists(model_path): model = PPO.load(model_path, device='cpu'); drl_reward = evaluate_agent(env, model, policy_type="drl")
    else: print(f"WARNING: Model for '{model_suffix}' not found.")
    greedy_reward = evaluate_agent(env, policy_type="greedy")
    random_reward = evaluate_agent(env, policy_type="random")
    print(f"Results: DRL={drl_reward:,.2f}, Greedy={greedy_reward:,.2f}, Random={random_reward:,.2f}")
    env.close()
    return {"DRL": drl_reward, "Greedy": greedy_reward, "Random": random_reward}

def generate_latex_table(all_results: dict):
    """Generates and prints the LaTeX code for the results table."""
    print("\n" + "="*60)
    print("--- LaTeX Code for Results Table ---")
    print("="*60)
    
    # Extract results for easier access
    static_res = all_results['Static']
    dynamic_res = all_results['Dynamic']
    realistic_res = all_results['Realistic (300k)']
    realistic_3M_res = all_results['Realistic (3M)']
    
    # Calculate relative performance
    greedy_realistic_perf = realistic_3M_res["Greedy"]
    drl_3m_relative_perf = (realistic_3M_res["DRL"] / greedy_realistic_perf) * 100
    random_relative_perf = (realistic_3M_res["Random"] / greedy_realistic_perf) * 100

    # The \
    latex_string = r"""
\begin{table}[t]
  \centering
  \caption{Summary of policy performance across all simulation scenarios. Rewards are the total secure key bits generated in a 24-hour simulation. Relative performance is normalized to the Greedy policy in the Realistic scenario.}
  \label{tab:results}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Policy} & \textbf{Static Env.} & \textbf{Dynamic Env.} & \textbf{Realistic Env.} & \textbf{Relative Perf.} \\
    \midrule
    DRL (300k steps) & -- & -- & \num[group-separator={,}]{%d} & -- \\
    DRL (3M steps) & \num[group-separator={,}]{%d} & \num[group-separator={,}]{%d} & \textbf{\num[group-separator={,}]{%d}} & \textbf{%.1f\%%} \\
    Greedy & \textbf{\num[group-separator={,}]{%d}} & \textbf{\num[group-separator={,}]{%d}} & \num[group-separator={,}]{%d} & 100.0\%% \\
    Random & \num[group-separator={,}]{%d} & \num[group-separator={,}]{%d} & \num[group-separator={,}]{%d} & %.1f\%% \\
    \bottomrule
  \end{tabular}
\end{table}
""" % (
    int(realistic_res["DRL"]),
    int(static_res["DRL"]),
    int(dynamic_res["DRL"]),
    int(realistic_3M_res["DRL"]),
    drl_3m_relative_perf,
    int(static_res["Greedy"]),
    int(dynamic_res["Greedy"]),
    int(greedy_realistic_perf),
    int(static_res["Random"]),
    int(dynamic_res["Random"]),
    int(realistic_3M_res["Random"]),
    random_relative_perf
)
    # Note: Using \num{} from the siunitx package for nice number formatting.
    # Add \usepackage{siunitx} to your LaTeX preamble.
    print(latex_string.strip())
    print("\n" + "="*60)

# --- plot_results function is the same as evaluate_final.py v1.0 ---
def plot_results(all_results: dict):
    # (Code is identical to the previous version, I'm omitting it for brevity)
    # It will create a plot comparing Static, Dynamic, and the new Realistic results.
    print("\n--- Generating High-Quality Final Plot ---")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'font.family': 'serif'})
    
    data = []
    # Using a clearer naming convention for the final plot
    scenarios = ['Static', 'Dynamic', 'Realistic']
    policies_map = {
        'Static': all_results['Static'],
        'Dynamic': all_results['Dynamic'],
        'Realistic': {
            'DRL (300k)': all_results['Realistic (300k)']['DRL'],
            'DRL (3M)': all_results['Realistic (3M)']['DRL'],
            'Greedy': all_results['Realistic (3M)']['Greedy'],
            'Random': all_results['Realistic (3M)']['Random'],
        }
    }
    for scenario in scenarios:
        for policy, reward in policies_map[scenario].items():
            data.append({"Scenario": scenario, "Policy": policy, "Reward": reward})

    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(x="Scenario", y="Reward", hue="Policy", data=df, palette="viridis", edgecolor='black', linewidth=1.0, ax=ax)
    ax.set_title("Policy Performance Comparison Across Simulation Scenarios", pad=20)
    ax.set_ylabel("Total Secure Key Generated (bits)")
    ax.set_xlabel("Environment Scenario")
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x != 0 else '0'))
    ax.legend(title='Policy')
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.0f}', padding=3, fontsize=9, rotation=90)
    fig.tight_layout()
    output_filename = "final_comparison_plot.png"
    fig.savefig(output_filename, dpi=600)
    print(f"Saved final comparison chart to {output_filename}")
    plt.close(fig)

def main():
    all_results = {}
    all_results['Static'] = run_evaluation(scenario='static', model_suffix='static')
    all_results['Dynamic'] = run_evaluation(scenario='dynamic', model_suffix='dynamic')
    # Evaluate both realistic models to compare them
    all_results['Realistic (300k)'] = run_evaluation(scenario='realistic', model_suffix='realistic')
    all_results['Realistic (3M)'] = run_evaluation(scenario='realistic', model_suffix='realistic_3M')
    
    plot_results(all_results)
    generate_latex_table(all_results)
    
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
