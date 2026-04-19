"""
analysis/plot_model_comparison.py

Empirical Validation of Reinforcement Learning Convergence.
This script evaluates the performance of three distinct RL architectures 
(Reward Shaping, TD-Lambda, and Boltzmann Exploration) against an unenhanced 
Standard baseline and the exact Dynamic Programming (DP) 'Win Probability' target.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from analysis.evaluator import TournamentEngine

# N = 100,000: The Monte Carlo sample size used to approximate the Expected Win Rate.
# A large N is required to minimize variance in the stochastic game environment.
NUM_SIMULATED_GAMES = 100000 

def generate_comparison_plot():
    """
    Generates the learning curve comparison plot. 
    It tracks how effectively each RL algorithm approximates the optimal Value Function V*(s) 
    by testing the policies derived from their Q-tables against the perfect DP agent.
    """
    
    # Define the RL methodologies evaluated in the thesis, including the new baseline.
    models = {
        # Equation 17:The unenhanced baseline (Standard Minimax-Q with Epsilon-Greedy, terminal rewards only)
        'Standard Baseline': {'prefix': 'V_rl_standard_', 'color': '#8E44AD', 'marker': 'd'},
        # Equation 18: Modifies the MDP reward structure with domain-specific heuristics
        'Reward Shaping': {'prefix': 'V_rl_reward_shape_', 'color': '#4A90E2', 'marker': 'o'},
        # Equations 19-22: Uses replacing eligibility traces to backpropagate rewards across trajectories
        'TD-Lambda': {'prefix': 'V_rl_td_lambda_', 'color': '#D0021B', 'marker': 's'},
        # Equation 23: Softmax action selection over the Q-values to balance exploration/exploitation
        'Boltzmann': {'prefix': 'V_rl_boltzmann_', 'color': '#F5A623', 'marker': '^'}
    }
    
    # Training checkpoints (in millions of episodes/iterations)
    milestones = [5, 10, 15, 20]
    plot_data = {name: [] for name in models.keys()}
    
    print(f"\nStarting the Checkpoint Tournament ({NUM_SIMULATED_GAMES:,} games per checkpoint)...\n")

    # Step 1: Evaluate each RL agent sequentially to prevent nested multiprocessing crashes
    for model_name, info in models.items():
        print(f"--- Evaluating {model_name} ---")
        
        # t=0: Untrained baseline (Random Policy expected win rate is effectively 0% against perfect DP)
        plot_data[model_name].append(0.0) 
        
        for m in milestones:
            file_path = f"data/checkpoints/{info['prefix']}{m}000K.npy"
            
            if os.path.exists(file_path):
                print(f"  Loading {m}M Checkpoint...")
                
                # DELEGATE TO THE UNIFIED ENGINE
                custom_paths = {'RL_AGENT': file_path}
                rl_wins = TournamentEngine.run_pure_minmax_matchup('RL_AGENT', 'WIN', custom_paths, num_games=NUM_SIMULATED_GAMES)
                
                # Calculate empirical win probability
                win_rate = (rl_wins / NUM_SIMULATED_GAMES) * 100
                plot_data[model_name].append(win_rate)
                print(f"  Result: {win_rate:.1f}% Win Rate")
            else:
                print(f"  [!] Missing file: {file_path}. Filling with NaN.")
                plot_data[model_name].append(np.nan)

    print("\nGenerating the Graph...")
    os.makedirs('plots', exist_ok=True)
    
    # ========================================================
    # MATPLOTLIB VISUALIZATION
    # ========================================================
    plt.figure(figsize=(12, 7))
    x_axis = [0, 5, 10, 15, 20]
    
    for model_name, info in models.items():
        plt.plot(x_axis, plot_data[model_name], label=model_name, 
                 color=info['color'], marker=info['marker'], linewidth=2.5, markersize=8)

    plt.title("RL Agent Performance vs. Exact DP (Win Probability)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Training Iterations (Millions)", fontsize=13, fontweight='bold')
    plt.ylabel("Win Rate (%) against Perfect DP", fontsize=13, fontweight='bold')
    
    plt.xticks(x_axis, [f"{x}M" for x in x_axis])
    
    # Dynamically scale the Y-axis
    max_val = max([max([v for v in vals if not np.isnan(v)], default=0) for vals in plot_data.values()])
    plt.ylim(0, max(50, max_val + 10))
    
    plt.axhline(y=50.0, color='gray', linestyle='--', alpha=0.7, label='Theoretical Max (~50%)')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()

    save_path = 'plots/rl_model_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Success! Plot saved beautifully to {save_path}")
    
    plt.show()

if __name__ == '__main__':
    generate_comparison_plot()