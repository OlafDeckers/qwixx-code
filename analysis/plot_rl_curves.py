import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rl_curves():
    print("Loading RL training histories...")
    try:
        mse_sparse = np.load('data/mse_history_sparse.npy')
        mse_dense = np.load('data/mse_history_dense.npy')
        mse_minimax = np.load('data/mse_history_minimax.npy')
    except FileNotFoundError:
        print("Error: Could not find one or more MSE history files. Make sure all 3 models have finished training!")
        return

    plt.figure(figsize=(10, 6))
    
    # 1. Dynamically generate the exact X-axis for EACH array independently
    x_sparse = np.arange(len(mse_sparse)) * 50000
    x_dense = np.arange(len(mse_dense)) * 50000
    x_minimax = np.arange(len(mse_minimax)) * 50000

    # 2. Plot all three curves with explicit data point markers
    plt.plot(x_sparse, mse_sparse, label='Model A: Sparse Q-Learning', color='#2ca02c', linewidth=2, marker='o', markersize=5)
    plt.plot(x_dense, mse_dense, label='Model B: Dense Q-Learning (Greedy)', color='#d62728', linewidth=2, marker='s', markersize=5)
    plt.plot(x_minimax, mse_minimax, label='Model C: Minimax Q-Learning (Adversarial)', color='#1f77b4', linewidth=2, marker='^', markersize=5)

    # 3. Add the target convergence line
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, label='Perfect Convergence (MSE=0)')

    # 4. Formatting for Academic Presentation
    plt.title('Reinforcement Learning Convergence over 1,000,000 Episodes\n(Target: True Nash Equilibrium of +0.6462)', fontsize=14, fontweight='bold')
    plt.xlabel('Training Episodes', fontsize=12)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Dynamically cap the Y-axis so Model B's explosion doesn't squash the graph completely flat
    max_y = min(np.max(mse_dense) + 0.5, 5.0) 
    plt.ylim(-0.2, max_y)
    
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/rl_learning_curves.png', dpi=300)
    print(f"Plotted {len(mse_sparse)} Sparse points, {len(mse_dense)} Dense points, and {len(mse_minimax)} Minimax points.")
    print("Graph saved successfully to 'plots/rl_learning_curves.png'!")
    
    # Display the graph in a popup window
    plt.show()

if __name__ == '__main__':
    plot_rl_curves()