import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curve():
    print("Loading RL MSE History...")
    try:
        mse_history = np.load('data/mse_history_minimax_hybrid.npy')
    except FileNotFoundError:
        print("Error: Could not find data/mse_history_minimax_hybrid.npy")
        return

    # Create the X-axis (Episodes). We logged data every 1000 episodes.
    episodes = np.arange(len(mse_history)) * 1000

    # Calculate a rolling average to smooth out the noisy RL spikes
    window_size = 50
    smoothed_mse = np.convolve(mse_history, np.ones(window_size)/window_size, mode='valid')
    smoothed_episodes = episodes[window_size - 1:]

    plt.figure(figsize=(10, 6))
    
    # Plot the raw, transparent data in the background
    plt.plot(episodes, mse_history, color='blue', alpha=0.2, label='Raw MSE')
    
    # Plot the smoothed data clearly on top
    plt.plot(smoothed_episodes, smoothed_mse, color='darkblue', linewidth=2.5, label=f'Trend (Rolling Avg: {window_size})')

    plt.title("Minimax Q-Learning Convergence (Hybrid-25 Target)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Training Episodes", fontsize=12, fontweight='bold')
    plt.ylabel("Mean Squared Error (MSE) of Start State", fontsize=12, fontweight='bold')
    
    # Set y-axis to log scale if the initial spike is massive, otherwise linear is fine
    plt.ylim(0, max(4.0, np.max(smoothed_mse) * 1.1)) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    os.makedirs('plots', exist_ok=True)
    plt.tight_layout()
    save_path = 'plots/rl_learning_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"Success! Learning curve saved to {save_path}")
    plt.show()

if __name__ == '__main__':
    plot_learning_curve()