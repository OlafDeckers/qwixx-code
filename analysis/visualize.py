import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def plot_price_of_anarchy_bar():
    print("Loading DP arrays for Price of Anarchy plot...")
    V_nash = np.load('data/V_nash.npy')
    V_solo = np.load('data/V_solo.npy')

    # Extract exact values for State 0 (The empty board, Player 1 active)
    nash_diff = V_nash[0, 0]
    solo_p1 = V_solo[0, 0, 0]
    solo_p2 = V_solo[0, 0, 1]
    solo_diff = solo_p1 - solo_p2

    # Set up the bar chart
    plt.figure(figsize=(8, 6))
    
    categories = ['Greedy Baseline\n(Solo Policy)', 'Zero-Sum Defense\n(Nash Equilibrium)']
    values = [solo_diff, nash_diff]
    colors = ['#ff6b6b', '#4834d4'] # Soft red vs. Strong deep blue

    bars = plt.bar(categories, values, color=colors, width=0.5, edgecolor='black', linewidth=1.5)

    # Add exact numeric labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'+{height:.2f} pts',
                 ha='center', va='bottom', fontweight='bold', fontsize=14)

    # Draw a dashed line to highlight the "Strategic Gap" (Price of Anarchy)
    plt.axhline(y=solo_diff, color='gray', linestyle='--', alpha=0.7)
    
    # Annotate the Price of Anarchy
    poa = nash_diff - solo_diff
    plt.text(1.28, (nash_diff + solo_diff)/2, f'Price of Anarchy\n(Strategic Gap: {poa:.2f})', 
             ha='left', va='center', color='#4834d4', fontweight='bold', fontstyle='italic',
             bbox=dict(facecolor='white', edgecolor='#4834d4', boxstyle='round,pad=0.5'))

    plt.title('The First-Mover Advantage in Mini-Qwixx', pad=20, fontweight='bold', fontsize=16)
    plt.ylabel('Expected Point Advantage\n(Player 1 vs. Player 2)', fontweight='bold')
    plt.ylim(0, max(values) + 1.0) # Give room for the labels at the top
    
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/price_of_anarchy_bar.png', dpi=300)
    print("Saved 'plots/price_of_anarchy_bar.png'")
    plt.close()

def plot_rl_learning_curve():
    print("Loading MSE history for RL Learning Curve plot...")
    try:
        mse_history = np.load('data/mse_history.npy')
    except FileNotFoundError:
        print("Could not find mse_history.npy. Skipping RL plot.")
        return

    # We saved the MSE every 20,000 episodes
    episodes = np.arange(0, len(mse_history) * 20000, 20000)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mse_history, color='#27ae60', lw=2.5) # Strong green
    
    plt.title('Minimax-Q Agent Learning Curve\n(Dense Rewards & Exploring Starts)', pad=15, fontweight='bold')
    plt.xlabel('Training Episodes', fontweight='bold')
    plt.ylabel('Mean Squared Error (vs. Exact Nash Value)', fontweight='bold')
    
    # Format x-axis to show "K" for thousands
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig('plots/rl_learning_curve.png', dpi=300)
    print("Saved 'plots/rl_learning_curve.png'")
    plt.close()

if __name__ == '__main__':
    print("Generating Thesis Visualizations...")
    plot_price_of_anarchy_bar()
    plot_rl_learning_curve()
    print("Done! Check the 'plots' folder.")