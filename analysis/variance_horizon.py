"""
analysis/variance_horizon.py

This script maps the entire Qwixx state space to visualize the "Variance Horizon".
It proves that a 10-point lead in the early game is highly vulnerable to variance, 
but a 10-point lead in the late game is mathematically unlosable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from core.state_encoder import decode_state
from core.environment import calculate_score
from core.constants import ROW_ID_TO_COUNT

def run_deep_analysis():
    print("Loading Exact DP Table (Instant Memory Map)...")
    # Load the exact Win Probability values
    V_win = np.load('data/V_nash_win_prob.npy', mmap_mode='r')
    dag = np.load('data/topological_dag.npy')

    depth_margin_data = []

    print("Scanning all 565,656 mathematical game states...")
    for state in dag:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)

        # Skip states where the game is already over
        if p1_p >= 3 or p2_p >= 3: continue

        # Game Depth = Total crosses + penalties for BOTH players combined
        # This acts as the "clock" for the game.
        depth = ROW_ID_TO_COUNT[p1_r] + ROW_ID_TO_COUNT[p1_b] + p1_p + \
                ROW_ID_TO_COUNT[p2_r] + ROW_ID_TO_COUNT[p2_b] + p2_p

        # Current Score Margin
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        margin = s1 - s2

        # Convert Expected Value [-1.0 to 1.0] into Win Probability [0% to 100%]
        win_prob = ((V_win[state, 0] + 1.0) / 2.0) * 100

        # For a clean heatmap, we bin the margins by 5 and the depth by 2
        if -25 <= margin <= 25:
            binned_margin = int(round(margin / 5.0) * 5)
            binned_depth = int(round(depth / 2.0) * 2)
            depth_margin_data.append({'Depth': binned_depth, 'Margin': binned_margin, 'WinProb': win_prob})

    print("Aggregating data and generating Heatmap...")
    df = pd.DataFrame(depth_margin_data)
    
    # Average the win probabilities for all states that share the same Margin and Depth
    pivot = df.pivot_table(index='Margin', columns='Depth', values='WinProb', aggfunc='mean')
    pivot = pivot.sort_index(ascending=False) # Put high positive margins at the top

    # --- Plotting ---
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    # RdYlBu gives a great Red (Losing) to Yellow (Tied) to Blue (Winning) gradient
    ax = sns.heatmap(pivot, cmap='RdYlBu', annot=False, vmin=0, vmax=100, 
                     cbar_kws={'label': 'Expected Win Probability (%)'})
    
    plt.title('The Variance Horizon: Win Probability by Score Margin and Game Depth', fontsize=15, fontweight='bold')
    plt.xlabel('Game Progression / Depth (Total Combined Marks & Penalties)', fontsize=12)
    plt.ylabel('Player 1 Score Margin', fontsize=12)
    
    # Formatting ticks
    ax.invert_yaxis() 
    plt.tight_layout()
    plt.savefig('plots/variance_horizon.png', dpi=300, bbox_inches='tight')
    print("\nSUCCESS! Heatmap saved to 'plots/variance_horizon.png'")

if __name__ == '__main__':
    run_deep_analysis()