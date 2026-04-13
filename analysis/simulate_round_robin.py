"""
analysis/simulate_round_robin.py

Empirical Evaluation of the Cross-Play Strategy Matrix.
This script orchestrates a full Round Robin tournament between all 7 evaluated 
policies derived via exact Backward Induction. It computes the pairwise expected 
values (win rates, point averages, and victory margins) when policy pi_i plays 
against policy pi_j.

These results empirically validate the theoretical dominance of the 
Win Probability (V_WP) and Hybrid (V_H) objective functions as defined 
in the Methodology.
"""

import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from analysis.evaluator import TournamentEngine

def plot_heatmaps(win_matrix, p1_score_matrix, p2_score_matrix, p1_margin_matrix, p2_margin_matrix, agents):
    """
    Visualizes the N x N asymmetric payoff matrices for the evaluated strategies.
    Diagonal elements (self-play) are masked with NaN to highlight cross-play dynamics.
    """
    np.fill_diagonal(win_matrix, np.nan)
    np.fill_diagonal(p1_score_matrix, np.nan)
    np.fill_diagonal(p2_score_matrix, np.nan)
    np.fill_diagonal(p1_margin_matrix, np.nan)
    np.fill_diagonal(p2_margin_matrix, np.nan)
    
    os.makedirs('plots', exist_ok=True)
    
    # Map internal representations to their mathematical objective equivalents
    display_names = ['Solo\n(Raw Pts)', 'Score\n(0 Bonus)', 'Hybrid\n(5 Bonus)', 'Hybrid\n(10 Bonus)', 
                     'Hybrid\n(25 Bonus)', 'Hybrid\n(50 Bonus)', 'Win Prob\n(Inf Bonus)']
    annot_settings = {"size": 12, "weight": "bold"}

    def format_and_save(title, xlabel, ylabel, filename):
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
        plt.close() 

    # --- Plot 1: Overall Win Rate % ---
    # Visualizes the empirical approximation of Equation 3 (Win Probability)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(win_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".1f", cmap="RdYlGn", center=50.0, 
                cbar_kws={'label': 'Win Rate % (Row vs Col)'},
                linewidths=1, linecolor='black', annot_kws=annot_settings)
    format_and_save("Qwixx AI: Overall Win Rate (%)", "Opponent Strategy (Column)", "Agent Strategy (Row)", "heatmap_1_win_rate.png")

    # --- Plots 2 & 3: P1 and P2 Points ---
    # Visualizes the expected absolute points (Equation 2) conditional on turn order.
    # Splitting these matrices isolates the "first-mover advantage" inherent in the Qwixx DAG.
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(p1_score_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Blues", 
                cbar_kws={'label': 'Average Points (As Starter)'},
                linewidths=1, linecolor='black', annot_kws=annot_settings)
    format_and_save("Qwixx AI: Average Points Scored (Starting First)", "Opponent (Player 2)", "Agent (Player 1)", "heatmap_2_p1_points.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(p2_score_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Blues", 
                cbar_kws={'label': 'Average Points (As Second)'},
                linewidths=1, linecolor='black', annot_kws=annot_settings)
    format_and_save("Qwixx AI: Average Points Scored (Starting Second)", "Opponent (Player 1)", "Agent (Player 2)", "heatmap_3_p2_points.png")

    # --- Plots 4 & 5: P1 and P2 Margins ---
    # Visualizes the conditional expected value of the Score Difference (Equation 4)
    # E[ S1 - S2 | S1 > S2 ]. This highlights the trade-off of the Hybrid method.
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(p1_margin_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Purples", 
                cbar_kws={'label': 'Winning Margin (When P1 Wins)'},
                linewidths=1, linecolor='black', annot_kws=annot_settings)
    format_and_save("Qwixx AI: Winning Margin (Starting First)", "Opponent (Player 2)", "Agent (Player 1)", "heatmap_4_p1_margin.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(p2_margin_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Purples", 
                cbar_kws={'label': 'Winning Margin (When P2 Wins)'},
                linewidths=1, linecolor='black', annot_kws=annot_settings)
    format_and_save("Qwixx AI: Winning Margin (Starting Second)", "Opponent (Player 1)", "Agent (Player 2)", "heatmap_5_p2_margin.png")

    # ==========================================
    # COMBINED PLOTS (For Side-by-Side Thesis Formatting)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(pd.DataFrame(p1_score_matrix, index=display_names, columns=display_names), annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Average Points (As Starter)'}, linewidths=1, linecolor='black', annot_kws=annot_settings, ax=axes[0])
    axes[0].set_title("Average Points Scored (Starting First)", fontsize=16, fontweight='bold', pad=15); axes[0].set_xlabel("Opponent (Player 2)", fontsize=12, fontweight='bold'); axes[0].set_ylabel("Agent (Player 1)", fontsize=12, fontweight='bold'); axes[0].tick_params(axis='x', rotation=45); axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(pd.DataFrame(p2_score_matrix, index=display_names, columns=display_names), annot=True, fmt=".2f", cmap="Blues", cbar_kws={'label': 'Average Points (As Second)'}, linewidths=1, linecolor='black', annot_kws=annot_settings, ax=axes[1])
    axes[1].set_title("Average Points Scored (Starting Second)", fontsize=16, fontweight='bold', pad=15); axes[1].set_xlabel("Opponent (Player 1)", fontsize=12, fontweight='bold'); axes[1].set_ylabel("Agent (Player 2)", fontsize=12, fontweight='bold'); axes[1].tick_params(axis='x', rotation=45); axes[1].tick_params(axis='y', rotation=0)
    plt.tight_layout(); plt.savefig('plots/heatmap_combined_points.png', dpi=300, bbox_inches='tight'); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    sns.heatmap(pd.DataFrame(p1_margin_matrix, index=display_names, columns=display_names), annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': 'Winning Margin (When P1 Wins)'}, linewidths=1, linecolor='black', annot_kws=annot_settings, ax=axes[0])
    axes[0].set_title("Winning Margin (Starting First)", fontsize=16, fontweight='bold', pad=15); axes[0].set_xlabel("Opponent (Player 2)", fontsize=12, fontweight='bold'); axes[0].set_ylabel("Agent (Player 1)", fontsize=12, fontweight='bold'); axes[0].tick_params(axis='x', rotation=45); axes[0].tick_params(axis='y', rotation=0)

    sns.heatmap(pd.DataFrame(p2_margin_matrix, index=display_names, columns=display_names), annot=True, fmt=".2f", cmap="Purples", cbar_kws={'label': 'Winning Margin (When P2 Wins)'}, linewidths=1, linecolor='black', annot_kws=annot_settings, ax=axes[1])
    axes[1].set_title("Winning Margin (Starting Second)", fontsize=16, fontweight='bold', pad=15); axes[1].set_xlabel("Opponent (Player 1)", fontsize=12, fontweight='bold'); axes[1].set_ylabel("Agent (Player 2)", fontsize=12, fontweight='bold'); axes[1].tick_params(axis='x', rotation=45); axes[1].tick_params(axis='y', rotation=0)
    plt.tight_layout(); plt.savefig('plots/heatmap_combined_margins.png', dpi=300, bbox_inches='tight'); plt.close()

    print("\nSuccessfully generated and saved 7 heatmaps!")


def run_round_robin():
    """
    Executes the pairwise simulation of the strategy space.
    Calculates the expected empirical outcomes using N = 100,000 to ensure 
    statistical significance over the stochastic chance nodes.
    """
    # The Strategy Space (S) defined by the various objective functions
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    
    # Calculate all C(7, 2) unique cross-play combinations
    matchups = list(itertools.combinations(agents, 2))
    
    games_per_matchup = 100000 
    
    # Initialize the N x N payoff matrices
    win_matrix = np.full((len(agents), len(agents)), 50.0)
    p1_score_matrix = np.full((len(agents), len(agents)), 0.0)
    p2_score_matrix = np.full((len(agents), len(agents)), 0.0)
    p1_margin_matrix = np.full((len(agents), len(agents)), np.nan)
    p2_margin_matrix = np.full((len(agents), len(agents)), np.nan)

    print(f"\n" + "="*75)
    print(f" ROUND ROBIN TOURNAMENT: {len(agents)} AGENTS | {len(matchups)} UNIQUE MATCHUPS")
    print(f" Total Simulated Games: {len(matchups) * games_per_matchup}")
    print("="*75)

    for a_idx, b_idx in itertools.combinations(range(len(agents)), 2):
        tag_a, tag_b = agents[a_idx], agents[b_idx]
        print(f"Simulating {games_per_matchup} matches: [{tag_a}] vs [{tag_b}]...")
        
        # DELEGATE TO THE UNIFIED ENGINE
        # Computes the empirical outcomes of the finite Markov game under the specified policies.
        stats = TournamentEngine.run_nash_matchup(tag_a, tag_b, games_per_matchup)

        # Calculate zero-sum win rates. 
        # A tie grants exactly 0.5 wins to both players to maintain the constant-sum property.
        a_win_rate = (((stats['a_as_p1_wins'] + stats['a_as_p2_wins']) + (0.5 * stats['ties'])) / games_per_matchup) * 100
        b_win_rate = (((stats['b_as_p1_wins'] + stats['b_as_p2_wins']) + (0.5 * stats['ties'])) / games_per_matchup) * 100
        
        win_matrix[a_idx][b_idx] = a_win_rate; win_matrix[b_idx][a_idx] = b_win_rate
        
        # Normalize the sum of absolute points by the subset of games played in that position (N / 2)
        half_games = games_per_matchup / 2
        p1_score_matrix[a_idx][b_idx] = stats['a_as_p1_pts'] / half_games
        p1_score_matrix[b_idx][a_idx] = stats['b_as_p1_pts'] / half_games
        p2_score_matrix[a_idx][b_idx] = stats['a_as_p2_pts'] / half_games
        p2_score_matrix[b_idx][a_idx] = stats['b_as_p2_pts'] / half_games

        # Calculate Conditional Expectations E[ Margin | Win ]
        if stats['a_as_p1_wins'] > 0: p1_margin_matrix[a_idx][b_idx] = stats['a_as_p1_margin_sum'] / stats['a_as_p1_wins']
        if stats['b_as_p1_wins'] > 0: p1_margin_matrix[b_idx][a_idx] = stats['b_as_p1_margin_sum'] / stats['b_as_p1_wins']
        if stats['a_as_p2_wins'] > 0: p2_margin_matrix[a_idx][b_idx] = stats['a_as_p2_margin_sum'] / stats['a_as_p2_wins']
        if stats['b_as_p2_wins'] > 0: p2_margin_matrix[b_idx][a_idx] = stats['b_as_p2_margin_sum'] / stats['b_as_p2_wins']
        
        print(f"  -> {tag_a}: {a_win_rate:.1f}% | {tag_b}: {b_win_rate:.1f}% | Ties: {(stats['ties']/games_per_matchup)*100:.1f}%")

    print("\nGenerating Master Heatmaps...")
    plot_heatmaps(win_matrix, p1_score_matrix, p2_score_matrix, p1_margin_matrix, p2_margin_matrix, agents)

if __name__ == '__main__':
    run_round_robin()