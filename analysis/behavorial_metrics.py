"""
analysis/behavioral_metrics.py

Evaluates the strategic behavior of the different Qwixx AI models.
By simulating self-play, this script tracks the terminal states to calculate:
1. Average Penalties Taken (Defensive Stalling)
2. Average Marks Skipped (Offensive Rushing)

Outputs a dual-axis bar chart to visually compare the risk profiles of the models.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import random
import time

from core.environment import MiniQwixxEnv, ROW_DETAILS_ARRAY
from core.state_encoder import decode_state
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS
from analysis.evaluator import (
    AGENT_TO_ID, init_tournament_worker, evaluate_state,
    get_nash_probs, fast_choice
)

def _behavior_chunk(args):
    """
    Simulates self-play games for a specific agent and records terminal behaviors.
    """
    num_games, agent_str = args
    stats = {'skips': 0, 'penalties': 0}
    agent_id = AGENT_TO_ID[agent_str]

    M_p1 = np.empty((3, 3), dtype=np.float32)
    M_p2 = np.empty((3, 3), dtype=np.float32)
    best_c_matrix = [[None]*3 for _ in range(3)]

    for _ in range(num_games):
        state = 0
        active_player = 1

        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            
            # 1. Terminal Check
            if p1_p >= 3 or p2_p >= 3 or ((p1_r >= 11 or p2_r >= 11) and (p1_b >= 11 or p2_b >= 11)):
                
                # Inline function to calculate skips mathematically via the DAG mapping
                def calc_skips(r_id, b_id):
                    s = 0
                    for row_id in (r_id, b_id):
                        idx, count = ROW_DETAILS_ARRAY[row_id]
                        if idx != -1: # If row is not empty
                            # Locked rows (ID >= 11) have a +1 phantom point bonus in the count
                            actual_marks = count - 1 if row_id >= 11 else count
                            # Skips = Total boxes passed - Actual boxes marked
                            s += (idx + 1) - actual_marks
                    return s

                stats['skips'] += calc_skips(p1_r, p1_b) + calc_skips(p2_r, p2_b)
                stats['penalties'] += p1_p + p2_p
                break

            # 2. Chance Node
            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_idx = 1 if active_player == 1 else 0

            # 3. Payoff Matrices
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_c = None
                    best_val = -9999.0 if active_player == 1 else 9999.0

                    for c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, c)
                        val = evaluate_state(ns, next_idx, term, agent_id, active_player)

                        if active_player == 1 and val > best_val:
                            best_val = val; best_c = c
                        elif active_player == 2 and val < best_val:
                            best_val = val; best_c = c

                    best_c_matrix[w1_idx][w2_idx] = best_c
                    final_ns, final_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c)

                    M_p1[w1_idx, w2_idx] = evaluate_state(final_ns, next_idx, final_term, agent_id, 1)
                    M_p2[w1_idx, w2_idx] = evaluate_state(final_ns, next_idx, final_term, agent_id, 2)

            p1_probs, _ = get_nash_probs(M_p1)
            _, p2_probs = get_nash_probs(M_p2)

            idx_w1 = fast_choice(p1_probs)
            idx_w2 = fast_choice(p2_probs)

            state, _ = MiniQwixxEnv.step(state, active_player, dice, WHITE_ACTIONS[idx_w1], WHITE_ACTIONS[idx_w2], best_c_matrix[idx_w1][idx_w2])
            active_player = 2 if active_player == 1 else 1

    return stats


def run_behavioral_analysis():
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    display_names = ['Solo\n(Raw Pts)', 'Score\n(0 Bonus)', 'Hybrid\n(5 Bonus)', 'Hybrid\n(10 Bonus)', 
                     'Hybrid\n(25 Bonus)', 'Hybrid\n(50 Bonus)', 'Win Prob\n(Inf Bonus)']

    games_per_agent = 100000 
    cores = mp.cpu_count()

    avg_skips = []
    avg_penalties = []

    print(f"\n" + "="*75)
    print(f" BEHAVIORAL ANALYSIS: {games_per_agent} SELF-PLAY GAMES PER AGENT")
    print("="*75)

    for agent in agents:
        print(f"Simulating self-play for [{agent}]...")
        
        # Partition workload across cores
        games_per_core = [games_per_agent // cores] * cores
        for i in range(games_per_agent % cores): games_per_core[i] += 1
        args = [(n, agent) for n in games_per_core]

        with mp.Pool(processes=cores, initializer=init_tournament_worker, initargs=([agent],)) as pool:
            results = pool.map(_behavior_chunk, args)

        total_skips = sum(r['skips'] for r in results)
        total_penalties = sum(r['penalties'] for r in results)

        # Divide by (games * 2) because there are 2 players generating stats in each game
        avg_skips.append(total_skips / (games_per_agent * 2))
        avg_penalties.append(total_penalties / (games_per_agent * 2))

        print(f"  -> Skips: {avg_skips[-1]:.2f}/player | Penalties: {avg_penalties[-1]:.2f}/player")

    # ==========================================
    # DUAL-AXIS BAR CHART PLOTTING
    # ==========================================
    print("\nGenerating Behavioral Bar Chart...")
    os.makedirs('plots', exist_ok=True)
    
    x = np.arange(len(agents))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Left Axis: Marks Skipped (Blue)
    color1 = '#1f77b4'
    rects1 = ax1.bar(x - width/2, avg_skips, width, label='Marks Skipped', color=color1, edgecolor='black', zorder=3)
    ax1.set_ylabel('Average Marks Skipped per Player', fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Right Axis: Penalties Taken (Red)
    ax2 = ax1.twinx()
    color2 = '#d62728'
    rects2 = ax2.bar(x + width/2, avg_penalties, width, label='Penalties Taken', color=color2, edgecolor='black', zorder=3)
    ax2.set_ylabel('Average Penalties Taken per Player', fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Title and Legend
    plt.title('Qwixx AI Behavior: Offensive Rushing vs. Defensive Stalling', fontsize=16, fontweight='bold', pad=15)
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig('plots/behavioral_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Successfully saved to plots/behavioral_metrics_comparison.png")

if __name__ == '__main__':
    run_behavioral_analysis()