"""
analysis/strategic_divergence.py

Exhaustive Multiprocessing version to mathematically prove the strategic 
difference between Point-Maximization and Defensive Play.
This script evaluates EVERY possible mid-game state against EVERY possible dice roll.
"""

import numpy as np
import os
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS
from solvers.matrix_math import get_nash_probs

# Global variables for shared memory across multiprocessing workers
V_score_shared = None
V_win_shared = None

# Because White Dice are symmetric (1-2 is identical to 2-1), 
# we don't need to evaluate all 81 permutations inside the hot loop.
UNIQUE_DICE = []
for w1 in (1, 2, 3):
    for w2 in range(w1, 4):  # w2 starts at w1 to avoid duplicates
        weight = 1 if w1 == w2 else 2 # Double the weight if it's an asymmetric roll
        for r in (1, 2, 3):
            for b in (1, 2, 3):
                UNIQUE_DICE.append(({'W1': w1, 'W2': w2, 'R': r, 'B': b}, weight))

def init_worker():
    """Initializes the Exact DP tables for each CPU core."""
    global V_score_shared, V_win_shared
    # OPTIMIZATION 1: Removed mmap_mode='r'. 
    # Since the tables are tiny, load them entirely into ultra-fast RAM!
    V_score_shared = np.load('data/V_nash_hybrid_50.npy')
    V_win_shared = np.load('data/V_nash_win_prob.npy')

def _analyze_chunk(states_chunk):
    """
    Worker function to process a chunk of states.
    Iterates exhaustively through the 54 unique dice combinations per state.
    """
    global V_score_shared, V_win_shared
    
    local_totals = {}
    local_disagreements = {}
    
    M_score = np.empty((3, 3), dtype=np.float32)
    M_win = np.empty((3, 3), dtype=np.float32)

    for state in states_chunk:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        
        # Skip late endgame states to focus purely on strategic mid-game divergence
        if p1_p >= 2 or p2_p >= 2: 
            continue
            
        margin = calculate_score(p1_r, p1_b, p1_p) - calculate_score(p2_r, p2_b, p2_p)
        
        # Loop through our 54 pre-calculated dice rolls instead of 81 nested loops
        for dice, weight in UNIQUE_DICE:
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_score = -9999.0
                    best_win = -9999.0
                    
                    for a_c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, 1, dice, a_w1, a_w2, a_c)
                        
                        if term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                            s1 = calculate_score(np1_r, np1_b, np1_p)
                            s2 = calculate_score(np2_r, np2_b, np2_p)
                            val_score = float(s1 - s2)
                            val_win = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else:
                            val_score = V_score_shared[ns, 1, 0] - V_score_shared[ns, 1, 1]
                            val_win = V_win_shared[ns, 1]
                                
                        if val_score > best_score: best_score = val_score
                        if val_win > best_win: best_win = val_win
                            
                    M_score[w1_idx, w2_idx] = best_score
                    M_win[w1_idx, w2_idx] = best_win

            p1_score, _ = get_nash_probs(M_score)
            p1_win, _ = get_nash_probs(M_win)
            
            act_score = np.argmax(p1_score)
            act_win = np.argmax(p1_win)
            
            if margin not in local_totals:
                local_totals[margin] = 0
                local_disagreements[margin] = 0
                
            # Apply the mathematical weight (1 or 2) instead of +1
            local_totals[margin] += weight
            if act_score != act_win:
                local_disagreements[margin] += weight
            
    return local_totals, local_disagreements

def run_exhaustive_divergence_analysis():
    print("Pre-compiling Numba C-Code (Warm-up)...")
    dummy_dice = {'W1': 1, 'W2': 1, 'R': 1, 'B': 1}
    MiniQwixxEnv.step(0, 1, dummy_dice, 'Pass', 'Pass', None)
    get_nash_probs(np.zeros((3, 3), dtype=np.float32))

    print("Loading DAG and distributing exhaustive workload across CPU cores...")
    print("WARNING: Evaluating over 1 BILLION transitions. This may take a few minutes...")
    
    dag = np.load('data/topological_dag.npy')
    
    # We no longer sample. We pass the entire DAG to the workers.
    # (The workers will internally skip the extreme endgame states).
    all_states = [int(s) for s in dag]
    
    cores = mp.cpu_count()
    chunk_size = len(all_states) // cores
    chunks = [all_states[i:i + chunk_size] for i in range(0, len(all_states), chunk_size)]
    
    start_time = time.time()
    with mp.Pool(processes=cores, initializer=init_worker) as pool:
        results = pool.map(_analyze_chunk, chunks)
    print(f"Exhaustive Analysis complete in {time.time() - start_time:.2f} seconds.")
        
    final_totals = {}
    final_disagreements = {}
    
    for local_totals, local_disagreements in results:
        for m, count in local_totals.items():
            final_totals[m] = final_totals.get(m, 0) + count
            final_disagreements[m] = final_disagreements.get(m, 0) + local_disagreements[m]

    # Because this is exhaustive, we can trust margins with even a low state count, 
    # but we filter out extreme edge cases (like +/- 25) to keep the graph clean.
    margins = sorted([m for m in final_totals.keys() if final_totals[m] > 500])
    divergence_rates = [(final_disagreements[m] / final_totals[m]) * 100 for m in margins]

    # --- Plotting ---
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = ['#d62728' if m < 0 else '#1f77b4' for m in margins]
    plt.bar(margins, divergence_rates, color=colors, alpha=0.8, edgecolor='black')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label="Tied Game")
    
    plt.title("Exact Strategic Disagreement: Win Probability vs. Hybrid-50", fontsize=14, fontweight='bold')
    plt.xlabel("Player 1 Current Score Margin", fontsize=12)
    plt.ylabel("Action Divergence Rate (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig('plots/strategic_divergence_exact2.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved successfully to 'plots/strategic_divergence_exact2.png'!")

if __name__ == '__main__':
    run_exhaustive_divergence_analysis()