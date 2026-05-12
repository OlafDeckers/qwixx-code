"""
analysis/find_intentional_penalty.py

Optimized Multiprocessing version to hunt for a "Golden State" 
where the Win Probability agent chooses an intentional penalty to "starve" 
the opponent, while the Score Difference agent chooses to cross a box.
"""

import numpy as np
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS
from solvers.matrix_math import get_nash_probs

# Pre-calculate the 54 unique dice combinations
UNIQUE_DICE = []
for w1 in (1, 2, 3):
    for w2 in range(w1, 4):
        weight = 1 if w1 == w2 else 2
        for r in (1, 2, 3):
            for b in (1, 2, 3):
                UNIQUE_DICE.append(({'W1': w1, 'W2': w2, 'R': r, 'B': b}, weight))

# Global variables for shared memory across multiprocessing workers
V_win_shared = None
V_score_shared = None

def init_worker():
    """Initializes the memory-mapped Exact DP tables for each CPU core."""
    global V_win_shared, V_score_shared
    V_win_shared = np.load('data/V_nash_win_prob.npy')
    V_score_shared = np.load('data/V_nash.npy')

def _hunt_chunk(states_chunk):
    """
    Worker function to process a chunk of states.
    Builds matrices for both Win Probability and Score Difference, 
    and finds a state where their optimal choices strictly diverge.
    """
    global V_win_shared, V_score_shared
    
    for state in states_chunk:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        
        # We want a state where P1 is leading but hasn't won/lost yet
        if s1 <= s2 or p1_p >= 3 or p2_p >= 3:
            continue
            
        for dice, _ in UNIQUE_DICE:
            # Matrices for Win Probability
            M_win_all = np.full((3, 3), -9999.0, dtype=np.float32)
            M_win_no_pen = np.full((3, 3), -9999.0, dtype=np.float32)
            
            # Matrices for Score Difference
            M_score_all = np.full((3, 3), -9999.0, dtype=np.float32)
            M_score_no_pen = np.full((3, 3), -9999.0, dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_w_all = -9999.0
                    best_w_no_pen = -9999.0
                    best_s_all = -9999.0
                    best_s_no_pen = -9999.0
                    
                    for a_c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, 1, dice, a_w1, a_w2, a_c)
                        
                        # Check the physical state to see if a penalty was actually taken
                        _, _, np1_p_check, _, _, _ = decode_state(ns)
                        is_penalty = (np1_p_check > p1_p)
                        
                        if term:
                            ns1 = calculate_score(*decode_state(ns)[:3])
                            ns2 = calculate_score(*decode_state(ns)[3:])
                            val_win = 1.0 if ns1 > ns2 else (-1.0 if ns1 < ns2 else 0.0)
                            val_score = float(ns1 - ns2)
                        else:
                            val_win = V_win_shared[ns, 1]
                            val_score = float(V_score_shared[ns, 1, 0] - V_score_shared[ns, 1, 1])
                            
                        # Update "All Actions" max values
                        if val_win > best_w_all: best_w_all = val_win
                        if val_score > best_s_all: best_s_all = val_score
                            
                        # Update "No Penalty" max values
                        if not is_penalty:
                            if val_win > best_w_no_pen: best_w_no_pen = val_win
                            if val_score > best_s_no_pen: best_s_no_pen = val_score
                                
                    M_win_all[w1_idx, w2_idx] = best_w_all
                    M_win_no_pen[w1_idx, w2_idx] = best_w_no_pen
                    
                    M_score_all[w1_idx, w2_idx] = best_s_all
                    M_score_no_pen[w1_idx, w2_idx] = best_s_no_pen
            
            # Verify P1 actually had a guaranteed choice to avoid a penalty
            valid_w1_exists = False
            for w1_idx in range(3):
                if np.all(M_win_no_pen[w1_idx] > -9000):
                    valid_w1_exists = True
                    break
                    
            if not valid_w1_exists:
                continue 
                
            # Solve the Nash Equilibrium probabilities for Win Probability
            p1_w_all, p2_w_all = get_nash_probs(M_win_all)
            p1_w_no, p2_w_no = get_nash_probs(M_win_no_pen)
            val_win_all = p1_w_all.dot(M_win_all).dot(p2_w_all)
            val_win_no_pen = p1_w_no.dot(M_win_no_pen).dot(p2_w_no)
            
            # Solve the Nash Equilibrium probabilities for Score Difference
            p1_s_all, p2_s_all = get_nash_probs(M_score_all)
            p1_s_no, p2_s_no = get_nash_probs(M_score_no_pen)
            val_score_all = p1_s_all.dot(M_score_all).dot(p2_s_all)
            val_score_no_pen = p1_s_no.dot(M_score_no_pen).dot(p2_s_no)
            
            # THE TRUE GOLDEN CONDITION:
            # 1. Win Probability agent strictly prefers taking a penalty.
            # 2. Score Difference agent strictly prefers crossing a box!
            win_agent_prefers_penalty = val_win_all > val_win_no_pen + 1e-4
            score_agent_prefers_cross = val_score_no_pen > val_score_all + 1e-4
            
            if win_agent_prefers_penalty and score_agent_prefers_cross:
                return {
                    'p1_state': f"Red={p1_r}, Blue={p1_b}, Penalties={p1_p} (Score: {s1})",
                    'p2_state': f"Red={p2_r}, Blue={p2_b}, Penalties={p2_p} (Score: {s2})",
                    'dice': dice,
                    'win_cross': val_win_no_pen,
                    'win_penalty': val_win_all,
                    'score_cross': val_score_no_pen,
                    'score_penalty': val_score_all
                }
    return None

def run_fast_hunter():
    print("Pre-compiling Numba C-Code (Warm-up)...")
    dummy_dice = {'W1': 1, 'W2': 1, 'R': 1, 'B': 1}
    MiniQwixxEnv.step(0, 1, dummy_dice, 'Pass', 'Pass', None)
    get_nash_probs(np.zeros((3, 3), dtype=np.float32))

    print("Loading DAG and unleashing CPU cores...")
    dag = np.load('data/topological_dag.npy')
    
    all_states = [int(s) for s in reversed(dag)]
    cores = mp.cpu_count()
    chunk_size = len(all_states) // cores
    chunks = [all_states[i:i + chunk_size] for i in range(0, len(all_states), chunk_size)]
    
    start_time = time.time()
    
    with mp.Pool(processes=cores, initializer=init_worker) as pool:
        for result in pool.imap_unordered(_hunt_chunk, chunks):
            if result is not None:
                print("\n" + "="*60)
                print("🎯 TRUE STRATEGIC DIVERGENCE FOUND!")
                print("="*60)
                print(f"P1 State: {result['p1_state']}")
                print(f"P2 State: {result['p2_state']}")
                print(f"Dice Roll: {result['dice']}")
                print("-" * 60)
                print("WIN PROBABILITY AGENT:")
                print(f"  Win% if crossing a box: {(result['win_cross']+1)/2 * 100:.2f}%")
                print(f"  Win% if taking PENALTY: {(result['win_penalty']+1)/2 * 100:.2f}%  <-- PREFERS PENALTY")
                print("-" * 60)
                print("SCORE DIFFERENCE AGENT:")
                print(f"  Expected Margin if crossing a box: {result['score_cross']:+.2f} pts  <-- PREFERS CROSS")
                print(f"  Expected Margin if taking PENALTY: {result['score_penalty']:+.2f} pts")
                print("="*60)
                
                print(f"Found in {time.time() - start_time:.2f} seconds.")
                pool.terminate()
                return

    print(f"Search complete in {time.time() - start_time:.2f} seconds. No state found.")

if __name__ == '__main__':
    run_fast_hunter()