"""
solvers/unified_backward_induction.py

Exact Dynamic Programming Engine for Mini-Qwixx.
Because the game's state transitions strictly monotonically (forming a Directed 
Acyclic Graph or DAG), we can solve the game exactly using single-pass Backward Induction 
rather than iterative approximation (e.g., Value Iteration). 

This script evaluates four distinct mathematical Objective Functions (Win Probability, 
Score Difference, Hybrid, and Solo) by traversing the DAG from terminal states (depth N) 
back to the origin state (s_0).

Thesis Reference: Section II-B "Structural Properties: The DAG Advantage" and Section II-C.
"""

import os
import time
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score, UNIQUE_DICE, get_state_depth
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS, TOTAL_STATES
from solvers.matrix_math import get_nash_probs

# Globals for the memory-mapped worker processes
shared_V = None
HYBRID_WIN_BONUS = 10.0 

def init_worker(shared_array, shape, bonus=10.0):
    """Allocates the shared Value Function array V(s) across all CPU cores."""
    global shared_V, HYBRID_WIN_BONUS
    shared_V = np.frombuffer(shared_array, dtype=np.float32).reshape(shape)
    HYBRID_WIN_BONUS = bonus

# ==========================================
# 1. WIN PROBABILITY SOLVER (Thesis Eq 3)
# ==========================================
def solve_win_prob(state_int):
    """
    Evaluates V_WP(s). 
    Terminal states are mapped strictly to the discrete set {1.0, 0.5, 0.0} (or {1.0, 0.0, -1.0} in zero-sum).
    """
    global shared_V
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    # Base Case: s \in S_terminal
    if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        # Thesis Eq 3 implementation: +1 for win, -1 for loss, 0 for tie
        val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
        shared_V[state_int, 0] = val
        shared_V[state_int, 1] = val
        return

    # Recursive Step: Evaluate Bellman Backup for both possible active players
    for active_player in [1, 2]:
        next_idx = 1 if active_player == 1 else 0
        expected_win_margin = 0.0
        
        # Thesis Eq 6: Integrate over the probability distribution of chance nodes (d \sim D)
        for dice in UNIQUE_DICE:
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            
            # Construct the 3x3 Zero-Sum Subgame (Simultaneous White Phase)
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_val = -9999.0 if active_player == 1 else 9999.0
                    
                    # Thesis Eq 7: Collapse the sequential Color Phase by assuming optimal play (max U_1)
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state_int, active_player, dice, a_w1, a_w2, a_c)
                        if is_term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                            s1 = calculate_score(np1_r, np1_b, np1_p)
                            s2 = calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else:
                            # Pull already computed exact value V(s') from deeper in the DAG
                            val = shared_V[next_s, next_idx]
                            
                        # Maximize evaluation for P1, Minimize evaluation for P2
                        if active_player == 1 and val > best_val: best_val = val
                        elif active_player == 2 and val < best_val: best_val = val
                            
                    payoff_matrix[w1_idx, w2_idx] = best_val
            
            # Thesis Eq 8: Solve for the exact Subgame Perfect Equilibrium probabilities (Minimax Theorem)
            p1_probs, p2_probs = get_nash_probs(payoff_matrix)
            
            # Expected value under optimal mixed strategy support: p1^T * A * p2
            val_matrix = p1_probs.T @ payoff_matrix @ p2_probs
            expected_win_margin += dice['prob'] * val_matrix
            
        shared_V[state_int, 0 if active_player == 1 else 1] = expected_win_margin

# ==========================================
# 2. HYBRID SOLVER (Thesis Eq 5)
# ==========================================
def solve_hybrid(state_int):
    """
    Evaluates V_H(s).
    Combines the Score Difference metric with a scalar Win Bonus (beta) to balance 
    the systemic social welfare loss (Price of Anarchy) against strict win conditions.
    """
    global shared_V, HYBRID_WIN_BONUS
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        score_diff = s1 - s2
        
        # Thesis Eq 5 implementation: Apply the phantom scalar shift (beta)
        if score_diff > 0: hybrid_val = score_diff + HYBRID_WIN_BONUS
        elif score_diff < 0: hybrid_val = score_diff - HYBRID_WIN_BONUS
        else: hybrid_val = 0.0 
            
        shared_V[state_int, 0] = hybrid_val
        shared_V[state_int, 1] = hybrid_val
        return

    for active_player in [1, 2]:
        next_idx = 1 if active_player == 1 else 0
        expected_hybrid_margin = 0.0
        
        for dice in UNIQUE_DICE:
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_val = -9999.0 if active_player == 1 else 9999.0
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state_int, active_player, dice, a_w1, a_w2, a_c)
                        if is_term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                            diff = calculate_score(np1_r, np1_b, np1_p) - calculate_score(np2_r, np2_b, np2_p)
                            if diff > 0: val = diff + HYBRID_WIN_BONUS
                            elif diff < 0: val = diff - HYBRID_WIN_BONUS
                            else: val = 0.0
                        else:
                            val = shared_V[next_s, next_idx]
                            
                        if active_player == 1 and val > best_val: best_val = val
                        elif active_player == 2 and val < best_val: best_val = val
                            
                    payoff_matrix[w1_idx, w2_idx] = best_val
            
            # Subgame Perfect Equilibrium calculation
            p1_probs, p2_probs = get_nash_probs(payoff_matrix)
            val_matrix = p1_probs.T @ payoff_matrix @ p2_probs
            expected_hybrid_margin += dice['prob'] * val_matrix
            
        shared_V[state_int, 0 if active_player == 1 else 1] = expected_hybrid_margin

# ==========================================
# 3. SCORE DIFFERENCE SOLVER (Thesis Eq 4)
# ==========================================
def solve_score_diff(state_int):
    """
    Evaluates V_SD(s). 
    Unlike Win Probability, this preserves the exact expected points. The Value function 
    here explicitly stores a tuple (E[S_1], E[S_2]) to track absolute points, utilizing 
    the difference internally for Minimax evaluation.
    """
    global shared_V
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        shared_V[state_int, 0, 0], shared_V[state_int, 0, 1] = s1, s2
        shared_V[state_int, 1, 0], shared_V[state_int, 1, 1] = s1, s2
        return

    for active_player in [1, 2]:
        next_idx = 1 if active_player == 1 else 0
        expected_s1, expected_s2 = 0.0, 0.0
        
        for dice in UNIQUE_DICE:
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            s1_matrix = np.zeros((3, 3), dtype=np.float32)
            s2_matrix = np.zeros((3, 3), dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_diff = -9999.0 if active_player == 1 else 9999.0
                    best_s1, best_s2 = 0.0, 0.0
                    
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state_int, active_player, dice, a_w1, a_w2, a_c)
                        if is_term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                            s1 = calculate_score(np1_r, np1_b, np1_p)
                            s2 = calculate_score(np2_r, np2_b, np2_p)
                        else:
                            s1 = shared_V[next_s, next_idx, 0]
                            s2 = shared_V[next_s, next_idx, 1]
                            
                        # Thesis Eq 4: Maximize/Minimize the gap S_1 - S_2
                        diff = s1 - s2
                        if active_player == 1 and diff > best_diff:
                            best_diff, best_s1, best_s2 = diff, s1, s2
                        elif active_player == 2 and diff < best_diff:
                            best_diff, best_s1, best_s2 = diff, s1, s2
                            
                    payoff_matrix[w1_idx, w2_idx] = best_diff
                    s1_matrix[w1_idx, w2_idx] = best_s1
                    s2_matrix[w1_idx, w2_idx] = best_s2
            
            p1_probs, p2_probs = get_nash_probs(payoff_matrix)
            
            # Calculate Expected Values by applying the mixed strategy support to the raw score matrices
            expected_s1 += dice['prob'] * (p1_probs.T @ s1_matrix @ p2_probs)
            expected_s2 += dice['prob'] * (p1_probs.T @ s2_matrix @ p2_probs)
            
        shared_V[state_int, 0 if active_player == 1 else 1, 0] = expected_s1
        shared_V[state_int, 0 if active_player == 1 else 1, 1] = expected_s2

# ==========================================
# 4. SOLO OPTIMIZATION SOLVER (Thesis Eq 9-13)
# ==========================================
def solve_solo(state_int):
    """
    Evaluates V_solo(s).
    Mathematically reduces the Markov Game to two independent, single-agent 
    Markov Decision Processes (MDPs). Both players strictly maximize their own points, 
    completely ignoring the zero-sum nature of the opponent's strategy space. 
    Serves as the optimal baseline for Social Welfare.
    """
    global shared_V
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    # Thesis Eq 9: Base Case
    if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        shared_V[state_int, 0, 0], shared_V[state_int, 0, 1] = s1, s2
        shared_V[state_int, 1, 0], shared_V[state_int, 1, 1] = s1, s2
        return

    for active_player in [1, 2]:
        next_idx = 1 if active_player == 1 else 0
        expected_score1, expected_score2 = 0.0, 0.0
        
        for dice in UNIQUE_DICE:
            p1_matrix = np.zeros((3, 3), dtype=np.float32)
            p2_matrix = np.zeros((3, 3), dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_active_val = -9999.0
                    best_vals_pair = (0.0, 0.0)
                    
                    # Thesis Eq 10: Color phase is purely greedily maximized for the active player's own score
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state_int, active_player, dice, a_w1, a_w2, a_c)
                        
                        f_score1 = shared_V[next_s, next_idx, 0]
                        f_score2 = shared_V[next_s, next_idx, 1]
                        
                        if active_player == 1 and f_score1 > best_active_val:
                            best_active_val, best_vals_pair = f_score1, (f_score1, f_score2)
                        elif active_player == 2 and f_score2 > best_active_val:
                            best_active_val, best_vals_pair = f_score2, (f_score1, f_score2)
                                
                    p1_matrix[w1_idx, w2_idx] = best_vals_pair[0]
                    p2_matrix[w1_idx, w2_idx] = best_vals_pair[1]
            
            # Thesis Eq 11 & 12: Solo players ignore the matrix equilibrium entirely.
            # They treat the opponent as uniform environmental noise, taking the argmax 
            # of their own expected row/col means (equivalent to the 1/|Aw| fraction).
            best_w1_idx = np.argmax(np.mean(p1_matrix, axis=1))
            best_w2_idx = np.argmax(np.mean(p2_matrix, axis=0))
            
            # Thesis Eq 13: The deterministic intersection of these independent choices.
            expected_score1 += dice['prob'] * p1_matrix[best_w1_idx, best_w2_idx]
            expected_score2 += dice['prob'] * p2_matrix[best_w1_idx, best_w2_idx]
            
        shared_V[state_int, 0 if active_player == 1 else 1, 0] = expected_score1
        shared_V[state_int, 0 if active_player == 1 else 1, 1] = expected_score2


# ==========================================
# UNIFIED ENGINE (The Topologically Sorted DP Loop)
# ==========================================
def run_unified_induction(model_type, hybrid_bonus=10.0):
    """
    Main orchestrator that maps the topologically sorted DAG onto CPU workers.
    Because depth N states are computed before depth N-1, the recursive Bellman 
    equation is guaranteed to have the exact future values available at execution time.
    """
    # 1. Strategy Mapping
    configs = {
        'win_prob':   {'func': solve_win_prob,   'shape': (TOTAL_STATES, 2),    'out': 'data/V_nash_win_prob.npy', 'msg': 'WIN PROBABILITY'},
        'hybrid':     {'func': solve_hybrid,     'shape': (TOTAL_STATES, 2),    'out': f'data/V_nash_hybrid_{int(hybrid_bonus)}.npy', 'msg': f'HYBRID (Bonus={hybrid_bonus})'},
        'score_diff': {'func': solve_score_diff, 'shape': (TOTAL_STATES, 2, 2), 'out': 'data/V_nash.npy',          'msg': 'SCORE DIFFERENCE (NASH)'},
        'solo':       {'func': solve_solo,       'shape': (TOTAL_STATES, 2, 2), 'out': 'data/V_solo.npy',          'msg': 'SOLO OPTIMIZATION'}
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    cfg = configs[model_type]
    
    # 2. Load and Group DAG
    print(f"Loading Topological DAG for {cfg['msg']}...")
    dag = np.load('data/topological_dag.npy')
    
    # Stratify the DAG by depth. V*(s) is computed from deepest to shallowest.
    depth_groups = defaultdict(list)
    for state in dag: 
        depth_groups[get_state_depth(state)].append(state)
    depths_sorted = sorted(depth_groups.keys(), reverse=True)
    
    # 3. Setup Shared Memory
    flat_size = int(np.prod(cfg['shape']))
    shared_array_base = mp.Array('f', flat_size, lock=False)
    
    start_time = time.time()
    cores = mp.cpu_count()
    print(f"Firing up {cores} CPU cores...")
    
    # 4. Execute Backward Induction Layer by Layer
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base, cfg['shape'], hybrid_bonus)) as pool:
        for depth in depths_sorted:
            states = depth_groups[depth]
            pool.map(cfg['func'], states)
            print(f"Completed Depth {depth:02d} | States solved: {len(states)}")

    # 5. Extract and Save exactly as before
    final_V = np.frombuffer(shared_array_base, dtype=np.float32).reshape(cfg['shape'])
    os.makedirs('data', exist_ok=True)
    np.save(cfg['out'], final_V)
    
    end_time = time.time()
    print(f"\n[{cfg['msg']}] Induction Complete in {round((end_time - start_time)/60, 2)} minutes!")
    
    # Custom Printouts based on model
    if model_type == 'win_prob':
        print(f"Expected Player 1 Win Rate: {((final_V[0, 0] + 1) / 2) * 100:.2f}%")
    elif model_type in ['score_diff', 'solo']:
        print(f"Optimal Expected P1 Score: {final_V[0, 0, 0]:.2f}, P2 Score: {final_V[0, 0, 1]:.2f}")


if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION BLOCK (Single Source of Truth)
    # ==========================================
    
    # Change this string to run different models!
    # Options: 'win_prob', 'hybrid', 'score_diff', 'solo'
    TARGET_MODEL = 'win_prob'  
    
    run_unified_induction(model_type=TARGET_MODEL, hybrid_bonus=10.0)