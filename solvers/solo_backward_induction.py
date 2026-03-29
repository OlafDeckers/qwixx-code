import os
import time
import numpy as np
import multiprocessing as mp
from collections import defaultdict, Counter

from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

# Number of marks for each Row ID (0 to 13)
ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

shared_V_solo = None

def init_worker(shared_array):
    global shared_V_solo
    # Dimensions: [1,048,576 states] x [2 active players] x [2 scores (P1, P2)]
    shared_V_solo = np.frombuffer(shared_array, dtype=np.float32).reshape((1048576, 2, 2))

def calculate_score(r_id, b_id, penalties):
    """Applies the triangular number formula for scoring: x(x+1)/2 - 3p"""
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return float(((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties))

def get_unique_dice_combinations():
    """Compresses 81 rolls down to 54 unique permutations."""
    combinations = []
    for w1 in [1,2,3]:
        for w2 in [1,2,3]:
            for r in [1,2,3]:
                for b in [1,2,3]:
                    w_tuple = tuple(sorted([w1, w2]))
                    combinations.append((w_tuple[0], w_tuple[1], r, b))
    counts = Counter(combinations)
    return [{'W1': d[0], 'W2': d[1], 'R': d[2], 'B': d[3], 'prob': count / 81.0} for d, count in counts.items()]

UNIQUE_DICE = get_unique_dice_combinations()
WHITE_ACTIONS = ['R', 'B', None]
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

def solve_single_state_solo(state_int):
    global shared_V_solo
    
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
    blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
    
    # 1. Terminal Check
    if p1_p >= 3 or p2_p >= 3 or (red_locked and blue_locked):
        score1 = calculate_score(p1_r, p1_b, p1_p)
        score2 = calculate_score(p2_r, p2_b, p2_p)
        
        shared_V_solo[state_int, 0, 0] = score1
        shared_V_solo[state_int, 0, 1] = score2
        shared_V_solo[state_int, 1, 0] = score1
        shared_V_solo[state_int, 1, 1] = score2
        return

    # 2. Evaluate Non-Terminal
    for active_player in [1, 2]:
        next_active_player_idx = 1 if active_player == 1 else 0
        expected_score1 = 0.0
        expected_score2 = 0.0
        
        for dice in UNIQUE_DICE:
            # 3x3 Matrices tracking P1's expected raw score and P2's expected raw score
            p1_matrix = np.zeros((3, 3), dtype=np.float32)
            p2_matrix = np.zeros((3, 3), dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    
                    best_active_val = -9999.0
                    best_vals_pair = (0.0, 0.0)
                    
                    # Color Phase: The Active Player picks the move that maximizes THEIR OWN score
                    for a_c in COLOR_ACTIONS:
                        next_state_int, is_terminal = MiniQwixxEnv.step(
                            state_int, active_player, dice, a_w1, a_w2, a_c
                        )
                        
                        f_score1 = shared_V_solo[next_state_int, next_active_player_idx, 0]
                        f_score2 = shared_V_solo[next_state_int, next_active_player_idx, 1]
                        
                        if active_player == 1:
                            if f_score1 > best_active_val:
                                best_active_val = f_score1
                                best_vals_pair = (f_score1, f_score2)
                        else:
                            if f_score2 > best_active_val:
                                best_active_val = f_score2
                                best_vals_pair = (f_score1, f_score2)
                                
                    p1_matrix[w1_idx, w2_idx] = best_vals_pair[0]
                    p2_matrix[w1_idx, w2_idx] = best_vals_pair[1]
            
            # Simultaneous White Phase (Multiplayer Solitaire Logic)
            # P1 entirely ignores P2 and picks the action that maximizes P1's average score
            # P2 entirely ignores P1 and picks the action that maximizes P2's average score
            best_w1_idx = np.argmax(np.mean(p1_matrix, axis=1))
            best_w2_idx = np.argmax(np.mean(p2_matrix, axis=0))
            
            val1 = p1_matrix[best_w1_idx, best_w2_idx]
            val2 = p2_matrix[best_w1_idx, best_w2_idx]
            
            expected_score1 += dice['prob'] * val1
            expected_score2 += dice['prob'] * val2
            
        active_idx = 0 if active_player == 1 else 1
        shared_V_solo[state_int, active_idx, 0] = expected_score1
        shared_V_solo[state_int, active_idx, 1] = expected_score2


def run_solo_backward_induction():
    print("Loading Topological DAG...")
    dag = np.load('data/topological_dag.npy')
    
    from solvers.state_space_graph import get_state_depth
    print("Grouping states by depth for multi-processing...")
    depth_groups = defaultdict(list)
    for state in dag:
        depth_groups[get_state_depth(state)].append(state)
        
    depths_sorted = sorted(depth_groups.keys(), reverse=True)
    print(f"Total Unique States: {len(dag)} | Total Depth Levels: {len(depths_sorted)}")

    # Shared Memory Array
    shared_array_base = mp.Array('f', 1048576 * 2 * 2, lock=False)
    
    start_time = time.time()
    cores = mp.cpu_count()
    print(f"Firing up {cores} CPU cores for Solo DP...")
    
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
        for depth in depths_sorted:
            states_at_depth = depth_groups[depth]
            pool.map(solve_single_state_solo, states_at_depth)
            print(f"Completed Depth {depth:02d} | States solved: {len(states_at_depth)}")

    final_V_solo = np.frombuffer(shared_array_base, dtype=np.float32).reshape((1048576, 2, 2))
    
    os.makedirs('data', exist_ok=True)
    np.save('data/V_solo.npy', final_V_solo)
    
    end_time = time.time()
    print(f"\nSolo Backward Induction Complete in {round((end_time - start_time)/60, 2)} minutes!")
    print(f"Optimal Expected Solo Score for P1 (P1 active at start): {final_V_solo[0, 0, 0]:.4f}")

if __name__ == '__main__':
    run_solo_backward_induction()