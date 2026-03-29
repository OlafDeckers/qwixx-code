import os
import time
import numpy as np
import multiprocessing as mp
from collections import defaultdict, Counter
from scipy.optimize import linprog

from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

shared_V_nash = None

def init_worker(shared_array):
    global shared_V_nash
    # We now track BOTH raw scores! Shape: [1,048,576 states] x [2 active players] x [2 scores]
    shared_V_nash = np.frombuffer(shared_array, dtype=np.float32).reshape((1048576, 2, 2))

def calculate_score(r_id, b_id, penalties):
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def get_unique_dice_combinations():
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

def get_nash_probs(A):
    """Fast solver that returns the (P1_probs, P2_probs) policy."""
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    
    # 1. Pure Strategy Saddle Point
    if np.max(row_mins) == np.min(col_maxs):
        p1 = np.zeros(A.shape[0]); p1[np.argmax(row_mins)] = 1.0
        p2 = np.zeros(A.shape[1]); p2[np.argmin(col_maxs)] = 1.0
        return p1, p2

    # 2. Iterative Domination Removal
    rows, cols = A.shape
    v_rows, v_cols = list(range(rows)), list(range(cols))
    for i in range(rows):
        for j in range(rows):
            if i != j and i in v_rows and j in v_rows and np.all(A[i, v_cols] <= A[j, v_cols]):
                v_rows.remove(i); break
    for i in range(cols):
        for j in range(cols):
            if i != j and i in v_cols and j in v_cols and np.all(A[v_rows, i] >= A[v_rows, j]):
                v_cols.remove(i); break

    A_sub = A[np.ix_(v_rows, v_cols)]
    
    # 3. 2x2 Algebraic Fallback
    if A_sub.shape == (2, 2):
        a, b = A_sub[0,0], A_sub[0,1]
        c, d = A_sub[1,0], A_sub[1,1]
        det = a - b - c + d
        if det != 0:
            p1_prob = (d - c) / det
            p2_prob = (d - b) / det
            if 0 <= p1_prob <= 1 and 0 <= p2_prob <= 1:
                p1_sub = np.array([p1_prob, 1 - p1_prob])
                p2_sub = np.array([p2_prob, 1 - p2_prob])
                p1 = np.zeros(rows); p1[v_rows] = p1_sub
                p2 = np.zeros(cols); p2[v_cols] = p2_sub
                return p1, p2

    # 4. SciPy Fallback (For 3x3 Mixed)
    c1 = np.zeros(A_sub.shape[0] + 1); c1[0] = -1
    A_ub1 = np.zeros((A_sub.shape[1], A_sub.shape[0] + 1)); A_ub1[:, 0] = 1; A_ub1[:, 1:] = -A_sub.T
    res1 = linprog(c1, A_ub=A_ub1, b_ub=np.zeros(A_sub.shape[1]), A_eq=np.array([[0] + [1]*A_sub.shape[0]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[0], method='highs')
    p1_sub = res1.x[1:] if res1.success else np.full(A_sub.shape[0], 1.0/A_sub.shape[0])

    c2 = np.zeros(A_sub.shape[1] + 1); c2[0] = 1
    A_ub2 = np.zeros((A_sub.shape[0], A_sub.shape[1] + 1)); A_ub2[:, 0] = -1; A_ub2[:, 1:] = A_sub
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A_sub.shape[0]), A_eq=np.array([[0] + [1]*A_sub.shape[1]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[1], method='highs')
    p2_sub = res2.x[1:] if res2.success else np.full(A_sub.shape[1], 1.0/A_sub.shape[1])

    p1, p2 = np.zeros(rows), np.zeros(cols)
    p1[v_rows], p2[v_cols] = np.clip(p1_sub, 0, 1), np.clip(p2_sub, 0, 1)
    p1 /= np.sum(p1); p2 /= np.sum(p2)
    return p1, p2

def solve_single_state(state_int):
    global shared_V_nash
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        shared_V_nash[state_int, 0, 0], shared_V_nash[state_int, 0, 1] = s1, s2
        shared_V_nash[state_int, 1, 0], shared_V_nash[state_int, 1, 1] = s1, s2
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
                            s1 = shared_V_nash[next_s, next_idx, 0]
                            s2 = shared_V_nash[next_s, next_idx, 1]
                            
                        diff = s1 - s2
                        if active_player == 1 and diff > best_diff:
                            best_diff, best_s1, best_s2 = diff, s1, s2
                        elif active_player == 2 and diff < best_diff:
                            best_diff, best_s1, best_s2 = diff, s1, s2
                            
                    payoff_matrix[w1_idx, w2_idx] = best_diff
                    s1_matrix[w1_idx, w2_idx] = best_s1
                    s2_matrix[w1_idx, w2_idx] = best_s2
            
            # The AI uses the ZERO-SUM matrix to find its policy probabilities
            p1_probs, p2_probs = get_nash_probs(payoff_matrix)
            
            # But we apply those probabilities to the RAW SCORE matrices to track exact points!
            val_s1 = p1_probs.T @ s1_matrix @ p2_probs
            val_s2 = p1_probs.T @ s2_matrix @ p2_probs
            
            expected_s1 += dice['prob'] * val_s1
            expected_s2 += dice['prob'] * val_s2
            
        active_idx = 0 if active_player == 1 else 1
        shared_V_nash[state_int, active_idx, 0] = expected_s1
        shared_V_nash[state_int, active_idx, 1] = expected_s2

def run_backward_induction():
    print("Loading Topological DAG...")
    dag = np.load('data/topological_dag.npy')
    from solvers.state_space_graph import get_state_depth
    
    depth_groups = defaultdict(list)
    for state in dag: depth_groups[get_state_depth(state)].append(state)
    depths_sorted = sorted(depth_groups.keys(), reverse=True)
    
    # 1048576 states * 2 players * 2 scores
    shared_array_base = mp.Array('f', 1048576 * 2 * 2, lock=False)
    
    start_time = time.time()
    cores = mp.cpu_count()
    print(f"Firing up {cores} CPU cores for EXACT Nash Tracking...")
    
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
        for depth in depths_sorted:
            states = depth_groups[depth]
            pool.map(solve_single_state, states)
            print(f"Completed Depth {depth:02d} | States solved: {len(states)}")

    final_V_nash = np.frombuffer(shared_array_base, dtype=np.float32).reshape((1048576, 2, 2))
    os.makedirs('data', exist_ok=True)
    np.save('data/V_nash.npy', final_V_nash)
    
    end_time = time.time()
    p1_s, p2_s = final_V_nash[0, 0, 0], final_V_nash[0, 0, 1]
    print(f"\nBackward Induction Complete in {round((end_time - start_time)/60, 2)} minutes!")
    print(f"Optimal Expected Nash Diff: {p1_s - p2_s:.4f} (P1: {p1_s:.2f}, P2: {p2_s:.2f})")

if __name__ == '__main__':
    run_backward_induction()