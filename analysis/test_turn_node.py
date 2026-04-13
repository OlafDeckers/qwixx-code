import numpy as np
from scipy.optimize import linprog
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS, TOTAL_STATES

def get_nash_probs(A):
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    
    if np.max(row_mins) == np.min(col_maxs):
        p1 = np.zeros(A.shape[0]); p1[np.argmax(row_mins)] = 1.0
        p2 = np.zeros(A.shape[1]); p2[np.argmin(col_maxs)] = 1.0
        return p1, p2

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

def visualize_turn_node(state_int, mock_dice):
    print(f"--- TESTING STATE INT: {state_int} ---")
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    print(f"P1: Red={p1_r}, Blue={p1_b}, Penalties={p1_p}")
    print(f"P2: Red={p2_r}, Blue={p2_b}, Penalties={p2_p}")
    print(f"Dice Roll: {mock_dice}\n")
    
    np.random.seed(42) 
    mock_V_table = np.random.uniform(-1.0, 1.0, (TOTAL_STATES, 2))
    payoff_matrix = np.zeros((3, 3), dtype=np.float32)
    
    for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
        for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
            best_val = -9999.0
            
            for a_c in COLOR_ACTIONS:
                next_s, is_term = MiniQwixxEnv.step(state_int, 1, mock_dice, a_w1, a_w2, a_c)
                
                if is_term:
                    val = 1.0 if np.random.random() > 0.5 else -1.0
                else:
                    val = mock_V_table[next_s, 1] 
                
                if val > best_val:
                    best_val = val
                    
            payoff_matrix[w1_idx, w2_idx] = best_val
            
    print("--- FINAL 3x3 PAYOFF MATRIX (U1) ---")
    print("Columns: P2(Red), P2(Blue), P2(Pass)")
    for i, row in enumerate(payoff_matrix):
        print(f"P1({WHITE_ACTIONS[i]}):\t {row[0]:.4f} \t {row[1]:.4f} \t {row[2]:.4f}")
        
    print("\n--- SOLVING NASH EQUILIBRIUM ---")
    p1_probs, p2_probs = get_nash_probs(payoff_matrix) 
    
    print(f"P1 Mixed Strategy (Red, Blue, Pass): {[round(p, 4) for p in p1_probs]}")
    print(f"P2 Mixed Strategy (Red, Blue, Pass): {[round(p, 4) for p in p2_probs]}")
    
    turn_expected_value = p1_probs.T @ payoff_matrix @ p2_probs
    print(f"\nExpected Value of this Turn Node (Q_turn): {turn_expected_value:.4f}")

if __name__ == "__main__":
    test_dice = {'W1': 2, 'W2': 3, 'R': 1, 'B': 2}
    visualize_turn_node(state_int=0, mock_dice=test_dice)