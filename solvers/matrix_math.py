"""
solvers/matrix_math.py

Exact Nash Equilibrium Solver for Zero-Sum Games.
This module resolves the simultaneous "White Phase" of the Mini-Qwixx turn. 
Because backward induction evaluates over 130 million 3x3 matrices, calling 
a standard Linear Programming (LP) solver for every node is computationally 
intractable. 

This solver uses a cascading sequence of exact mathematical shortcuts 
to find the Minimax value and Mixed Strategy probabilities in microseconds.
[Computational Upgrade: Completely refactored with Numba JIT and analytical 
 3x3 algebraic fallbacks to bypass SciPy's LP overhead entirely.]

Thesis Reference: Equation 8 (The Minimax theorem applied to the White Phase).
"""

import numpy as np
from numba import njit

@njit(nogil=True)
def get_nash_probs(A):
    """
    Computes the optimal Mixed Strategy probabilities p1* and p2* in \Delta.
    Used by Backward Induction to weight the expected point outcomes, and by 
    the Analysis Evaluator to sample empirical tournament moves.
    """
    rows = A.shape[0]
    cols = A.shape[1]
    
    # 1. Fast Saddle Point Check (Pure Strategies)
    row_mins = np.zeros(rows)
    for i in range(rows):
        row_mins[i] = np.min(A[i, :])
    col_maxs = np.zeros(cols)
    for j in range(cols):
        col_maxs[j] = np.max(A[:, j])
        
    max_row_min = np.max(row_mins)
    min_col_max = np.min(col_maxs)
    
    if max_row_min == min_col_max:
        p1 = np.zeros(rows)
        p1[np.argmax(row_mins)] = 1.0
        p2 = np.zeros(cols)
        p2[np.argmin(col_maxs)] = 1.0
        return p1, p2

    # 2. Iterated Elimination of Strictly Dominated Strategies (IESDS)
    # Refactored for Numba using high-speed boolean masking
    v_rows = np.ones(rows, dtype=np.bool_)
    v_cols = np.ones(cols, dtype=np.bool_)
    
    changed = True
    while changed:
        changed = False
        for i in range(rows):
            if not v_rows[i]: continue
            for j in range(rows):
                if i == j or not v_rows[j]: continue
                dominated = True
                for c in range(cols):
                    if v_cols[c] and A[i, c] >= A[j, c]:
                        dominated = False
                        break
                if dominated:
                    v_rows[i] = False
                    changed = True
                    break
        
        for i in range(cols):
            if not v_cols[i]: continue
            for j in range(cols):
                if i == j or not v_cols[j]: continue
                dominated = True
                for r in range(rows):
                    if v_rows[r] and A[r, i] <= A[r, j]:
                        dominated = False
                        break
                if dominated:
                    v_cols[i] = False
                    changed = True
                    break

    n_r = np.sum(v_rows)
    n_c = np.sum(v_cols)

    # 3. Explicit 2x2 Algebraic Formula for Mixed Probabilities
    # p1 = (d - c) / (a - b - c + d)
    if n_r == 2 and n_c == 2:
        r_idx = np.where(v_rows)[0]
        c_idx = np.where(v_cols)[0]
        a = A[r_idx[0], c_idx[0]]
        b = A[r_idx[0], c_idx[1]]
        c = A[r_idx[1], c_idx[0]]
        d = A[r_idx[1], c_idx[1]]
        
        det = a - b - c + d
        if det != 0:
            p1_prob = (d - c) / det
            p2_prob = (d - b) / det
            
            # Ensure probabilities are mathematically valid bounds [0, 1]
            if 0 <= p1_prob <= 1 and 0 <= p2_prob <= 1:
                p1 = np.zeros(rows)
                p2 = np.zeros(cols)
                p1[r_idx[0]] = p1_prob
                p1[r_idx[1]] = 1.0 - p1_prob
                p2[c_idx[0]] = p2_prob
                p2[c_idx[1]] = 1.0 - p2_prob
                return p1, p2

    # 4. Fallback: Analytical 3x3 Full Support (Bypassing Linear Programming entirely)
    if n_r == 3 and n_c == 3:
        # Shift A to strictly positive to natively avoid singular division errors
        shift = np.min(A) - 1.0
        A_pos = A - shift
        
        try:
            x = np.linalg.solve(A_pos.T, np.ones(3))
            y = np.linalg.solve(A_pos, np.ones(3))
            
            if np.all(x > -1e-9) and np.all(y > -1e-9):
                p1 = x / np.sum(x)
                p2 = y / np.sum(y)
                return np.clip(p1, 0, 1), np.clip(p2, 0, 1)
        except:
            pass # Fails safely if matrix is singular

    # 5. Final Fallback: Check all 2x2 subgames natively
    # Handles irreducible 3x3 matrices where the Nash Equilibrium is a 2x2 support
    r_act = np.where(v_rows)[0]
    c_act = np.where(v_cols)[0]
    for r1 in range(n_r):
        for r2 in range(r1 + 1, n_r):
            for c1 in range(n_c):
                for c2 in range(c1 + 1, n_c):
                    row_1, row_2 = r_act[r1], r_act[r2]
                    col_1, col_2 = c_act[c1], c_act[c2]
                    
                    a = A[row_1, col_1]
                    b = A[row_1, col_2]
                    c = A[row_2, col_1]
                    d = A[row_2, col_2]
                    
                    det = a - b - c + d
                    if det != 0:
                        p1_prob = (d - c) / det
                        p2_prob = (d - b) / det
                        
                        if 0 <= p1_prob <= 1 and 0 <= p2_prob <= 1:
                            v_sub = (a * d - b * c) / det
                            p1_test = np.zeros(rows)
                            p1_test[row_1] = p1_prob
                            p1_test[row_2] = 1.0 - p1_prob
                            
                            p2_test = np.zeros(cols)
                            p2_test[col_1] = p2_prob
                            p2_test[col_2] = 1.0 - p2_prob
                            
                            # Verify if the 2x2 subgame is a global equilibrium
                            exp_p1 = np.zeros(rows)
                            exp_p2 = np.zeros(cols)
                            for i in range(rows):
                                for j in range(cols):
                                    exp_p1[i] += A[i, j] * p2_test[j]
                                    exp_p2[j] += p1_test[i] * A[i, j]
                                    
                            if np.max(exp_p2) <= v_sub + 1e-6 and np.min(exp_p1) >= v_sub - 1e-6:
                                return p1_test, p2_test

    # Absolute Fallback (mathematically rare in proper zero-sum definitions)
    p1 = np.zeros(rows); p1[v_rows] = 1.0 / n_r
    p2 = np.zeros(cols); p2[v_cols] = 1.0 / n_c
    return p1, p2

@njit(nogil=True)
def solve_zero_sum_matrix(A):
    """
    Computes the Value of the Game (v*) for a given payoff matrix A.
    Used extensively by the RL agents to find the Temporal Difference target.
    """
    # Re-use the exact Numba-compiled probability solver directly
    p1, p2 = get_nash_probs(A)
    
    # Calculate expected value v* = p1^T * A * p2 using ultra-fast nested C-loops
    v = 0.0
    rows = A.shape[0]
    cols = A.shape[1]
    for i in range(rows):
        for j in range(cols):
            v += p1[i] * A[i, j] * p2[j]
            
    return float(v)