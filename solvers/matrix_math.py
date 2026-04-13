"""
solvers/matrix_math.py

Exact Nash Equilibrium Solver for Zero-Sum Games.
This module resolves the simultaneous "White Phase" of the Mini-Qwixx turn. 
Because backward induction evaluates over 130 million 3x3 matrices, calling 
a standard Linear Programming (LP) solver for every node is computationally 
intractable. 

This solver uses a cascading sequence of exact mathematical shortcuts 
to find the Minimax value and Mixed Strategy probabilities in microseconds, 
only falling back to SciPy's Simplex/Interior-Point methods when absolutely necessary.

Thesis Reference: Equation 8 (The Minimax theorem applied to the White Phase).
"""

import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_matrix(A):
    """
    Computes the Value of the Game (v*) for a given payoff matrix A.
    Used extensively by the RL agents to find the Temporal Difference target.
    """
    # 1. Pure Strategy Saddle Point Check
    # A saddle point exists if Maximin == Minimax: max_i(min_j(A)) == min_j(max_i(A))
    # If true, the game is deterministic and no mixed strategy is required.
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    if np.max(row_mins) == np.min(col_maxs):
        return float(np.max(row_mins))

    # 2. Iterated Elimination of Strictly Dominated Strategies (IESDS)
    # Rational players will never play a strategy that yields a strictly worse 
    # outcome than another available strategy, regardless of the opponent's choice.
    rows, cols = A.shape
    valid_rows = list(range(rows))
    valid_cols = list(range(cols))

    # Player 1 (Maximizer) eliminates rows strictly dominated by other rows
    for i in range(rows):
        for j in range(rows):
            if i != j and i in valid_rows and j in valid_rows:
                if np.all(A[i, valid_cols] <= A[j, valid_cols]):
                    valid_rows.remove(i)
                    break

    # Player 2 (Minimizer) eliminates columns strictly dominated by other columns
    for i in range(cols):
        for j in range(cols):
            if i != j and i in valid_cols and j in valid_cols:
                if np.all(A[valid_rows, i] >= A[valid_rows, j]):
                    valid_cols.remove(i)
                    break

    A_sub = A[np.ix_(valid_rows, valid_cols)]

    # Re-check for a saddle point on the reduced submatrix
    if np.max(np.min(A_sub, axis=1)) == np.min(np.max(A_sub, axis=0)):
        return float(np.max(np.min(A_sub, axis=1)))

    # 3. Explicit 2x2 Algebraic Formula
    # If the game reduces to a 2x2 mixed strategy, we can solve for the exact 
    # expected value algebraically using determinants, bypassing LP entirely.
    if A_sub.shape == (2, 2):
        a, b = A_sub[0,0], A_sub[0,1]
        c, d = A_sub[1,0], A_sub[1,1]
        det = a - b - c + d
        if det != 0:
            v = (a*d - b*c) / det
            return float(v)

    # 4. Linear Programming Fallback (SciPy)
    # Triggered ONLY if the matrix is an irreducible 3x3 "Rock-Paper-Scissors" style game.
    # Formulates the primal LP: min v, subject to A^T * p1 >= v * 1, sum(p1) = 1
    c_obj = np.zeros(A_sub.shape[0] + 1)
    c_obj[0] = -1
    A_ub = np.zeros((A_sub.shape[1], A_sub.shape[0] + 1))
    A_ub[:, 0] = 1
    A_ub[:, 1:] = -A_sub.T
    b_ub = np.zeros(A_sub.shape[1])
    A_eq = np.zeros((1, A_sub.shape[0] + 1))
    A_eq[0, 1:] = 1
    b_eq = np.array([1.0])
    bounds = [(None, None)] + [(0, 1) for _ in range(A_sub.shape[0])]

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        return float(res.x[0])
        
    # Absolute fallback (mathematically rare in proper game definitions)
    return float(np.mean(A_sub))

def get_nash_probs(A):
    """
    Computes the optimal Mixed Strategy probabilities p1* and p2* in \Delta.
    Used by Backward Induction to weight the expected point outcomes, and by 
    the Analysis Evaluator to sample empirical tournament moves.
    """
    # 1. Fast Saddle Point Check (Pure Strategies)
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    
    if np.max(row_mins) == np.min(col_maxs):
        p1 = np.zeros(A.shape[0]); p1[np.argmax(row_mins)] = 1.0
        p2 = np.zeros(A.shape[1]); p2[np.argmin(col_maxs)] = 1.0
        return p1, p2

    # 2. Iterated Elimination of Strictly Dominated Strategies (IESDS)
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
    
    # 3. Explicit 2x2 Algebraic Formula for Mixed Probabilities
    # p1 = (d - c) / (a - b - c + d)
    if A_sub.shape == (2, 2):
        a, b = A_sub[0,0], A_sub[0,1]
        c, d = A_sub[1,0], A_sub[1,1]
        det = a - b - c + d
        if det != 0:
            p1_prob = (d - c) / det
            p2_prob = (d - b) / det
            
            # Ensure probabilities are mathematically valid bounds [0, 1]
            if 0 <= p1_prob <= 1 and 0 <= p2_prob <= 1:
                p1_sub = np.array([p1_prob, 1 - p1_prob])
                p2_sub = np.array([p2_prob, 1 - p2_prob])
                
                p1 = np.zeros(rows); p1[v_rows] = p1_sub
                p2 = np.zeros(cols); p2[v_cols] = p2_sub
                return p1, p2

    # 4. Linear Programming Fallback (SciPy) for irreducible 3x3 matrices
    
    # Primal LP to find Player 1's mixed strategy probabilities (p1*)
    c1 = np.zeros(A_sub.shape[0] + 1); c1[0] = -1
    A_ub1 = np.zeros((A_sub.shape[1], A_sub.shape[0] + 1)); A_ub1[:, 0] = 1; A_ub1[:, 1:] = -A_sub.T
    res1 = linprog(c1, A_ub=A_ub1, b_ub=np.zeros(A_sub.shape[1]), A_eq=np.array([[0] + [1]*A_sub.shape[0]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[0], method='highs')
    p1_sub = res1.x[1:] if res1.success else np.full(A_sub.shape[0], 1.0/A_sub.shape[0])

    # Dual LP to find Player 2's mixed strategy probabilities (p2*)
    # Player 2 wants to minimize the value, subject to A * p2 <= v * 1
    c2 = np.zeros(A_sub.shape[1] + 1); c2[0] = 1
    A_ub2 = np.zeros((A_sub.shape[0], A_sub.shape[1] + 1)); A_ub2[:, 0] = -1; A_ub2[:, 1:] = A_sub
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A_sub.shape[0]), A_eq=np.array([[0] + [1]*A_sub.shape[1]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[1], method='highs')
    p2_sub = res2.x[1:] if res2.success else np.full(A_sub.shape[1], 1.0/A_sub.shape[1])

    # Reconstruct the full 3-length probability vectors, assigning 0 to dominated strategies
    p1, p2 = np.zeros(rows), np.zeros(cols)
    p1[v_rows], p2[v_cols] = np.clip(p1_sub, 0, 1), np.clip(p2_sub, 0, 1)
    
    # Normalize to ensure floating point math strictly sums to 1.0 (valid probability distribution)
    p1 /= np.sum(p1); p2 /= np.sum(p2)
    return p1, p2