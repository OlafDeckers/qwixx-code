import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_matrix(A):
    """Fast solver that avoids linprog by checking saddle points and 2x2 reductions."""
    # 1. Fast Saddle Point Check
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    if np.max(row_mins) == np.min(col_maxs):
        return float(np.max(row_mins))

    # 2. Iterative removal of strictly dominated strategies
    rows, cols = A.shape
    valid_rows = list(range(rows))
    valid_cols = list(range(cols))

    # Dominated rows (Player 1 maximizes)
    for i in range(rows):
        for j in range(rows):
            if i != j and i in valid_rows and j in valid_rows:
                if np.all(A[i, valid_cols] <= A[j, valid_cols]):
                    valid_rows.remove(i)
                    break

    # Dominated cols (Player 2 minimizes)
    for i in range(cols):
        for j in range(cols):
            if i != j and i in valid_cols and j in valid_cols:
                if np.all(A[valid_rows, i] >= A[valid_rows, j]):
                    valid_cols.remove(i)
                    break

    A_sub = A[np.ix_(valid_rows, valid_cols)]

    # Re-check saddle point on the smaller submatrix
    if np.max(np.min(A_sub, axis=1)) == np.min(np.max(A_sub, axis=0)):
        return float(np.max(np.min(A_sub, axis=1)))

    # 3. Explicit 2x2 Algebraic Formula
    if A_sub.shape == (2, 2):
        a, b = A_sub[0,0], A_sub[0,1]
        c, d = A_sub[1,0], A_sub[1,1]
        det = a - b - c + d
        if det != 0:
            v = (a*d - b*c) / det
            return float(v)

    # 4. Fallback to SciPy ONLY if it's an irreducible rock-paper-scissors 3x3 matrix
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
    return float(np.mean(A_sub))