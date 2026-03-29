import numpy as np
from solvers.matrix_math import solve_zero_sum_matrix

def run_matrix_test(matrix_data, expected_value, test_name, description):
    A = np.array(matrix_data, dtype=np.float32)
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"DESC: {description}")
    print(f"{'-'*60}")
    print("MATRIX:")
    print(A)
    
    # Run the solver
    calculated_value = solve_zero_sum_matrix(A)
    
    print(f"\nExpected Value:   {expected_value:.4f}")
    print(f"Calculated Value: {calculated_value:.4f}")
    
    if abs(calculated_value - expected_value) < 0.0001:
        print("-> [PASS] Mathematically Perfect.")
    else:
        print("-> [FAIL] Mismatch detected!")

if __name__ == '__main__':
    print("Initializing Game Theory Matrix Diagnostics...\n")

    # TEST 1: Pure Strategy (Saddle Point)
    # Row 0 minimum is 2. Col maxes are [4, 2, 2]. 
    # Max of row mins (2) == Min of col maxs (2). Saddle point is exactly 2.
    run_matrix_test(
        matrix_data=[[4, 2, 2], 
                     [1, 0, -1], 
                     [3, -2, 1]],
        expected_value=2.0,
        test_name="Pure Strategy (Fast Saddle Point)",
        description="Checks if the O(n) saddle point detection catches simple equilibria."
    )

    # TEST 2: Dominated Strategies -> 2x2 Reduction
    # Row 2 is strictly dominated by Row 0 (-1 <= 4, -1 <= 0, 5 <= 5). (Removed)
    # Col 2 is strictly worse for the minimizer than Col 0 (5 >= 4, 5 >= 0). (Removed)
    # Reduces to [[4, 0], [0, 4]], which has an expected value of 2.0.
    run_matrix_test(
        matrix_data=[[4, 0, 5], 
                     [0, 4, 5], 
                     [-1, -1, 5]],
        expected_value=2.0,
        test_name="Strict Domination",
        description="Forces the solver to prune bad rows/cols and reduce to a 2x2 formula."
    )

    # TEST 3: Matching Pennies (2x2 Formula)
    # The classic zero-sum game. Both players randomize 50/50. Expected value is 0.
    run_matrix_test(
        matrix_data=[[1, -1, -1], 
                     [-1, 1, -1],
                     [-1, -1, 1]], # We pad it with a dominated row/col to force 2x2 test
        expected_value=0.0,
        test_name="Matching Pennies (Padded)",
        description="Tests the explicit (ad-bc)/(a-b-c+d) algebraic fallback."
    )
    
    # We test a pure 2x2 asymmetric matrix just to be safe
    # P1 plays Row 0 (75%) and Row 1 (25%). Expected value is 0.75
    run_matrix_test(
        matrix_data=[[1, 0], 
                     [0, 3]],
        expected_value=0.75,
        test_name="Asymmetric 2x2",
        description="Tests the 2x2 algebraic formula with uneven weights."
    )

    # TEST 4: Rock-Paper-Scissors (SciPy Fallback)
    # Irreducible 3x3 matrix. All players play 1/3, 1/3, 1/3. Expected value is 0.
    run_matrix_test(
        matrix_data=[[ 0, -1,  1], 
                     [ 1,  0, -1], 
                     [-1,  1,  0]],
        expected_value=0.0,
        test_name="Rock-Paper-Scissors (SciPy Linprog)",
        description="Forces the code to fall back to the heavy SciPy linear programming solver."
    )