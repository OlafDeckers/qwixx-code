"""
analysis/test_turn_node.py

Visualizer for Subgame Perfect Equilibrium (Thesis Equations 7 & 8).
This testing script isolates a single non-terminal state and a fixed stochastic 
chance node (dice roll). It visibly computes the 3x3 Payoff Matrix by collapsing 
the sequential Color Phase, and prints the mixed strategy probabilities found via 
the Minimax theorem.
"""
import numpy as np
from scipy.optimize import linprog
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS, TOTAL_STATES
from solvers.matrix_math import get_nash_probs


def visualize_turn_node(state_int, mock_dice):
    """
    Simulates the evaluation of Q_turn(s, d) for a single fixed dice roll.
    """
    print(f"--- TESTING STATE INT: {state_int} ---")
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    print(f"P1: Red={p1_r}, Blue={p1_b}, Penalties={p1_p}")
    print(f"P2: Red={p2_r}, Blue={p2_b}, Penalties={p2_p}")
    print(f"Dice Roll: {mock_dice}\n")
    
    # Mocking the DP array with uniform noise to simulate V*(s') values
    np.random.seed(42) 
    mock_V_table = np.random.uniform(-1.0, 1.0, (TOTAL_STATES, 2))
    
    # Initialize the zero-sum payoff matrix for the simultaneous White Phase
    payoff_matrix = np.zeros((3, 3), dtype=np.float32)
    
    for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
        for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
            best_val = -9999.0
            
            # Thesis Eq 7: Maximize over the Sequential Color Phase (U_1)
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
    
    # Thesis Eq 8: Applying Minimax to find mixed strategies p1* and p2*
    p1_probs, p2_probs = get_nash_probs(payoff_matrix) 
    
    print(f"P1 Mixed Strategy (Red, Blue, Pass): {[round(p, 4) for p in p1_probs]}")
    print(f"P2 Mixed Strategy (Red, Blue, Pass): {[round(p, 4) for p in p2_probs]}")
    
    # Calculating the expected value under the mixed strategy distribution
    turn_expected_value = p1_probs.T @ payoff_matrix @ p2_probs
    print(f"\nExpected Value of this Turn Node (Q_turn): {turn_expected_value:.4f}")

if __name__ == "__main__":
    # Deterministic test inputs
    test_dice = {'W1': 2, 'W2': 3, 'R': 1, 'B': 2}
    visualize_turn_node(state_int=0, mock_dice=test_dice)