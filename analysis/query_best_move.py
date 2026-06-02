"""
Interactive Game State Advisor and Best-Response Calculator.
This script allows users to input any game state, turn profile, and dice rolls,
select from all 7 strategic objectives evaluated in the thesis, and computes
the exact mixed strategy for the White Phase or the best-response for the Color Phase.
"""

import os
import numpy as np
import random
from core.constants import COLOR_ACTIONS
from core.state_encoder import decode_state, encode_state
from core.environment import MiniQwixxEnv, calculate_score
from solvers.matrix_math import get_nash_probs

def get_int_input(prompt, min_val, max_val):
    while True:
        try:
            val = int(input(prompt))
            if min_val <= val <= max_val:
                return val
            print(f"Invalid range. Please enter an integer between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def main():
    print("=" * 75)
    print(" MINI-QWIXX INTERACTIVE ADVISOR: MULTI-OBJECTIVE STRATEGY SOLVER")
    print("=" * 75)

    # 1. Policy/Objective Selection Menu
    print("Select the Strategy Objective / Agent Policy to load:")
    print("  [1] Solo Strategy (Greedy Point Maximization)")
    print("  [2] Score Difference (Zero-Sum Point Margin Maximization)")
    print("  [3] Hybrid 5 (Score Diff + 5 Point Win Bonus)")
    print("  [4] Hybrid 10 (Score Diff + 10 Point Win Bonus)")
    print("  [5] Hybrid 25 (Score Diff + 25 Point Win Bonus)")
    print("  [6] Hybrid 50 (Score Diff + 50 Point Win Bonus)")
    print("  [7] Win Probability (Pure Binary Win Optimization)")
    
    choice = get_int_input("\nEnter choice (1-7): ", 1, 7)
    
    policies = {
        1: ("Solo Strategy", "data/V_solo.npy"),
        2: ("Score Difference", "data/V_nash.npy"),
        3: ("Hybrid 5", "data/V_nash_hybrid_5.npy"),
        4: ("Hybrid 10", "data/V_nash_hybrid_10.npy"),
        5: ("Hybrid 25", "data/V_nash_hybrid_25.npy"),
        6: ("Hybrid 50", "data/V_nash_hybrid_50.npy"),
        7: ("Win Probability", "data/V_nash_win_prob.npy")
    }
    
    policy_name, matrix_path = policies[choice]
    is_solo = (choice == 1)

    if not os.path.exists(matrix_path):
        print(f"\nError: Target matrix file '{matrix_path}' not found.")
        print("Ensure you are running the advisor from the project's root repository path.")
        return
    
    print(f"\nLoading exact backward-induction matrix for {policy_name}...")
    V_matrix = np.load(matrix_path, mmap_mode='r')
    print("Value function loaded successfully.\n")

    # 2. Gather Sheet Layout from User
    print("--- STEP 1: DEFINE THE MARKOV SHEET STATE ---")
    p1_r = get_int_input("Player 1 Red Row ID (0-10): ", 0, 10)
    p1_b = get_int_input("Player 1 Blue Row ID (0-10): ", 0, 10)
    p1_p = get_int_input("Player 1 Penalties (0-2): ", 0, 2)
    
    p2_r = get_int_input("Player 2 Red Row ID (0-10): ", 0, 10)
    p2_b = get_int_input("Player 2 Blue Row ID (0-10): ", 0, 10)
    p2_p = get_int_input("Player 2 Penalties (0-2): ", 0, 2)

    state = encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p)
    active_player = get_int_input("\nWho is the active player this turn? (1 or 2): ", 1, 2)
    user_perspective = get_int_input("Which player perspective do you want to optimize? (1 or 2): ", 1, 2)

    # 3. Gather Stochastic Dice Roll Layout
    print("\n--- STEP 2: ENTER THE DICE ROLL COMBINATION ---")
    w1 = get_int_input("White Die 1 Value (1-3): ", 1, 3)
    w2 = get_int_input("White Die 2 Value (1-3): ", 1, 3)
    r_die = get_int_input("Red Die Value (1-3): ", 1, 3)
    b_die = get_int_input("Blue Die Value (1-3): ", 1, 3)
    
    dice = {'W1': w1, 'W2': w2, 'R': r_die, 'B': b_die}
    white_sum = w1 + w2
    print(f"White Phase Sum: {white_sum}")

    white_actions = ['Nothing', 'Red', 'Blue']
    payoff_matrix = np.zeros((3, 3))
    next_idx = 1 if active_player == 1 else 0

    # 4. Construct Payoff Matrix via Backward Subgame Induction
    for i, p1_a in enumerate(white_actions):
        for j, p2_a in enumerate(white_actions):
            best_val = -9999 if active_player == 1 else 9999
            
            for a_c in COLOR_ACTIONS:
                ns, term = MiniQwixxEnv.step(state, active_player, dice, p1_a, p2_a, a_c)
                
                if term:
                    np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                    s1 = calculate_score(np1_r, np1_b, np1_p)
                    s2 = calculate_score(np2_r, np2_b, np2_p)
                    
                    if is_solo:
                        val = s1 if user_perspective == 1 else s2
                    elif choice == 7: # Pure Win Prob
                        val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                    elif choice == 2: # Score Difference
                        val = s1 - s2
                    else: # Hybrid Policies
                        beta = {3: 5, 4: 10, 5: 25, 6: 50}[choice]
                        wp = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        val = (s1 - s2) + beta * wp
                else:
                    if is_solo:
                        val = V_matrix[ns, next_idx, user_perspective - 1]
                    elif V_matrix.ndim == 2:
                        # Dynamically handle 2D scalar arrays (like Win Probability)
                        val = V_matrix[ns, next_idx]
                    else:
                        # Dynamically handle 3D tuple arrays (like Score Difference)
                        val = V_matrix[ns, next_idx, 0] - V_matrix[ns, next_idx, 1]
                
                if active_player == 1:
                    if val > best_val: best_val = val
                else:
                    if val < best_val: best_val = val
                        
            payoff_matrix[i, j] = best_val

    # 5. Output White Phase Strategy Profiles
    print("\n" + "="*55)
    print(f" PHASE 1: WHITE PHASE ACTION SELECTION ({policy_name.upper()})")
    print("="*55)
    
    if is_solo:
        if user_perspective == 1:
            row_sums = np.sum(payoff_matrix, axis=1)
            best_idx = np.argmax(row_sums)
        else:
            col_sums = np.sum(payoff_matrix, axis=0)
            best_idx = np.argmax(col_sums)
            
        print(f"Deterministic Solo Choice Recommendation:")
        print(f"  >>> Select action: **{white_actions[best_idx]}**")
    else:
        p1_probs, p2_probs = get_nash_probs(payoff_matrix)
        probs_to_show = p1_probs if user_perspective == 1 else p2_probs
        role = "Player 1 - Maximizer" if user_perspective == 1 else "Player 2 - Minimizer"
        
        print(f"Your Optimal Mixed Strategy Allocation ({role}):")
        for act, prob in zip(white_actions, probs_to_show):
            print(f"  * '{act}': {prob*100:.2f}%")

    # 6. Evaluate Sequential Color Phase Best-Response
    print("\n" + "="*55)
    print(" PHASE 2: COLOR PHASE Best-Response TARGET")
    print("="*55)
    print("Declare the actual selections made in Phase 1 to solve the subgame:")
    p1_chosen_white = white_actions[get_int_input("Player 1 White Action (0: Nothing, 1: Red, 2: Blue): ", 0, 2)]
    p2_chosen_white = white_actions[get_int_input("Player 2 White Action (0: Nothing, 1: Red, 2: Blue): ", 0, 2)]
    
    best_color_move = None
    if is_solo:
        target_val = -9999
    else:
        target_val = -9999 if user_perspective == 1 else 9999
        
    for a_c in COLOR_ACTIONS:
        ns, term = MiniQwixxEnv.step(state, active_player, dice, p1_chosen_white, p2_chosen_white, a_c)
        
        if term:
            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
            s1 = calculate_score(np1_r, np1_b, np1_p)
            s2 = calculate_score(np2_r, np2_b, np2_p)
            
            if is_solo:
                val = s1 if user_perspective == 1 else s2
            elif choice == 7:
                val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
            elif choice == 2:
                val = s1 - s2
            else:
                beta = {3: 5, 4: 10, 5: 25, 6: 50}[choice]
                wp = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                val = (s1 - s2) + beta * wp
        else:
            if is_solo:
                val = V_matrix[ns, next_idx, user_perspective - 1]
            elif V_matrix.ndim == 2:
                # Dynamically handle 2D scalar arrays here too!
                val = V_matrix[ns, next_idx]
            else:
                # Dynamically handle 3D tuple arrays here too!
                val = V_matrix[ns, next_idx, 0] - V_matrix[ns, next_idx, 1]
                
        if is_solo:
            if val > target_val:
                target_val = val
                best_color_move = a_c
        else:
            if user_perspective == 1:
                if val > target_val:
                    target_val = val
                    best_color_move = a_c
            else:
                if val < target_val:
                    target_val = val
                    best_color_move = a_c

    print(f"\n>>> COLOR PHASE RECOURSE ADVICE:")
    if active_player == user_perspective:
        print(f"  As the Active Player, your exact best choice is: **{best_color_move}**")
        metric_desc = "points" if (is_solo or choice == 2) else "objective units"
        print(f"  Expected value outcome: {target_val:.2f} {metric_desc}.")
    else:
        print(f"  You are passive this phase. If the active opponent plays optimally under this objective,")
        print(f"  they are mathematically expected to select: **{best_color_move}**")
    print("=" * 75)

if __name__ == '__main__':
    main()