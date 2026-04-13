"""
tests/verify_solo.py

Solo Baseline Diagnostic Tool.
This script validates the "Multiplayer Solitaire" MDP solver (Thesis Equations 9-12).
It proves that under the Solo objective, the agents strictly maximize their own expected 
points and completely ignore zero-sum dynamics, leaving themselves exposed to 
adversarial traps.
"""

import numpy as np
from core.constants import ROW_ID_TO_COUNT
from core.environment import calculate_score
from core.state_encoder import encode_state

def run_solo_test(V_solo, p1_r, p1_b, p1_p, p2_r, p2_b, p2_p, test_name):
    state_int = encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p)
    
    p1_score = calculate_score(p1_r, p1_b, p1_p)
    p2_score = calculate_score(p2_r, p2_b, p2_p)
    
    # Extract values for when P1 is active
    p1_active_p1_expected = V_solo[state_int, 0, 0]
    p1_active_p2_expected = V_solo[state_int, 0, 1]
    
    # Extract values for when P2 is active
    p2_active_p1_expected = V_solo[state_int, 1, 0]
    p2_active_p2_expected = V_solo[state_int, 1, 1]
    
    print(f"\n{'='*65}")
    print(f"TEST: {test_name}")
    print(f"{'-'*65}")
    print(f"  P1: Red={p1_r:02d} ({ROW_ID_TO_COUNT[p1_r]} marks), Blue={p1_b:02d} ({ROW_ID_TO_COUNT[p1_b]} marks), Pens={p1_p}")
    print(f"  P2: Red={p2_r:02d} ({ROW_ID_TO_COUNT[p2_r]} marks), Blue={p2_b:02d} ({ROW_ID_TO_COUNT[p2_b]} marks), Pens={p2_p}")
    print(f"  CURRENT RAW SCORES: P1 = {p1_score}, P2 = {p2_score}")
    print(f"{'-'*65}")
    print(f"  IF P1 IS ACTIVE (Rolls Next):")
    print(f"    -> Expected P1 Final Score: {p1_active_p1_expected:.4f}")
    print(f"    -> Expected P2 Final Score: {p1_active_p2_expected:.4f}")
    print(f"  IF P2 IS ACTIVE (Rolls Next):")
    print(f"    -> Expected P1 Final Score: {p2_active_p1_expected:.4f}")
    print(f"    -> Expected P2 Final Score: {p2_active_p2_expected:.4f}")
    
    # Mathematical Symmetry Check for the Independent MDPs
    if p1_r == p2_r and p1_b == p2_b and p1_p == p2_p:
        # If P1 is active, P1's score should perfectly match P2's score when P2 is active
        sym_check_1 = abs(p1_active_p1_expected - p2_active_p2_expected) < 0.001
        sym_check_2 = abs(p1_active_p2_expected - p2_active_p1_expected) < 0.001
        if sym_check_1 and sym_check_2:
            print("  -> [PASS] Perfect Solo Symmetry Detected.")
        else:
            print("  -> [FAIL] Asymmetry Detected in Solo Array!")
            
    return p1_active_p1_expected, p1_active_p2_expected, p2_active_p1_expected, p2_active_p2_expected

if __name__ == '__main__':
    print("Loading Solo Array for Diagnostics...\n")
    try:
        V_solo = np.load('data/V_solo.npy')
    except FileNotFoundError:
        print("Could not find V_solo.npy. Run solo_backward_induction.py first.")
        exit()

    # 1. The Baseline
    run_solo_test(V_solo, 0,0,0, 0,0,0, "Empty Board (Checking Base Symmetry)")

    # 2 & 3. The Mirror Test 
    print("\n\n>>> THE SOLO MIRROR TEST <<<")
    v1_p1_act_p1, _, _, v1_p2_act_p2 = run_solo_test(V_solo, 10, 5, 0,  2, 1, 1, "P1 Dominating")
    v2_p1_act_p1, _, _, v2_p2_act_p2 = run_solo_test(V_solo, 2, 1, 1,  10, 5, 0, "P2 Dominating")
    
    if abs(v1_p1_act_p1 - v2_p2_act_p2) < 0.001:
        print("\n  [PASS] THE MIRROR TEST SUCCEEDED! The Solo AI treats both sides of the board identically.")
    else:
        print("\n  [FAIL] Mirror Test Failed.")

    # 4. The Terminal Test
    print("\n\n>>> TERMINAL STATE TEST <<<")
    run_solo_test(V_solo, 3, 2, 0,  0, 0, 3, "P2 has 3 Penalties (Game Over)")

    # 5. The "Greedy Ignorance" Test
    print("\n\n>>> THE GREEDY IGNORANCE TEST <<<")
    run_solo_test(V_solo, 10,0,0,  10,0,0, "Both players have 4 Red Marks")
    print("  ^ Look at the expected scores. Because both players are purely greedy,")
    print("    they will BOTH push their expected scores highly positive, ignoring the fact")
    print("    that in Nash, one of them would try to lock the other out.")