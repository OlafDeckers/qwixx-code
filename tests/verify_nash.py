"""
tests/verify_nash.py

Zero-Sum Symmetry and Equilibrium Diagnostic Tool.
This script proves the mathematical soundness of the exact Dynamic Programming (DP) 
arrays. In a perfectly symmetric zero-sum game, swapping Player 1 and Player 2's 
board states must yield the exact inverse Expected Value. 

Furthermore, it empirically demonstrates the "First-Mover Advantage" inherent 
in the DAG structure: an empty board does not yield an expected value of 0.0, 
but rather a slight positive advantage for the active player.
"""

import numpy as np
from core.constants import ROW_ID_TO_COUNT
from core.environment import calculate_score
from core.state_encoder import encode_state

def run_test(V_nash, p1_r, p1_b, p1_p, p2_r, p2_b, p2_p, test_name):
    # Map the formal state tuple s to its scalar integer ID
    state_int = encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p)
    
    p1_score = calculate_score(p1_r, p1_b, p1_p)
    p2_score = calculate_score(p2_r, p2_b, p2_p)
    raw_diff = p1_score - p2_score
    
    # Extract the exact mathematical expectations E[S_1] and E[S_2] 
    # computed via Backward Induction when Player 1 is the active player.
    p1_active_s1 = V_nash[state_int, 0, 0]
    p1_active_s2 = V_nash[state_int, 0, 1]
    val_p1 = p1_active_s1 - p1_active_s2 # Expected point gap (V_SD)
    
    # Extract the expectations when Player 2 is the active player.
    p2_active_s1 = V_nash[state_int, 1, 0]
    p2_active_s2 = V_nash[state_int, 1, 1]
    val_p2 = p2_active_s1 - p2_active_s2 # Expected point gap (V_SD)
    
    print(f"\n{'='*65}")
    print(f"TEST: {test_name}")
    print(f"{'-'*65}")
    print(f"  P1: Red={p1_r:02d} ({ROW_ID_TO_COUNT[p1_r]} marks), Blue={p1_b:02d} ({ROW_ID_TO_COUNT[p1_b]} marks), Pens={p1_p}")
    print(f"  P2: Red={p2_r:02d} ({ROW_ID_TO_COUNT[p2_r]} marks), Blue={p2_b:02d} ({ROW_ID_TO_COUNT[p2_b]} marks), Pens={p2_p}")
    print(f"  RAW SCORE DIFF: {raw_diff} (P1: {p1_score}, P2: {p2_score})")
    print(f"{'-'*65}")
    print(f"  IF P1 ACTIVE:")
    print(f"    Expected Scores -> P1: {p1_active_s1:.2f} | P2: {p1_active_s2:.2f}")
    print(f"    Expected Diff   -> {val_p1:.4f}")
    print(f"  IF P2 ACTIVE:")
    print(f"    Expected Scores -> P1: {p2_active_s1:.2f} | P2: {p2_active_s2:.2f}")
    print(f"    Expected Diff   -> {val_p2:.4f}")
    
    # Zero-Sum Symmetry Check:
    # If the board is perfectly symmetric, the advantage P1 has when active 
    # must perfectly mirror the advantage P2 has when active.
    # Therefore, V(s, P1_active) + V(s, P2_active) must equal 0.
    if p1_r == p2_r and p1_b == p2_b and p1_p == p2_p:
        if abs(val_p1 + val_p2) < 0.001:
            print("  -> [PASS] Perfect Symmetry Detected.")
        else:
            print("  -> [FAIL] Asymmetry Detected!")
            
    return val_p1, val_p2

if __name__ == '__main__':
    print("Loading Nash Array for Extended Diagnostics...\n")
    V_nash = np.load('data/V_nash.npy')

    # 1. The Baseline
    run_test(V_nash, 0,0,0, 0,0,0, "Empty Board")

    # 2 & 3. The Mirror Test (Crucial for proving zero-sum logic)
    print("\n\n>>> THE MIRROR TEST <<<")
    v1_p1, v1_p2 = run_test(V_nash, 10, 5, 0,  2, 1, 1, "P1 Dominating")
    v2_p1, v2_p2 = run_test(V_nash, 2, 1, 1,  10, 5, 0, "P2 Dominating")
    
    if abs(v1_p1 + v2_p2) < 0.001 and abs(v1_p2 + v2_p1) < 0.001:
        print("\n  [PASS] THE MIRROR TEST SUCCEEDED! P1 winning is the exact mathematical inverse of P2 winning.")
    else:
        print("\n  [FAIL] Mirror Test Failed.")

    # 4 & 5. The Lock Test
    print("\n\n>>> THE DEFENSIVE LOCK TEST <<<")
    run_test(V_nash, 10,0,0,  5,0,0, "Un-Locked (P1 has 4 Red marks)")
    run_test(V_nash, 12,0,0,  5,0,0, "Locked (P1 has 4 Red marks + Lock Bonus)")
    print("  ^ Look at the P1 Active Expected Diff. It should jump higher in the Locked state because P2 is blocked!")

    # 6. Mutual Destruction
    print("\n\n>>> EXTREME SCENARIOS <<<")
    run_test(V_nash, 0,0,2,  0,0,2, "Mutual Destruction (Both have 2 pens)")
    
    # 7. Almost over by Locks
    run_test(V_nash, 13,0,0,  0,13,0, "One lock each (Next lock ends game)")