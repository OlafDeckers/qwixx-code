"""
tests/strategic_gap.py

Empirical Divergence: Solo Optimization vs. Zero-Sum Equilibrium.
This script scans the entire state space graph |S| to identify the specific states 
where the independent MDP (Solo) assumption mathematically diverges the most from the 
Zero-Sum Markov Game (Nash) reality. 

These identified states represent "Greedy Traps"—configurations where acting selfishly 
results in the largest punitive loss of Win Probability or Expected Points.
"""

import numpy as np
from core.constants import ROW_ID_TO_COUNT
from core.state_encoder import decode_state

def get_state_details(state_int):
    """String formatter for mathematical states."""
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    return (f"P1: Red={ROW_ID_TO_COUNT[p1_r]} marks, Blue={ROW_ID_TO_COUNT[p1_b]} marks, Pens={p1_p} | "
            f"P2: Red={ROW_ID_TO_COUNT[p2_r]} marks, Blue={ROW_ID_TO_COUNT[p2_b]} marks, Pens={p2_p}")

def calculate_strategic_gap():
    print("Loading Dynamic Programming Arrays...")
    try:
        V_solo = np.load('data/V_solo.npy')
        V_nash = np.load('data/V_nash.npy')
    except FileNotFoundError:
        print("Error: Could not find the .npy files.")
        return

    # --- 1. BASELINE START STATE ---
    # Expected advantage at s_0
    solo_p1 = V_solo[0, 0, 0]
    solo_p2 = V_solo[0, 0, 1]
    solo_diff_start = solo_p1 - solo_p2
    
    # Exact Nash equilibrium advantage at s_0
    nash_p1 = V_nash[0, 0, 0]
    nash_p2 = V_nash[0, 0, 1]
    nash_diff_start = nash_p1 - nash_p2
    
    print("\n" + "="*60)
    print("     THE STRATEGIC GAP (VALUE OF DEFENSIVE PLAY)")
    print("="*60)
    print("\n[STARTING BOARD] - State 0")
    print(f"  Expected Solo Diff (Greedy):   {solo_diff_start:.4f} pts")
    print(f"  Expected Nash Diff (Defense):  {nash_diff_start:.4f} pts")
    print(f"  Strategic Gap at Start:        {abs(nash_diff_start - solo_diff_start):.4f} pts")

    # --- 2. SCANNING THE ENTIRE DIRECTED ACYCLIC GRAPH ---
    print("\n" + "-"*60)
    print("Scanning all 565,656 states for the maximum Strategic Gap...")
    
    gaps = []
    # Iterate through all s \in S. Check states where P1 is active (Index 0).
    for state in range(len(V_nash)):
        # Calculate the Solo E[S_1 - S_2] assumption
        solo_expected_diff = V_solo[state, 0, 0] - V_solo[state, 0, 1]
        
        # Calculate the true Nash Equilibrium V*(s)
        nash_expected_diff = V_nash[state, 0, 0] - V_nash[state, 0, 1]
        
        # The divergence metric
        gap = abs(nash_expected_diff - solo_expected_diff)
        
        # Filter out terminal states (where the gap mathematically converges to 0)
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        if p1_p < 3 and p2_p < 3 and gap > 0:
            gaps.append((gap, state, solo_expected_diff, nash_expected_diff))

    # Sort descending to find the largest divergences
    gaps.sort(reverse=True, key=lambda x: x[0])

    print("\n[TOP 3 DEADLIEST 'GREEDY TRAPS']")
    print("These are the board states where ignoring the Nash equilibrium costs you the most points:\n")
    
    for i in range(3):
        gap, state, s_diff, n_diff = gaps[i]
        print(f"Rank {i+1}: Gap of {gap:.4f} points!")
        print(f"  Board: {get_state_details(state)}")
        print(f"  If played Greedily (Solo): AI thinks P1 advantage is {s_diff:.2f}")
        print(f"  If played Defensively (Nash): True P1 advantage is {n_diff:.2f}\n")

if __name__ == '__main__':
    calculate_strategic_gap()