import numpy as np
from core.state_encoder import decode_state

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

def get_state_details(state_int):
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
    solo_p1 = V_solo[0, 0, 0]
    solo_p2 = V_solo[0, 0, 1]
    solo_diff_start = solo_p1 - solo_p2
    
    # NEW: Unpack the exact scores from the 3D Nash array
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

    # --- 2. SCANNING EVERY SINGLE CHOICE ---
    print("\n" + "-"*60)
    print("Scanning all 565,656 states for the maximum Strategic Gap...")
    
    gaps = []
    # We loop through all states. We only check states where P1 is active (Index 0)
    for state in range(len(V_nash)):
        # Calculate the Solo expectation for this state
        solo_expected_diff = V_solo[state, 0, 0] - V_solo[state, 0, 1]
        
        # NEW: Calculate the Nash expectation for this state
        nash_expected_diff = V_nash[state, 0, 0] - V_nash[state, 0, 1]
        
        # The gap is the absolute difference between the two paradigms
        gap = abs(nash_expected_diff - solo_expected_diff)
        
        # We only care about states that haven't ended yet
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        if p1_p < 3 and p2_p < 3 and gap > 0:
            gaps.append((gap, state, solo_expected_diff, nash_expected_diff))

    # Sort to find the biggest gaps
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