import numpy as np
from core.state_encoder import decode_state

# Define the array locally instead of importing it
ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

def get_state_details(state_int):
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    return (f"P1: Red={ROW_ID_TO_COUNT[p1_r]}, Blue={ROW_ID_TO_COUNT[p1_b]}, Pens={p1_p} | "
            f"P2: Red={ROW_ID_TO_COUNT[p2_r]}, Blue={ROW_ID_TO_COUNT[p2_b]}, Pens={p2_p}")

def hunt_for_hail_marys():
    print("Loading both DP Arrays for cross-examination...")
    try:
        # V_nash is 3D: [states, 2 active players, 2 scores]
        V_score_diff = np.load('data/V_nash.npy')
        # V_win is 2D: [states, 2 active players]
        V_win_prob = np.load('data/V_nash_win_prob.npy')
    except FileNotFoundError:
        print("Waiting for win_prob_backward_induction to finish...")
        return

    print("Scanning 565,656 states for divergent behavior...\n")
    
    hail_marys = []
    
    for state in range(len(V_score_diff)):
        # We only check states where P1 is active
        
        # 1. How does the Score-Difference agent view this state?
        p1_score = V_score_diff[state, 0, 0]
        p2_score = V_score_diff[state, 0, 1]
        expected_diff = p1_score - p2_score
        
        # 2. How does the Win-Probability agent view this state?
        # Convert the [-1.0 to 1.0] margin into a clean 0% to 100% win rate
        win_margin = V_win_prob[state, 0]
        win_pct = ((win_margin + 1) / 2) * 100
        
        # 3. THE HAIL MARY FILTER:
        # We are looking for a state where P1 is mathematically getting crushed (Expected loss > 10 points)
        # BUT the Win-Probability agent still thinks it has a tiny > 1% chance to win.
        if expected_diff < -10.0 and 1.0 < win_pct < 15.0:
            
            # Ensure the game isn't already over
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p < 3 and p2_p < 3:
                hail_marys.append({
                    'state': state,
                    'diff': expected_diff,
                    'win_pct': win_pct
                })

    # Sort them by the worst score difference to find the most extreme examples
    hail_marys.sort(key=lambda x: x['diff'])

    print("="*65)
    print(" THE 'HAIL MARY' PROOF: WIN PROBABILITY VS SCORE DIFFERENCE")
    print("="*65)
    print("These are states where a Win-Probability optimizer will take")
    print("irrational risks, actively destroying its own score just to")
    print("chase a tiny, statistically improbable victory.\n")
    
    # Print the top 3 most extreme divergences
    for i in range(min(3, len(hail_marys))):
        data = hail_marys[i]
        print(f"EXTREME DIVERGENCE #{i+1}")
        print(f"  Board: {get_state_details(data['state'])}")
        print(f"  Score Diff Agent: 'Play safe. We will lose by {abs(data['diff']):.2f} points.'")
        print(f"  Win Prob Agent:   'Take the risk! We have a {data['win_pct']:>5.2f}% chance to win!'\n")

if __name__ == '__main__':
    hunt_for_hail_marys()