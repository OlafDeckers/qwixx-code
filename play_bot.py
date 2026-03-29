import numpy as np
import random
import time
from scipy.optimize import linprog

from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

def calculate_score(r_id, b_id, penalties):
    cr, cb = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: cr += 1
    if b_id >= 11: cb += 1
    return ((cr * (cr + 1)) // 2) + ((cb * (cb + 1)) // 2) - (3 * penalties)

def solve_nash_policy(A, player):
    """Returns the optimal mixed strategy probabilities [p1, p2, p3] for the given player."""
    # Fallback to pure strategy if saddle point exists
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    if np.max(row_mins) == np.min(col_maxs):
        if player == 1:
            policy = np.zeros(A.shape[0])
            policy[np.argmax(row_mins)] = 1.0
            return policy
        else:
            policy = np.zeros(A.shape[1])
            policy[np.argmin(col_maxs)] = 1.0
            return policy

    # Solve using Linear Programming for Mixed Strategy
    if player == 1:
        c = np.zeros(A.shape[0] + 1)
        c[0] = -1
        A_ub = np.zeros((A.shape[1], A.shape[0] + 1))
        A_ub[:, 0] = 1
        A_ub[:, 1:] = -A.T
        b_ub = np.zeros(A.shape[1])
        A_eq = np.zeros((1, A.shape[0] + 1))
        A_eq[0, 1:] = 1
        b_eq = np.array([1.0])
        bounds = [(None, None)] + [(0, 1) for _ in range(A.shape[0])]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success: return res.x[1:]
    else:
        c = np.zeros(A.shape[1] + 1)
        c[0] = 1
        A_ub = np.zeros((A.shape[0], A.shape[1] + 1))
        A_ub[:, 0] = -1
        A_ub[:, 1:] = A
        b_ub = np.zeros(A.shape[0])
        A_eq = np.zeros((1, A.shape[1] + 1))
        A_eq[0, 1:] = 1
        b_eq = np.array([1.0])
        bounds = [(None, None)] + [(0, 1) for _ in range(A.shape[1])]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success: return res.x[1:]
        
    return np.array([1/3, 1/3, 1/3]) # Fallback safety

def action_to_str(action):
    if action is None: return "Skip"
    if isinstance(action, tuple): return f"Cross {action[0]} (Die {action[1]})"
    return f"Cross {action} (White Sum)"

def play_game():
    print("Loading AI Brains (Nash Array)...")
    V_nash = np.load('data/V_nash.npy')
    
    state = 0
    active_player = 1
    turn_num = 1
    
    white_actions = ['R', 'B', None]
    color_actions = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]
    
    print("\n" + "="*50)
    print("    MINI-QWIXX: CLASH OF THE MATHEMATICAL GODS")
    print("="*50 + "\n")
    
    while True:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
        
        if p1_p >= 3 or p2_p >= 3 or (red_locked and blue_locked):
            break
            
        print(f"--- TURN {turn_num} | Player {active_player} is Active ---")
        
        # 1. Roll Dice
        dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
                'R': random.randint(1, 3), 'B': random.randint(1, 3)}
        w_sum = dice['W1'] + dice['W2']
        print(f"🎲 Rolled: White=[{dice['W1']}, {dice['W2']}] (Sum: {w_sum}) | Red=[{dice['R']}] | Blue=[{dice['B']}]")
        
        # 2. Build the Payoff Matrix for the White Phase
        payoff_matrix = np.zeros((3, 3), dtype=np.float32)
        next_active_idx = 1 if active_player == 1 else 0
        
        best_color_moves = {} # Store the best color move for every white combination
        
        for w1_idx, a_w1 in enumerate(white_actions):
            for w2_idx, a_w2 in enumerate(white_actions):
                best_val = -9999.0 if active_player == 1 else 9999.0
                best_c = None
                
                for a_c in color_actions:
                    next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                    if is_term:
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                        val = calculate_score(np1_r, np1_b, np1_p) - calculate_score(np2_r, np2_b, np2_p)
                    else:
                        val = V_nash[next_s, next_active_idx]
                        
                    if active_player == 1 and val > best_val:
                        best_val, best_c = val, a_c
                    elif active_player == 2 and val < best_val:
                        best_val, best_c = val, a_c
                        
                payoff_matrix[w1_idx, w2_idx] = best_val
                best_color_moves[(w1_idx, w2_idx)] = best_c

        # 3. Calculate Nash Policy and Sample White Actions
        p1_probs = solve_nash_policy(payoff_matrix, 1)
        p2_probs = solve_nash_policy(payoff_matrix, 2)
        
        # Clean up slight floating point errors for the random choice
        p1_probs = np.clip(p1_probs, 0, 1); p1_probs /= np.sum(p1_probs)
        p2_probs = np.clip(p2_probs, 0, 1); p2_probs /= np.sum(p2_probs)
        
        w1_idx = np.random.choice([0, 1, 2], p=p1_probs)
        w2_idx = np.random.choice([0, 1, 2], p=p2_probs)
        
        a_w1 = white_actions[w1_idx]
        a_w2 = white_actions[w2_idx]
        
        # 4. Deterministic Color Action
        a_c = best_color_moves[(w1_idx, w2_idx)]
        
        # 5. Print Decisions
        print(f"P1 White Action: {action_to_str(a_w1)}")
        print(f"P2 White Action: {action_to_str(a_w2)}")
        print(f"P{active_player} Color Action: {action_to_str(a_c)}")
        
        # 6. Apply Step
        state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
        
        # Print current score
        cp1_r, cp1_b, cp1_p, cp2_r, cp2_b, cp2_p = decode_state(state)
        s1 = calculate_score(cp1_r, cp1_b, cp1_p)
        s2 = calculate_score(cp2_r, cp2_b, cp2_p)
        print(f"Scoreboard -> P1: {s1} | P2: {s2} \n")
        
        active_player = 2 if active_player == 1 else 1
        turn_num += 1
        time.sleep(1) # Dramatic pause between turns

    print("="*50)
    print("GAME OVER!")
    print(f"Final Score -> P1: {s1} | P2: {s2}")
    if s1 > s2: print("🏆 Player 1 Wins!")
    elif s2 > s1: print("🏆 Player 2 Wins!")
    else: print("🤝 It's a Tie!")
    print("="*50)

if __name__ == '__main__':
    play_game() 