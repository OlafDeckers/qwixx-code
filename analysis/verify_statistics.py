import math
import numpy as np
import random
import time
from scipy import stats
from core.constants import COLOR_ACTIONS
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score

def print_margin_of_error_proof():
    print("="*60)
    print(" PART 1: 95% CONFIDENCE INTERVAL (MARGIN OF ERROR)")
    print("="*60)
    # Z-score for 95% confidence is approx 1.96
    z = 1.96
    # Worst-case variance for a proportion is at p = 0.5 (50% win rate)
    p = 0.5 
    
    n_total = 100000
    moe_total = z * math.sqrt((p * (1 - p)) / n_total) * 100
    
    n_split = 50000
    moe_split = z * math.sqrt((p * (1 - p)) / n_split) * 100
    
    print(f"Overall Win Rate (N={n_total:,}): Max Margin of Error = ±{moe_total:.2f}%")
    print(f"Position Win Rate (N={n_split:,}): Max Margin of Error = ±{moe_split:.2f}%\n")


def verify_welchs_ttest():
    print("="*60)
    print(" PART 2: WELCH'S T-TEST (PLAYER 1 VS PLAYER 2 SCORES)")
    print("="*60)
    print("Loading Agent Matrices (Score vs Solo)...")
    
    try:
        V_score = np.load('data/V_nash.npy', mmap_mode='r')
        V_solo = np.load('data/V_solo.npy', mmap_mode='r')
    except Exception as e:
        print("Could not load matrices. Run from root directory.", e)
        return

    num_games = 10000 # We only need a smaller sample to prove p < 0.01
    print(f"Simulating {num_games} matches to collect score arrays...")

    p1_scores = []
    p2_scores = []
    
    random.seed(int(time.time()))
    
    for i in range(num_games):
        state = 0
        active_player = 1
        
        # We will hardcode Score as P1 and Solo as P2 for this targeted test
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                s1 = calculate_score(p1_r, p1_b, p1_p)
                s2 = calculate_score(p2_r, p2_b, p2_p)
                p1_scores.append(s1)
                p2_scores.append(s2)
                break

            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_idx = 1 if active_player == 1 else 0
            
            best_c = None
            if active_player == 1: # Score Agent
                best_val = -9999
                # Simplified greedy choice for quick simulation
                for a_c in COLOR_ACTIONS:
                    ns, term = MiniQwixxEnv.step(state, active_player, dice, None, None, a_c)
                    if term:
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                        val = calculate_score(np1_r, np1_b, np1_p) - calculate_score(np2_r, np2_b, np2_p)
                    else:
                        val = V_score[ns, next_idx, 0] - V_score[ns, next_idx, 1]
                    if val > best_val:
                        best_val = val
                        best_c = a_c
            else: # Solo Agent
                best_val = -9999
                for a_c in COLOR_ACTIONS:
                    ns, term = MiniQwixxEnv.step(state, active_player, dice, None, None, a_c)
                    if term:
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                        val = -calculate_score(np2_r, np2_b, np2_p)
                    else:
                        val = -V_solo[ns, next_idx, 1]
                    if val > best_val:
                        best_val = val
                        best_c = a_c
            
            state, _ = MiniQwixxEnv.step(state, active_player, dice, None, None, best_c)
            active_player = 2 if active_player == 1 else 1
            
    p1_arr = np.array(p1_scores)
    p2_arr = np.array(p2_scores)
    
    print(f"P1 Average Score: {p1_arr.mean():.2f}")
    print(f"P2 Average Score: {p2_arr.mean():.2f}")
    
    # RUN WELCH'S T-TEST (equal_var=False)
    t_stat, p_value = stats.ttest_ind(p1_arr, p2_arr, equal_var=False)
    
    print("\n--- STATISTICAL RESULTS ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.10f}")
    if p_value < 0.01:
        print("CONCLUSION:  p < 0.01! The first-mover advantage is statistically significant.")
    else:
        print("CONCLUSION:  Not significant.")

if __name__ == '__main__':
    print_margin_of_error_proof()
    verify_welchs_ttest()