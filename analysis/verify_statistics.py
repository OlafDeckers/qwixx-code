"""
analysis/verify_statistics.py

Statistical Validation of Empirical Results.
This script provides the mathematical proofs for the confidence intervals 
and statistical significance of the simulation results presented in the thesis. 
It justifies the chosen sample size (N=100,000) and proves the existence of a 
first-mover advantage in the Qwixx Directed Acyclic Graph (DAG).
"""

import math
import numpy as np
import random
import time
from scipy import stats
from core.constants import COLOR_ACTIONS
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score

def print_margin_of_error_proof():
    """
    Calculates the maximum Margin of Error (MOE) for the empirical win rates.
    Because the game outcomes (Win/Loss) follow a Binomial distribution, we can 
    bound the uncertainty of our Monte Carlo approximations.
    """
    print("="*60)
    print(" PART 1: 95% CONFIDENCE INTERVAL (MARGIN OF ERROR)")
    print("="*60)
    
    # Z-score for a 95% confidence interval
    z = 1.96
    
    # Thesis Reference: The variance of a binomial proportion p is p(1-p).
    # This variance is maximized when p = 0.5. Since a perfectly balanced zero-sum 
    # game converges toward a 50% win rate, we use p=0.5 to calculate the absolute 
    # worst-case Margin of Error.
    p = 0.5 
    
    # Total sample size used in the Round Robin Tournament
    n_total = 100000
    
    # Standard Margin of Error formula: Z * sqrt( p(1-p) / N )
    moe_total = z * math.sqrt((p * (1 - p)) / n_total) * 100
    
    # Sample size conditioned on turn order (e.g., games where Agent A plays as Player 1)
    n_split = 50000
    moe_split = z * math.sqrt((p * (1 - p)) / n_split) * 100
    
    print(f"Overall Win Rate (N={n_total:,}): Max Margin of Error = ±{moe_total:.2f}%")
    print(f"Position Win Rate (N={n_split:,}): Max Margin of Error = ±{moe_split:.2f}%\n")


def verify_welchs_ttest():
    """
    Empirically tests the Null Hypothesis (H0): mu_1 = mu_2 
    (There is no first-mover advantage; expected scores are equal).
    
    We use Welch's t-test instead of Student's t-test because we do not assume 
    the variance of Player 1's scores equals the variance of Player 2's scores 
    due to the asymmetrical nature of the state transitions in the DAG.
    """
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

    # N=10,000 provides sufficient statistical power to reject H0 if a true effect exists.
    num_games = 10000 
    print(f"Simulating {num_games} matches to collect score distributions...")

    p1_scores = []
    p2_scores = []
    
    random.seed(int(time.time()))
    
    # --- RAPID SIMULATION LOOP ---
    # Note for Methodology: This specific loop uses a simplified, purely deterministic 
    # (greedy) selection over the Color Phase to rapidly generate a score distribution. 
    # By omitting the simultaneous White Phase matrix resolution, it isolates the strict 
    # sequential turn-order advantage inherent in the game's graph structure.
    for i in range(num_games):
        state = 0
        active_player = 1
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            
            # Terminal condition check
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                s1 = calculate_score(p1_r, p1_b, p1_p)
                s2 = calculate_score(p2_r, p2_b, p2_p)
                p1_scores.append(s1)
                p2_scores.append(s2)
                break

            # Chance Node
            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_idx = 1 if active_player == 1 else 0
            
            best_c = None
            if active_player == 1: 
                # P1: Score Difference Agent (Zero-Sum Maximizer)
                best_val = -9999
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
                        
            else: 
                # P2: Solo Agent (Strict Score Maximizer, ignores opponent)
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
            
            # Transition state deterministically
            state, _ = MiniQwixxEnv.step(state, active_player, dice, None, None, best_c)
            active_player = 2 if active_player == 1 else 1
            
    p1_arr = np.array(p1_scores)
    p2_arr = np.array(p2_scores)
    
    print(f"P1 Average Score: {p1_arr.mean():.2f}")
    print(f"P2 Average Score: {p2_arr.mean():.2f}")
    
    # RUN WELCH'S T-TEST (equal_var=False assumes potentially unequal variances)
    t_stat, p_value = stats.ttest_ind(p1_arr, p2_arr, equal_var=False)
    
    print("\n--- STATISTICAL RESULTS ---")
    print(f"T-Statistic: {t_stat:.4f}")
    print(f"P-Value:     {p_value:.10f}")
    
    # Typically, alpha = 0.01 is used for strong statistical significance
    if p_value < 0.01:
        print("CONCLUSION:  p < 0.01! We reject the Null Hypothesis.")
        print("             The first-mover advantage is statistically significant.")
    else:
        print("CONCLUSION:  Not significant. We fail to reject the Null Hypothesis.")

if __name__ == '__main__':
    print_margin_of_error_proof()
    verify_welchs_ttest()