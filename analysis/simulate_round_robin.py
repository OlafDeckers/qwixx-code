import numpy as np
import random
import time
import os
import multiprocessing as mp
import itertools
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]
WHITE_ACTIONS = ['R', 'B', None]
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

# Global variables to hold all 7 Brains!
V_solo = None
V_score, V_win = None, None
V_h5, V_h10, V_h25, V_h50 = None, None, None, None

def init_worker():
    global V_solo, V_score, V_win, V_h5, V_h10, V_h25, V_h50
    V_solo = np.load('data/V_solo.npy', mmap_mode='r')
    V_score = np.load('data/V_nash.npy', mmap_mode='r')
    V_win = np.load('data/V_nash_win_prob.npy', mmap_mode='r')
    V_h10 = np.load('data/V_nash_hybrid.npy', mmap_mode='r') 
    V_h5 = np.load('data/V_nash_hybrid_5.npy', mmap_mode='r')
    V_h25 = np.load('data/V_nash_hybrid_25.npy', mmap_mode='r')
    V_h50 = np.load('data/V_nash_hybrid_50.npy', mmap_mode='r')
    np.random.seed(os.getpid() + int(time.time()))
    random.seed(os.getpid() + int(time.time()))

def calculate_score(r_id, b_id, penalties):
    cr, cb = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: cr += 1
    if b_id >= 11: cb += 1
    return ((cr * (cr + 1)) // 2) + ((cb * (cb + 1)) // 2) - (3 * penalties)

def get_nash_probs(A):
    row_mins = np.min(A, axis=1)
    col_maxs = np.max(A, axis=0)
    if np.max(row_mins) == np.min(col_maxs):
        p1 = np.zeros(A.shape[0]); p1[np.argmax(row_mins)] = 1.0
        p2 = np.zeros(A.shape[1]); p2[np.argmin(col_maxs)] = 1.0
        return p1, p2
    
    c1 = np.zeros(A.shape[0] + 1); c1[0] = -1
    A_ub1 = np.zeros((A.shape[1], A.shape[0] + 1)); A_ub1[:, 0] = 1; A_ub1[:, 1:] = -A.T
    res1 = linprog(c1, A_ub=A_ub1, b_ub=np.zeros(A.shape[1]), A_eq=np.array([[0] + [1]*A.shape[0]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A.shape[0], method='highs')
    p1 = res1.x[1:] if res1.success else np.full(A.shape[0], 1.0/A.shape[0])

    c2 = np.zeros(A.shape[1] + 1); c2[0] = 1
    A_ub2 = np.zeros((A.shape[0], A.shape[1] + 1)); A_ub2[:, 0] = -1; A_ub2[:, 1:] = A
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A.shape[0]), A_eq=np.array([[0] + [1]*A.shape[1]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A.shape[1], method='highs')
    p2 = res2.x[1:] if res2.success else np.full(A.shape[1], 1.0/A.shape[1])

    return np.clip(p1, 0, 1) / np.sum(np.clip(p1, 0, 1)), np.clip(p2, 0, 1) / np.sum(np.clip(p2, 0, 1))

def get_eval(state, active_idx, is_term, agent_type, evaluating_player):
    """
    evaluating_player: 1 or 2. 
    If agent is zero-sum, always returns from P1's perspective.
    If agent is SOLO, returns its OWN raw score. (For P2, we return -score, so minimizing it maximizes P2's score)
    """
    if is_term:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        
        if agent_type == 'SOLO': 
            return s1 if evaluating_player == 1 else -s2
            
        diff = s1 - s2
        if agent_type == 'SCORE': return diff
        elif agent_type == 'WIN': return 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
        elif agent_type.startswith('HYBRID'):
            bonus = float(agent_type.split('_')[1])
            return (diff + bonus) if diff > 0 else ((diff - bonus) if diff < 0 else 0.0)

    # Non-Terminal lookups
    if agent_type == 'SOLO':
        return V_solo[state, active_idx, 0] if evaluating_player == 1 else -V_solo[state, active_idx, 1]
    elif agent_type == 'SCORE': return V_score[state, active_idx, 0] - V_score[state, active_idx, 1]
    elif agent_type == 'WIN': return V_win[state, active_idx]
    elif agent_type == 'HYBRID_5': return V_h5[state, active_idx]
    elif agent_type == 'HYBRID_10': return V_h10[state, active_idx]
    elif agent_type == 'HYBRID_25': return V_h25[state, active_idx]
    elif agent_type == 'HYBRID_50': return V_h50[state, active_idx]

def simulate_matchup_chunk(args):
    num_games, agent_a_type, agent_b_type = args
    stats = {
        'a_wins': 0, 'b_wins': 0, 'ties': 0,
        'a_pts': 0, 'b_pts': 0,
        'a_margins': [], 'b_margins': []
    }

    for _ in range(num_games):
        state = 0
        active_player = 1
        
        # Coin flip to determine seating arrangement
        a_is_p1 = random.choice([True, False])
        agent_p1 = agent_a_type if a_is_p1 else agent_b_type
        agent_p2 = agent_b_type if a_is_p1 else agent_a_type

        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                s1 = calculate_score(p1_r, p1_b, p1_p)
                s2 = calculate_score(p2_r, p2_b, p2_p)
                a_pts = s1 if a_is_p1 else s2
                b_pts = s2 if a_is_p1 else s1
                
                stats['a_pts'] += a_pts
                stats['b_pts'] += b_pts
                
                if a_pts > b_pts: 
                    stats['a_wins'] += 1
                    stats['a_margins'].append(a_pts - b_pts)
                elif b_pts > a_pts: 
                    stats['b_wins'] += 1
                    stats['b_margins'].append(b_pts - a_pts)
                else: 
                    stats['ties'] += 1
                break

            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_idx = 1 if active_player == 1 else 0
            
            M_p1 = np.zeros((3, 3)) 
            M_p2 = np.zeros((3, 3))   
            best_c_dict = {}

            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_c = None
                    best_val = -9999 if active_player == 1 else 9999
                    
                    for c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, c)
                        current_eval_agent = agent_p1 if active_player == 1 else agent_p2
                        val = get_eval(ns, next_idx, term, current_eval_agent, active_player)
                        
                        if active_player == 1 and val > best_val: best_val = val; best_c = c
                        elif active_player == 2 and val < best_val: best_val = val; best_c = c

                    best_c_dict[(w1_idx, w2_idx)] = best_c
                    final_ns, final_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c)
                    
                    # Both players evaluate the board from their own perspective
                    M_p1[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_p1, 1)
                    M_p2[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_p2, 2)

            p1_probs, _ = get_nash_probs(M_p1)
            _, p2_probs = get_nash_probs(M_p2)
            
            idx_w1 = np.random.choice([0,1,2], p=p1_probs)
            idx_w2 = np.random.choice([0,1,2], p=p2_probs)
            c_action = best_c_dict[(idx_w1, idx_w2)]

            state, _ = MiniQwixxEnv.step(state, active_player, dice, WHITE_ACTIONS[idx_w1], WHITE_ACTIONS[idx_w2], c_action)
            active_player = 2 if active_player == 1 else 1
            
    return stats

def plot_heatmaps(win_matrix, score_matrix, margin_matrix, agents):
    # Fill self-play diagonals with NaN to make the heatmap cleaner
    np.fill_diagonal(win_matrix, np.nan)
    np.fill_diagonal(score_matrix, np.nan)
    np.fill_diagonal(margin_matrix, np.nan)
    
    os.makedirs('plots', exist_ok=True)
    
    display_names = ['Solo\n(Raw Pts)', 'Score\n(0 Bonus)', 'Hybrid\n(5 Bonus)', 'Hybrid\n(10 Bonus)', 
                     'Hybrid\n(25 Bonus)', 'Hybrid\n(50 Bonus)', 'Win Prob\n(Inf Bonus)']

    def format_and_save(title, filename):
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.ylabel("Agent Strategy (Row)", fontsize=12, fontweight='bold')
        plt.xlabel("Opponent Strategy (Column)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
        plt.close() # Close the figure so they don't overlap

    # --- Plot 1: Win Rate % ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(win_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".1f", cmap="RdYlGn", center=50.0, 
                cbar_kws={'label': 'Win Rate % (Row vs Col)'},
                linewidths=1, linecolor='black')
    format_and_save("Qwixx AI: Win Rate (%)", "heatmap_1_win_rate.png")

    # --- Plot 2: Average Points ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(score_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Blues", 
                cbar_kws={'label': 'Average Points Scored'},
                linewidths=1, linecolor='black')
    format_and_save("Qwixx AI: Average Points Scored", "heatmap_2_avg_points.png")

    # --- Plot 3: Margin of Victory ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(margin_matrix, index=display_names, columns=display_names), 
                annot=True, fmt=".2f", cmap="Purples", 
                cbar_kws={'label': 'Average Point Margin'},
                linewidths=1, linecolor='black')
    format_and_save("Qwixx AI: Average Margin of Victory", "heatmap_3_margin.png")

    print("\nSuccessfully generated and saved 3 separate heatmaps:")
    print("  1. plots/heatmap_1_win_rate.png")
    print("  2. plots/heatmap_2_avg_points.png")
    print("  3. plots/heatmap_3_margin.png")

def run_round_robin():
    # Full Spectrum of 7 Agents!
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    matchups = list(itertools.combinations(agents, 2))
    
    games_per_matchup = 100000 
    cores = mp.cpu_count()
    
    win_matrix = np.full((len(agents), len(agents)), 50.0)
    score_matrix = np.full((len(agents), len(agents)), 0.0)
    margin_matrix = np.full((len(agents), len(agents)), 0.0)

    print(f"\n" + "="*75)
    print(f" ROUND ROBIN TOURNAMENT: {len(agents)} AGENTS | {len(matchups)} UNIQUE MATCHUPS")
    print(f" Total Simulated Games: {len(matchups) * games_per_matchup}")
    print("="*75)

    for a_idx, b_idx in itertools.combinations(range(len(agents)), 2):
        tag_a, tag_b = agents[a_idx], agents[b_idx]
        print(f"Simulating {games_per_matchup} matches: [{tag_a}] vs [{tag_b}]...")
        
        games_per_core = [games_per_matchup // cores] * cores
        for i in range(games_per_matchup % cores): games_per_core[i] += 1
        args = [(n, tag_a, tag_b) for n in games_per_core]

        with mp.Pool(processes=cores, initializer=init_worker) as pool:
            results = pool.map(simulate_matchup_chunk, args)

        a_w, b_w, t = 0, 0, 0
        a_pts, b_pts = 0, 0
        a_margins, b_margins = [], []

        for r in results:
            a_w += r['a_wins']; b_w += r['b_wins']; t += r['ties']
            a_pts += r['a_pts']; b_pts += r['b_pts']
            a_margins.extend(r['a_margins']); b_margins.extend(r['b_margins'])

        # Win Rates (Ties = 0.5)
        a_win_rate = ((a_w + (0.5 * t)) / games_per_matchup) * 100
        b_win_rate = ((b_w + (0.5 * t)) / games_per_matchup) * 100
        win_matrix[a_idx][b_idx] = a_win_rate
        win_matrix[b_idx][a_idx] = b_win_rate
        
        # Average Points
        score_matrix[a_idx][b_idx] = a_pts / games_per_matchup
        score_matrix[b_idx][a_idx] = b_pts / games_per_matchup

        # Margin of Victory
        margin_matrix[a_idx][b_idx] = sum(a_margins) / max(1, a_w)
        margin_matrix[b_idx][a_idx] = sum(b_margins) / max(1, b_w)
        
        print(f"  -> {tag_a}: {a_win_rate:.1f}% | {tag_b}: {b_win_rate:.1f}% | Ties: {(t/games_per_matchup)*100:.1f}%")

    print("\nGenerating Master Heatmaps...")
    plot_heatmaps(win_matrix, score_matrix, margin_matrix, agents)

if __name__ == '__main__':
    run_round_robin()