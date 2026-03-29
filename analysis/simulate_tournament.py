import numpy as np
import random
import time
import os
import multiprocessing as mp
from scipy.optimize import linprog

from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]
WHITE_ACTIONS = ['R', 'B', None]
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

# Global variables to hold all our different AI "Brains"
V_score_shared = None
V_win_shared = None
V_hybrid_5_shared = None
V_hybrid_25_shared = None
V_hybrid_50_shared = None

def init_worker():
    global V_score_shared, V_win_shared, V_hybrid_5_shared, V_hybrid_25_shared, V_hybrid_50_shared
    V_score_shared = np.load('data/V_nash.npy', mmap_mode='r')
    V_win_shared = np.load('data/V_nash_win_prob.npy', mmap_mode='r')
    V_hybrid_5_shared = np.load('data/V_nash_hybrid_5.npy', mmap_mode='r')
    V_hybrid_25_shared = np.load('data/V_nash_hybrid_25.npy', mmap_mode='r')
    V_hybrid_50_shared = np.load('data/V_nash_hybrid_50.npy', mmap_mode='r')
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

    rows, cols = A.shape
    v_rows, v_cols = list(range(rows)), list(range(cols))
    for i in range(rows):
        for j in range(rows):
            if i != j and i in v_rows and j in v_rows and np.all(A[i, v_cols] <= A[j, v_cols]):
                v_rows.remove(i); break
    for i in range(cols):
        for j in range(cols):
            if i != j and i in v_cols and j in v_cols and np.all(A[v_rows, i] >= A[v_rows, j]):
                v_cols.remove(i); break

    A_sub = A[np.ix_(v_rows, v_cols)]
    if A_sub.shape == (2, 2):
        a, b = A_sub[0,0], A_sub[0,1]
        c, d = A_sub[1,0], A_sub[1,1]
        det = a - b - c + d
        if det != 0:
            p1_prob = (d - c) / det
            p2_prob = (d - b) / det
            if 0 <= p1_prob <= 1 and 0 <= p2_prob <= 1:
                p1 = np.zeros(rows); p1[v_rows] = [p1_prob, 1 - p1_prob]
                p2 = np.zeros(cols); p2[v_cols] = [p2_prob, 1 - p2_prob]
                return p1, p2

    c1 = np.zeros(A_sub.shape[0] + 1); c1[0] = -1
    A_ub1 = np.zeros((A_sub.shape[1], A_sub.shape[0] + 1)); A_ub1[:, 0] = 1; A_ub1[:, 1:] = -A_sub.T
    res1 = linprog(c1, A_ub=A_ub1, b_ub=np.zeros(A_sub.shape[1]), A_eq=np.array([[0] + [1]*A_sub.shape[0]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[0], method='highs')
    p1_sub = res1.x[1:] if res1.success else np.full(A_sub.shape[0], 1.0/A_sub.shape[0])

    c2 = np.zeros(A_sub.shape[1] + 1); c2[0] = 1
    A_ub2 = np.zeros((A_sub.shape[0], A_sub.shape[1] + 1)); A_ub2[:, 0] = -1; A_ub2[:, 1:] = A_sub
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A_sub.shape[0]), A_eq=np.array([[0] + [1]*A_sub.shape[1]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A_sub.shape[1], method='highs')
    p2_sub = res2.x[1:] if res2.success else np.full(A_sub.shape[1], 1.0/A_sub.shape[1])

    p1, p2 = np.zeros(rows), np.zeros(cols)
    p1[v_rows], p2[v_cols] = np.clip(p1_sub, 0, 1), np.clip(p2_sub, 0, 1)
    return p1 / np.sum(p1), p2 / np.sum(p2)

def get_eval(state, active_idx, is_term, agent_type):
    # Dynamic evaluation based on the specific agent's psychological profile
    if is_term:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        diff = s1 - s2
        
        if agent_type == 'SCORE': return diff
        elif agent_type == 'WIN': return 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
        elif agent_type.startswith('HYBRID'):
            bonus = float(agent_type.split('_')[1])
            return (diff + bonus) if diff > 0 else ((diff - bonus) if diff < 0 else 0.0)

    if agent_type == 'SCORE': return V_score_shared[state, active_idx, 0] - V_score_shared[state, active_idx, 1]
    elif agent_type == 'WIN': return V_win_shared[state, active_idx]
    elif agent_type == 'HYBRID_5': return V_hybrid_5_shared[state, active_idx]
    elif agent_type == 'HYBRID_25': return V_hybrid_25_shared[state, active_idx]
    elif agent_type == 'HYBRID_50': return V_hybrid_50_shared[state, active_idx]

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
                        val = get_eval(ns, next_idx, term, current_eval_agent)
                        
                        if active_player == 1 and val > best_val: best_val = val; best_c = c
                        elif active_player == 2 and val < best_val: best_val = val; best_c = c

                    best_c_dict[(w1_idx, w2_idx)] = best_c
                    final_ns, final_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c)
                    
                    M_p1[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_p1)
                    M_p2[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_p2)

            p1_probs, _ = get_nash_probs(M_p1)
            _, p2_probs = get_nash_probs(M_p2)

            a_w1_idx = np.random.choice([0, 1, 2], p=p1_probs)
            a_w2_idx = np.random.choice([0, 1, 2], p=p2_probs)
            c_action = best_c_dict[(a_w1_idx, a_w2_idx)]

            state, _ = MiniQwixxEnv.step(state, active_player, dice, WHITE_ACTIONS[a_w1_idx], WHITE_ACTIONS[a_w2_idx], c_action)
            active_player = 2 if active_player == 1 else 1
            
    return stats

def run_tournament():
    games_per_matchup = 100000
    cores = mp.cpu_count()
    
    # We test the extremes against the baselines, and then against each other.
    matchups = [
        ('Conservative Hybrid', 'HYBRID_5', 'Pure Score Agent', 'SCORE'),
        ('Aggressive Hybrid', 'HYBRID_25', 'Pure Score Agent', 'SCORE'),
        ('Conservative Hybrid', 'HYBRID_5', 'Pure Win Agent', 'WIN'),
        ('Fanatic Hybrid', 'HYBRID_50', 'Pure Win Agent', 'WIN'),
        ('Conservative Hybrid', 'HYBRID_5', 'Fanatic Hybrid', 'HYBRID_50')
    ]

    print(f"\n" + "="*75)
    print(" THE SWEEP BATTLE ROYALE (" + str(games_per_matchup * len(matchups)) + " Games Total)")
    print("="*75)

    for name_a, tag_a, name_b, tag_b in matchups:
        print(f"\nSimulating {games_per_matchup} matches: [{name_a}] vs [{name_b}]...")
        start_t = time.time()

        games_per_core = [games_per_matchup // cores] * cores
        for i in range(games_per_matchup % cores): games_per_core[i] += 1
        
        args = [(n, tag_a, tag_b) for n in games_per_core]

        with mp.Pool(processes=cores, initializer=init_worker) as pool:
            results = pool.map(simulate_matchup_chunk, args)

        final_stats = {
            'a_wins': 0, 'b_wins': 0, 'ties': 0,
            'a_pts': 0, 'b_pts': 0,
            'a_margins': [], 'b_margins': []
        }
        
        for r in results:
            for k in ['a_wins', 'b_wins', 'ties', 'a_pts', 'b_pts']: final_stats[k] += r[k]
            final_stats['a_margins'].extend(r['a_margins'])
            final_stats['b_margins'].extend(r['b_margins'])

        a_win_pct = (final_stats['a_wins'] / games_per_matchup) * 100
        b_win_pct = (final_stats['b_wins'] / games_per_matchup) * 100
        tie_pct = (final_stats['ties'] / games_per_matchup) * 100
        
        avg_a_margin = sum(final_stats['a_margins']) / max(1, final_stats['a_wins'])
        avg_b_margin = sum(final_stats['b_margins']) / max(1, final_stats['b_wins'])

        print(f"  Result: [{name_a}] won {a_win_pct:.1f}% | [{name_b}] won {b_win_pct:.1f}% | Ties: {tie_pct:.1f}%")
        print(f"  Average Points: {name_a} = {final_stats['a_pts']/games_per_matchup:.2f} | {name_b} = {final_stats['b_pts']/games_per_matchup:.2f}")
        print(f"  When {tag_a} wins, margin is +{avg_a_margin:.2f}. When {tag_b} wins, margin is +{avg_b_margin:.2f}.")

if __name__ == '__main__':
    run_tournament()