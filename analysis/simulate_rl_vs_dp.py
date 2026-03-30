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

# We use the Hybrid-25 rules since that is what the RL agent was trained on
WIN_BONUS = 25.0

V_dp_shared = None
V_rl_shared = None

def init_worker():
    global V_dp_shared, V_rl_shared
    # Load the Exact Mathematical DP array
    V_dp_shared = np.load('data/V_nash_hybrid_25.npy', mmap_mode='r')
    # Load the AI's Self-Taught RL array
    V_rl_shared = np.load('data/V_rl_minimax_hybrid.npy', mmap_mode='r')
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
    p1_sub = res1.x[1:] if res1.success else np.full(A.shape[0], 1.0/A.shape[0])

    c2 = np.zeros(A.shape[1] + 1); c2[0] = 1
    A_ub2 = np.zeros((A.shape[0], A.shape[1] + 1)); A_ub2[:, 0] = -1; A_ub2[:, 1:] = A
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(A.shape[0]), A_eq=np.array([[0] + [1]*A.shape[1]]), b_eq=np.array([1.0]), bounds=[(None, None)] + [(0, 1)]*A.shape[1], method='highs')
    p2_sub = res2.x[1:] if res2.success else np.full(A.shape[1], 1.0/A.shape[1])

    p1, p2 = np.clip(p1_sub, 0, 1), np.clip(p2_sub, 0, 1)
    return p1 / np.sum(p1), p2 / np.sum(p2)

def get_eval(state, active_idx, is_term, agent_type):
    if is_term:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        diff = s1 - s2
        return (diff + WIN_BONUS) if diff > 0 else ((diff - WIN_BONUS) if diff < 0 else 0.0)

    if agent_type == 'DP': return V_dp_shared[state, active_idx]
    elif agent_type == 'RL': return V_rl_shared[state, active_idx]

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
        
        # Randomize Seating to neutralize First-Mover Advantage
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

def run_rl_showdown():
    total_games = 1000000
    cores = mp.cpu_count()
    
    print("\n" + "="*70)
    print(" THE FINAL SHOWDOWN: EXACT DP vs SELF-TAUGHT RL (100,000 Games)")
    print("="*70)
    start_t = time.time()

    games_per_core = [total_games // cores] * cores
    for i in range(total_games % cores): games_per_core[i] += 1
    
    args = [(n, 'DP', 'RL') for n in games_per_core]

    with mp.Pool(processes=cores, initializer=init_worker) as pool:
        results = pool.map(simulate_matchup_chunk, args)

    final_stats = {
        'dp_wins': 0, 'rl_wins': 0, 'ties': 0,
        'dp_pts': 0, 'rl_pts': 0,
        'dp_margins': [], 'rl_margins': []
    }
    
    for r in results:
        final_stats['dp_wins'] += r['a_wins']
        final_stats['rl_wins'] += r['b_wins']
        final_stats['ties'] += r['ties']
        final_stats['dp_pts'] += r['a_pts']
        final_stats['rl_pts'] += r['b_pts']
        final_stats['dp_margins'].extend(r['a_margins'])
        final_stats['rl_margins'].extend(r['b_margins'])

    dp_win_pct = (final_stats['dp_wins'] / total_games) * 100
    rl_win_pct = (final_stats['rl_wins'] / total_games) * 100
    tie_pct = (final_stats['ties'] / total_games) * 100
    
    avg_dp_margin = sum(final_stats['dp_margins']) / max(1, final_stats['dp_wins'])
    avg_rl_margin = sum(final_stats['rl_margins']) / max(1, final_stats['rl_wins'])

    print(f"Simulation completed in {round((time.time() - start_t)/60, 2)} minutes.\n")
    print(f"[WIN RATES]")
    print(f"  Exact DP Agent: {dp_win_pct:.1f}%")
    print(f"  Self-Taught RL: {rl_win_pct:.1f}%")
    print(f"  Ties:           {tie_pct:.1f}%")
    print(f"\n[AVERAGE POINTS SCORED]")
    print(f"  Exact DP Agent: {final_stats['dp_pts']/total_games:.2f}")
    print(f"  Self-Taught RL: {final_stats['rl_pts']/total_games:.2f}")
    print(f"\n[MARGIN OF VICTORY]")
    print(f"  When DP wins, it wins by {avg_dp_margin:.2f} pts.")
    print(f"  When RL wins, it wins by {avg_rl_margin:.2f} pts.")
    print("="*70)

if __name__ == '__main__':
    run_rl_showdown()