import numpy as np
import random
import time
import os
import multiprocessing as mp
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from core.constants import COLOR_ACTIONS, ROW_ID_TO_COUNT, WHITE_ACTIONS
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score, roll_dice, UNIQUE_DICE, get_state_depth
from solvers.matrix_math import solve_zero_sum_matrix, get_nash_probs

# 1. Added the missing global variables
V_solo, V_score, V_win = None, None, None
V_h5, V_h10, V_h25, V_h50 = None, None, None, None

def init_worker():
    global V_solo, V_score, V_win, V_h5, V_h10, V_h25, V_h50
    V_solo = np.load('data/V_solo.npy', mmap_mode='r')
    V_score = np.load('data/V_nash.npy', mmap_mode='r')
    V_win = np.load('data/V_nash_win_prob.npy', mmap_mode='r')
    V_h5 = np.load('data/V_nash_hybrid_5.npy', mmap_mode='r')
    V_h10 = np.load('data/V_nash_hybrid_10.npy', mmap_mode='r')
    V_h25 = np.load('data/V_nash_hybrid_25.npy', mmap_mode='r')
    V_h50 = np.load('data/V_nash_hybrid_50.npy', mmap_mode='r')
    np.random.seed(os.getpid() + int(time.time()))
    random.seed(os.getpid() + int(time.time()))

def get_eval(state, active_idx, is_term, agent_type, eval_player):
    if is_term:
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        if agent_type == 'SOLO': return s1 if eval_player == 1 else -s2
        diff = s1 - s2
        if agent_type == 'SCORE': return diff
        elif agent_type == 'WIN': return 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
        elif agent_type.startswith('HYBRID'):
            bonus = float(agent_type.split('_')[1])
            return (diff + bonus) if diff > 0 else ((diff - bonus) if diff < 0 else 0.0)

    if agent_type == 'SOLO': return V_solo[state, active_idx, 0] if eval_player == 1 else -V_solo[state, active_idx, 1]
    elif agent_type == 'SCORE': return V_score[state, active_idx, 0] - V_score[state, active_idx, 1]
    elif agent_type == 'WIN': return V_win[state, active_idx]
    elif agent_type == 'HYBRID_5': return V_h5[state, active_idx]
    elif agent_type == 'HYBRID_10': return V_h10[state, active_idx]
    elif agent_type == 'HYBRID_25': return V_h25[state, active_idx]
    elif agent_type == 'HYBRID_50': return V_h50[state, active_idx]

def simulate_self_play(args):
    num_games, agent_type = args
    total_welfare = 0

    for _ in range(num_games):
        state = 0
        active_player = 1
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                s1 = calculate_score(p1_r, p1_b, p1_p)
                s2 = calculate_score(p2_r, p2_b, p2_p)
                total_welfare += (s1 + s2) 
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
                        val = get_eval(ns, next_idx, term, agent_type, active_player)
                        if active_player == 1 and val > best_val: best_val = val; best_c = c
                        elif active_player == 2 and val < best_val: best_val = val; best_c = c

                    best_c_dict[(w1_idx, w2_idx)] = best_c
                    final_ns, final_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c)
                    M_p1[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_type, 1)
                    M_p2[w1_idx, w2_idx] = get_eval(final_ns, next_idx, final_term, agent_type, 2)

            p1_probs, _ = get_nash_probs(M_p1)
            _, p2_probs = get_nash_probs(M_p2)
            idx_w1 = np.random.choice([0,1,2], p=p1_probs)
            idx_w2 = np.random.choice([0,1,2], p=p2_probs)
            c_action = best_c_dict[(idx_w1, idx_w2)]

            state, _ = MiniQwixxEnv.step(state, active_player, dice, WHITE_ACTIONS[idx_w1], WHITE_ACTIONS[idx_w2], c_action)
            active_player = 2 if active_player == 1 else 1
            
    return total_welfare

def calculate_spectrum_poa():
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    games_per_agent = 100000 
    cores = mp.cpu_count()
    
    results_welfare = {}

    print("\n" + "="*75)
    print(f" CALCULATING PRICE OF ANARCHY SPECTRUM ({games_per_agent} Games/Agent)")
    print("="*75)

    for agent in agents:
        print(f"Simulating {games_per_agent} matches for {agent} vs {agent}...")
        
        games_per_core = [games_per_agent // cores] * cores
        for i in range(games_per_agent % cores): games_per_core[i] += 1
        args = [(n, agent) for n in games_per_core]

        with mp.Pool(processes=cores, initializer=init_worker) as pool:
            welfares = pool.map(simulate_self_play, args)

        avg_welfare = sum(welfares) / games_per_agent
        results_welfare[agent] = avg_welfare

    w_opt = results_welfare['SOLO']

    print("\n" + "="*75)
    print(" FINAL RESULTS: THE PRICE OF ANARCHY SPECTRUM")
    print("="*75)
    print(f"Optimal Social Welfare (W_opt): {w_opt:.2f} Total Points per game\n")
    print(f"{'Strategy Type':<25} | {'Social Welfare':<15} | {'PoA (W_opt / W_nash)'}")
    print("-" * 75)
    
    plot_labels = []
    welfares_to_plot = []
    poas_to_plot = []

    for agent in ['SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']:
        w_nash = results_welfare[agent]
        poa = w_opt / w_nash
        print(f"{agent:<25} | {w_nash:<15.2f} | {poa:.4f}")
        
        # Prepare data for plotting
        label_map = {'SCORE': 'Score', 'HYBRID_5': 'Hybrid 5', 'HYBRID_10': 'Hybrid 10', 
                     'HYBRID_25': 'Hybrid 25', 'HYBRID_50': 'Hybrid 50', 'WIN': 'Win Prob'}
        plot_labels.append(label_map[agent])
        welfares_to_plot.append(w_nash)
        poas_to_plot.append(poa)
        
    print("="*75)
    
    # --- MATPLOTLIB PLOTTING LOGIC ---
    print("\nGenerating visual plot...")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary Y-Axis (Social Welfare Bar Chart)
    color_bar = '#4A90E2'
    ax1.set_xlabel('Strategy Objective', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Social Welfare (Combined Points)', color=color_bar, fontsize=12, fontweight='bold')
    ax1.bar(plot_labels, welfares_to_plot, color=color_bar, alpha=0.8, edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    
    # Add Solo Baseline Line
    ax1.axhline(y=w_opt, color='gray', linestyle='--', linewidth=2, label=f'Optimal Baseline ({w_opt:.2f} pts)')
    ax1.legend(loc='upper left')

    # Secondary Y-Axis (Price of Anarchy Line Graph)
    ax2 = ax1.twinx()  
    color_line = '#D0021B'
    ax2.set_ylabel('Price of Anarchy (PoA)', color=color_line, fontsize=12, fontweight='bold')
    ax2.plot(plot_labels, poas_to_plot, color=color_line, marker='o', markersize=8, linewidth=3, label='PoA')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim([0.9, max(poas_to_plot) + 0.15]) # Scale nicely above 1.0

    # Clean up layout
    plt.title('Price of Anarchy Spectrum in Mini-Qwixx', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout() 

    # Save to disk
    os.makedirs('data/plots', exist_ok=True)
    plot_path = 'data/plots/price_of_anarchy_spectrum.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")
    
    # Show the plot window
    plt.show()

if __name__ == '__main__':
    calculate_spectrum_poa()