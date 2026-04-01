import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

# ---> CHANGE THIS VALUE ONCE HERE <---
NUM_SIMULATED_GAMES = 100000 

def calculate_score(r_id, b_id, penalties):
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def roll_dice():
    return {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
            'R': random.randint(1, 3), 'B': random.randint(1, 3)}

def simulate_games_chunk(args):
    """The worker function that runs a smaller chunk of games on a single CPU core."""
    chunk_size, rl_V, dp_V = args
    
    # CRITICAL: Re-seed the random generator so cores don't mirror each other's dice rolls
    random.seed(os.getpid() + int(time.time() * 1000))
    np.random.seed((os.getpid() + int(time.time() * 1000)) % 4294967295)
    
    local_rl_wins = 0
    white_actions = ['R', 'B', None]
    color_actions = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]
    
    for _ in range(chunk_size):
        state = 0
        # Randomly select who goes first
        active_player = random.choice([1, 2]) 
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                break
                
            dice = roll_dice()
            next_active_idx = 1 if active_player == 1 else 0
            
            # --- STEP 1: INDEPENDENT WHITE ACTIONS ---
            rl_payoff = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(white_actions):
                for w2_idx, a_w2 in enumerate(white_actions):
                    best_val = -9999.0
                    for a_c in color_actions:
                        next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        if is_term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                            s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else: val = rl_V[next_s, next_active_idx]
                        if val > best_val: best_val = val
                    rl_payoff[w1_idx, w2_idx] = best_val

            dp_payoff = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(white_actions):
                for w2_idx, a_w2 in enumerate(white_actions):
                    best_val = 9999.0
                    for a_c in color_actions:
                        next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        if is_term:
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                            s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else: val = dp_V[next_s, next_active_idx]
                        if val < best_val: best_val = val
                    dp_payoff[w1_idx, w2_idx] = best_val

            p1_w1_idx = np.argmax(np.min(rl_payoff, axis=1))
            a_w1_chosen = white_actions[p1_w1_idx]

            p2_w2_idx = np.argmin(np.max(dp_payoff, axis=0))
            a_w2_chosen = white_actions[p2_w2_idx]

            # --- STEP 2: ACTIVE PLAYER PICKS COLOR ACTION ---
            best_final_c = None
            if active_player == 1: 
                best_val = -9999.0
                for a_c in color_actions:
                    next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, a_c)
                    if is_term:
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                        s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                        val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                    else: val = rl_V[next_s, next_active_idx]
                    
                    if val > best_val:
                        best_val = val
                        best_final_c = a_c
            else: 
                best_val = 9999.0
                for a_c in color_actions:
                    next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, a_c)
                    if is_term:
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_s)
                        s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                        val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                    else: val = dp_V[next_s, next_active_idx]
                    
                    if val < best_val:
                        best_val = val
                        best_final_c = a_c

            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, best_final_c)
            active_player = 2 if active_player == 1 else 1
            
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        if calculate_score(p1_r, p1_b, p1_p) > calculate_score(p2_r, p2_b, p2_p):
            local_rl_wins += 1
            
    return local_rl_wins

def simulate_games(rl_V, dp_V, num_games=NUM_SIMULATED_GAMES):
    """The Multi-Core Manager: Splits the games across all available CPU cores."""
    cores = mp.cpu_count()
    chunk_size = num_games // cores
    remainder = num_games % cores
    
    args_list = []
    for i in range(cores):
        games_for_this_core = chunk_size + (remainder if i == 0 else 0)
        args_list.append((games_for_this_core, rl_V, dp_V))
        
    with mp.Pool(processes=cores) as pool:
        results = pool.map(simulate_games_chunk, args_list)
        
    total_rl_wins = sum(results)
    return (total_rl_wins / num_games) * 100

def generate_comparison_plot():
    print("Loading Exact DP Agent...")
    try:
        dp_V = np.load('data/V_nash_win_prob.npy')
    except FileNotFoundError:
        print("Error: Could not find V_nash_win_prob.npy")
        return

    models = {
        'Reward Shaping': {'prefix': 'V_rl_reward_shape_', 'color': '#4A90E2', 'marker': 'o'},
        'TD-Lambda': {'prefix': 'V_rl_td_lambda_', 'color': '#D0021B', 'marker': 's'},
        'Boltzmann': {'prefix': 'V_rl_boltzmann_', 'color': '#F5A623', 'marker': '^'}
    }
    
    milestones = [5, 10, 15, 20]
    plot_data = {name: [] for name in models.keys()}
    
    print(f"\nStarting the Checkpoint Tournament ({NUM_SIMULATED_GAMES:,} games per checkpoint). Running on {mp.cpu_count()} cores...\n")

    for model_name, info in models.items():
        print(f"--- Evaluating {model_name} ---")
        plot_data[model_name].append(0.0) # Untrained baseline
        
        for m in milestones:
            file_path = f"data/checkpoints/{info['prefix']}{m}M.npy"
            try:
                print(f"  Loading {m}M Checkpoint...")
                rl_V = np.load(file_path)
                
                win_rate = simulate_games(rl_V, dp_V, num_games=NUM_SIMULATED_GAMES)
                
                plot_data[model_name].append(win_rate)
                print(f"  Result: {win_rate:.1f}% Win Rate")
            except FileNotFoundError:
                print(f"  [!] Missing file: {file_path}. Filling with NaN.")
                plot_data[model_name].append(np.nan)

    print("\nGenerating the Graph...")
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 7))
    
    x_axis = [0, 5, 10, 15, 20]
    
    for model_name, info in models.items():
        plt.plot(x_axis, plot_data[model_name], label=model_name, 
                 color=info['color'], marker=info['marker'], linewidth=2.5, markersize=8)

    plt.title("RL Agent Performance vs. Exact DP (Win Probability)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Training Iterations (Millions)", fontsize=13, fontweight='bold')
    plt.ylabel("Win Rate (%) against Perfect DP", fontsize=13, fontweight='bold')
    
    plt.xticks(x_axis, [f"{x}M" for x in x_axis])
    plt.ylim(0, max(50, max([max(v) for v in plot_data.values() if not np.isnan(v).all()]) + 10))
    plt.axhline(y=48.0, color='gray', linestyle='--', alpha=0.7, label='Theoretical Max (~48%)')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()

    save_path = 'plots/rl_model_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Success! Plot saved beautifully to {save_path}")
    plt.show()

if __name__ == '__main__':
    generate_comparison_plot()