import os
import numpy as np
import random
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from solvers.matrix_math import solve_zero_sum_matrix

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

shared_V_learned = None

def init_worker(shared_array):
    global shared_V_learned
    shared_V_learned = np.frombuffer(shared_array, dtype=np.float32).reshape((1048576, 2))
    np.random.seed(os.getpid() + int(time.time()))
    random.seed(os.getpid() + int(time.time()))

def calculate_score(r_id, b_id, penalties):
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def roll_dice():
    return {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
            'R': random.randint(1, 3), 'B': random.randint(1, 3)}

def worker_process(args):
    worker_id, episodes, dag, true_win_margin, alpha_start, alpha_end, tau_start, tau_end = args
    global shared_V_learned
    
    white_actions = ['R', 'B', None]
    color_actions = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]
    start_state = 0
    mse_history = []
    
    # Calculate how many local episodes equal 5 million global episodes for checkpointing
    local_checkpoint_mark = 5000000 // mp.cpu_count()
    
    for episode in range(1, episodes + 1):
        if random.random() < 0.8: state = random.choice(dag)
        else: state = start_state
            
        active_player = random.choice([1, 2])
        progress = episode / episodes
        
        # Temperature (tau) replaces epsilon
        tau = max(tau_end, tau_start - (progress / 0.8) * (tau_start - tau_end))
        alpha = max(alpha_end, alpha_start - progress * (alpha_start - alpha_end))
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                break
                
            dice = roll_dice()
            active_idx = 0 if active_player == 1 else 1
            next_active_idx = 1 if active_player == 1 else 0
            
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            best_c_actions = {}
            
            for w1_idx, a_w1 in enumerate(white_actions):
                for w2_idx, a_w2 in enumerate(white_actions):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    best_a_c = None
                    
                    for a_c in color_actions:
                        next_state, is_terminal = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        
                        if is_terminal: 
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_state)
                            s1 = calculate_score(np1_r, np1_b, np1_p)
                            s2 = calculate_score(np2_r, np2_b, np2_p)
                            if s1 > s2: future_val = 1.0     
                            elif s1 < s2: future_val = -1.0  
                            else: future_val = 0.0           
                        else: 
                            future_val = 0.0 + shared_V_learned[next_state, next_active_idx]
                        
                        if active_player == 1 and future_val > best_future_val: 
                            best_future_val = future_val
                            best_a_c = a_c
                        elif active_player == 2 and future_val < best_future_val: 
                            best_future_val = future_val
                            best_a_c = a_c
                                
                    payoff_matrix[w1_idx, w2_idx] = best_future_val
                    best_c_actions[(a_w1, a_w2)] = best_a_c

            v_target = solve_zero_sum_matrix(payoff_matrix)
            
            # --- ASYNCHRONOUS UPDATE TO SHARED MEMORY ---
            shared_V_learned[state, active_idx] += alpha * (v_target - shared_V_learned[state, active_idx])
            
            # --- BOLTZMANN LOGIC ---
            p1_vals = np.min(payoff_matrix, axis=1)
            p1_exp = np.exp((p1_vals - np.max(p1_vals)) / tau) 
            p1_probs = p1_exp / np.sum(p1_exp)
            
            p2_vals = np.max(payoff_matrix, axis=0)
            p2_exp = np.exp((-p2_vals - np.max(-p2_vals)) / tau) 
            p2_probs = p2_exp / np.sum(p2_exp)
            
            a_w1 = np.random.choice(white_actions, p=p1_probs)
            a_w2 = np.random.choice(white_actions, p=p2_probs)
            a_c = best_c_actions[(a_w1, a_w2)]
            
            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
            active_player = 2 if active_player == 1 else 1

        if worker_id == 0 and episode % 10000 == 0:
            p1_start_val = shared_V_learned[start_state, 0]
            error = (p1_start_val - true_win_margin) ** 2
            mse_history.append(error)
            print(f"Worker 0 | Episode {episode:07d} | Tau: {tau:.3f} | Alpha: {alpha:.3f} | Start Val: {p1_start_val:.4f} | MSE: {error:.4f}")

        # --- THE FIXED CHECKPOINT LOGIC ---
        if worker_id == 0 and local_checkpoint_mark > 0 and episode % local_checkpoint_mark == 0:
            global_milestone = (episode // local_checkpoint_mark) * 5
            chkpt_name = f'data/checkpoints/V_rl_boltzmann_{global_milestone}M.npy'
            np.save(chkpt_name, np.copy(shared_V_learned))
            print(f"\n[>>> CHECKPOINT SAVED: {chkpt_name} <<<]\n")

    return mse_history if worker_id == 0 else []

def train_multicore(total_episodes=20000000):
    print("Loading exact WIN PROBABILITY Nash values and DAG...")
    V_exact = np.load('data/V_nash_win_prob.npy')
    dag = np.load('data/topological_dag.npy')
    true_win_margin = V_exact[0, 0] 
    
    shared_array_base = mp.Array('f', 1048576 * 2, lock=False)
    
    cores = mp.cpu_count()
    episodes_per_core = total_episodes // cores
    print(f"Firing up {cores} CPU cores for Boltzmann! Each runs {episodes_per_core} episodes.")
    
    os.makedirs('data/checkpoints', exist_ok=True)
    start_time = time.time()
    
    args_list = []
    # Note: args list contains tau_start (0.5) and tau_end (0.01) instead of epsilon
    for i in range(cores):
        args_list.append((i, episodes_per_core, dag, true_win_margin, 0.1, 0.01, 0.5, 0.01))
        
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
        results = pool.map(worker_process, args_list)
        
    final_V_learned = np.frombuffer(shared_array_base, dtype=np.float32).reshape((1048576, 2))
    final_mse_history = results[0] 
    
    os.makedirs('data', exist_ok=True)
    np.save('data/V_rl_minimax_boltzmann.npy', final_V_learned)
    np.save('data/mse_history_boltzmann.npy', np.array(final_mse_history))
    
    end_time = time.time()
    print(f"\nMassive 20M Episode Boltzmann Complete in {round((end_time - start_time)/60, 2)} minutes!")

if __name__ == '__main__':
    train_multicore()