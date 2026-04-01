import os
import numpy as np
import random
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from solvers.matrix_math import solve_zero_sum_matrix

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

# Global variable for the worker processes to access the shared memory
shared_V_learned = None

def init_worker(shared_array):
    """Initializes the shared memory array for each CPU core."""
    global shared_V_learned
    shared_V_learned = np.frombuffer(shared_array, dtype=np.float32).reshape((1048576, 2))
    # Give each process a unique random seed so they don't roll the exact same dice
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
    """The actual RL loop that runs on every CPU core simultaneously."""
    worker_id, episodes, dag, true_win_margin, alpha_start, alpha_end, epsilon_start, epsilon_end = args
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
        epsilon = max(epsilon_end, epsilon_start - (progress / 0.8) * (epsilon_start - epsilon_end))
        alpha = max(alpha_end, alpha_start - progress * (alpha_start - alpha_end))
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                break
                
            dice = roll_dice()
            active_idx = 0 if active_player == 1 else 1
            next_active_idx = 1 if active_player == 1 else 0
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(white_actions):
                for w2_idx, a_w2 in enumerate(white_actions):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    
                    for a_c in color_actions:
                        next_state, is_terminal = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_state)
                        
                        if is_terminal: 
                            s1 = calculate_score(np1_r, np1_b, np1_p)
                            s2 = calculate_score(np2_r, np2_b, np2_p)
                            if s1 > s2: future_val = 1.0     
                            elif s1 < s2: future_val = -1.0  
                            else: future_val = 0.0           
                        else: 
                            # REWARD SHAPING LOGIC
                            step_reward = 0.0
                            if active_player == 1 and np1_p > p1_p: step_reward -= 0.05
                            elif active_player == 2 and np2_p > p2_p: step_reward += 0.05 
                            
                            c_np1_r, c_np1_b = ROW_ID_TO_COUNT[np1_r], ROW_ID_TO_COUNT[np1_b]
                            c_p1_r, c_p1_b = ROW_ID_TO_COUNT[p1_r], ROW_ID_TO_COUNT[p1_b]
                            c_np2_r, c_np2_b = ROW_ID_TO_COUNT[np2_r], ROW_ID_TO_COUNT[np2_b]
                            c_p2_r, c_p2_b = ROW_ID_TO_COUNT[p2_r], ROW_ID_TO_COUNT[p2_b]
                            
                            if active_player == 1 and (c_np1_r > c_p1_r or c_np1_b > c_p1_b): step_reward += 0.01
                            elif active_player == 2 and (c_np2_r > c_p2_r or c_np2_b > c_p2_b): step_reward -= 0.01
                            
                            future_val = step_reward + shared_V_learned[next_state, next_active_idx]
                        
                        if active_player == 1 and future_val > best_future_val: best_future_val = future_val
                        elif active_player == 2 and future_val < best_future_val: best_future_val = future_val
                                
                    payoff_matrix[w1_idx, w2_idx] = best_future_val

            v_target = solve_zero_sum_matrix(payoff_matrix)
            
            # --- ASYNCHRONOUS UPDATE TO SHARED MEMORY ---
            shared_V_learned[state, active_idx] += alpha * (v_target - shared_V_learned[state, active_idx])
            
            if random.random() < epsilon:
                a_w1, a_w2 = random.choice(white_actions), random.choice(white_actions)
            else:
                a_w1, a_w2 = random.choice(white_actions), random.choice(white_actions)
                
            a_c = random.choice(color_actions)
            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
            active_player = 2 if active_player == 1 else 1

        # Only Worker 0 prints to the terminal and saves checkpoints
        if worker_id == 0 and episode % 10000 == 0:
            p1_start_val = shared_V_learned[start_state, 0]
            error = (p1_start_val - true_win_margin) ** 2
            mse_history.append(error)
            print(f"Worker 0 | Episode {episode:07d} | Eps: {epsilon:.2f} | Alpha: {alpha:.3f} | Start Val: {p1_start_val:.4f} | MSE: {error:.4f}")

        # --- THE FIXED CHECKPOINT LOGIC ---
        if worker_id == 0 and local_checkpoint_mark > 0 and episode % local_checkpoint_mark == 0:
            global_milestone = (episode // local_checkpoint_mark) * 5
            chkpt_name = f'data/checkpoints/V_rl_reward_shape_{global_milestone}M.npy'
            np.save(chkpt_name, np.copy(shared_V_learned))
            print(f"\n[>>> CHECKPOINT SAVED: {chkpt_name} <<<]\n")

    return mse_history if worker_id == 0 else []

# CHANGED: total_episodes defaults to 20,000,000
def train_minimax_multicore(total_episodes=20000000):
    print("Loading exact WIN PROBABILITY Nash values and DAG...")
    V_exact = np.load('data/V_nash_win_prob.npy')
    dag = np.load('data/topological_dag.npy')
    true_win_margin = V_exact[0, 0] 
    
    # --- 1. SET UP SHARED MEMORY ---
    shared_array_base = mp.Array('f', 1048576 * 2, lock=False)
    
    cores = mp.cpu_count()
    episodes_per_core = total_episodes // cores
    print(f"Firing up {cores} CPU cores! Each core will run {episodes_per_core} episodes (Total: {total_episodes}).")
    print(f"Target Start State Win Margin to Learn: {true_win_margin:.4f}")
    
    os.makedirs('data/checkpoints', exist_ok=True)
    start_time = time.time()
    
    # --- 2. DISPATCH WORKERS ---
    args_list = []
    for i in range(cores):
        args_list.append((i, episodes_per_core, dag, true_win_margin, 0.1, 0.01, 1.0, 0.01))
        
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
        results = pool.map(worker_process, args_list)
        
    # --- 3. GATHER RESULTS ---
    final_V_learned = np.frombuffer(shared_array_base, dtype=np.float32).reshape((1048576, 2))
    
    # Get the MSE history from Worker 0
    final_mse_history = results[0] 
    
    os.makedirs('data', exist_ok=True)
    np.save('data/V_rl_minimax_reward_shape.npy', final_V_learned)
    np.save('data/mse_history_reward_shape.npy', np.array(final_mse_history))
    
    end_time = time.time()
    print(f"\nMassive {total_episodes} Episode Multicore Q-Learning Complete in {round((end_time - start_time)/60, 2)} minutes!")

if __name__ == '__main__':
    train_minimax_multicore()