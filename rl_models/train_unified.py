import os
import numpy as np
import random
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from core.constants import ROW_ID_TO_COUNT, WHITE_ACTIONS, COLOR_ACTIONS, TOTAL_STATES
from solvers.matrix_math import solve_zero_sum_matrix
from rl_models.agents import RewardShapingAgent, TDLambdaAgent, BoltzmannAgent

shared_V_learned = None

def init_worker(shared_array):
    global shared_V_learned
    shared_V_learned = np.frombuffer(shared_array, dtype=np.float32).reshape((TOTAL_STATES, 2))
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
    # Notice how configuration is cleanly injected here
    worker_id, episodes, dag, true_win_margin, alpha_bounds, param_bounds, model_type, chkpt_interval = args
    global shared_V_learned
    
    # Initialize the specific Strategy
    if model_type == 'reward_shape': agent = RewardShapingAgent()
    elif model_type == 'td_lambda': agent = TDLambdaAgent()
    elif model_type == 'boltzmann': agent = BoltzmannAgent()
    else: raise ValueError("Unknown model_type")
    
    start_state = 0
    mse_history = []
    
    # Calculate local checkpoint based on injected global interval
    local_checkpoint_mark = max(1, chkpt_interval // mp.cpu_count())
    
    for episode in range(1, episodes + 1):
        state = random.choice(dag) if random.random() < 0.8 else start_state
        active_player = random.choice([1, 2])
        progress = episode / episodes
        
        param = max(param_bounds[1], param_bounds[0] - (progress / 0.8) * (param_bounds[0] - param_bounds[1]))
        alpha = max(alpha_bounds[1], alpha_bounds[0] - progress * (alpha_bounds[0] - alpha_bounds[1]))
        
        agent.reset_episode()
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                break
                
            dice = roll_dice()
            active_idx = 0 if active_player == 1 else 1
            next_active_idx = 1 if active_player == 1 else 0
            
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            best_c_actions = {}
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    best_a_c = None
                    
                    for a_c in COLOR_ACTIONS:
                        next_state, is_terminal = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(next_state)
                        s1 = calculate_score(np1_r, np1_b, np1_p) if is_terminal else 0
                        s2 = calculate_score(np2_r, np2_b, np2_p) if is_terminal else 0
                        
                        env_info = {
                            'active_player': active_player, 'np1_p': np1_p, 'p1_p': p1_p, 'np2_p': np2_p, 'p2_p': p2_p,
                            'c_np1_r': ROW_ID_TO_COUNT[np1_r], 'c_p1_r': ROW_ID_TO_COUNT[p1_r],
                            'c_np1_b': ROW_ID_TO_COUNT[np1_b], 'c_p1_b': ROW_ID_TO_COUNT[p1_b],
                            'c_np2_r': ROW_ID_TO_COUNT[np2_r], 'c_p2_r': ROW_ID_TO_COUNT[p2_r],
                            'c_np2_b': ROW_ID_TO_COUNT[np2_b], 'c_p2_b': ROW_ID_TO_COUNT[p2_b]
                        }
                        
                        future_val = agent.get_future_value(is_terminal, s1, s2, shared_V_learned, next_state, next_active_idx, env_info)
                        
                        if active_player == 1 and future_val > best_future_val: 
                            best_future_val, best_a_c = future_val, a_c
                        elif active_player == 2 and future_val < best_future_val: 
                            best_future_val, best_a_c = future_val, a_c
                                
                    payoff_matrix[w1_idx, w2_idx] = best_future_val
                    best_c_actions[(a_w1, a_w2)] = best_a_c

            v_target = solve_zero_sum_matrix(payoff_matrix)
            agent.update_value(state, active_idx, v_target, alpha, shared_V_learned)
            
            a_w1, a_w2 = agent.select_actions(payoff_matrix, WHITE_ACTIONS, best_c_actions, param)
            a_c = best_c_actions[(a_w1, a_w2)] if model_type == 'boltzmann' else random.choice(COLOR_ACTIONS)
            
            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
            active_player = 2 if active_player == 1 else 1

        # Print logic (scales based on episodes)
        print_interval = max(1, episodes // 10)
        if worker_id == 0 and episode % print_interval == 0:
            p1_start_val = shared_V_learned[start_state, 0]
            error = (p1_start_val - true_win_margin) ** 2
            mse_history.append(error)
            print(f"Worker 0 | Episode {episode:07d} | Param: {param:.3f} | Alpha: {alpha:.3f} | MSE: {error:.4f}")

        # Checkpoint logic
        if worker_id == 0 and episode % local_checkpoint_mark == 0:
            global_milestone = (episode // local_checkpoint_mark) * (chkpt_interval // 1000)
            chkpt_name = f'data/checkpoints/V_rl_{model_type}_{global_milestone}K.npy'
            np.save(chkpt_name, np.copy(shared_V_learned))
            print(f"\n[>>> CHECKPOINT SAVED: {chkpt_name} <<<]\n")

    return mse_history if worker_id == 0 else []

def train_unified(model_type, total_episodes, checkpoint_interval):
    """
    Main orchestrator for training. Dependencies (episodes, intervals) are injected here.
    """
    print(f"Loading environment for {model_type.upper()} training...")
    V_exact = np.load('data/V_nash_win_prob.npy')
    dag = np.load('data/topological_dag.npy')
    true_win_margin = V_exact[0, 0] 
    
    shared_array_base = mp.Array('f', TOTAL_STATES * 2, lock=False)
    cores = mp.cpu_count()
    episodes_per_core = total_episodes // cores
    
    print(f"Firing up {cores} CPU cores! Each core will run {episodes_per_core} episodes (Total: {total_episodes}).")
    
    os.makedirs('data/checkpoints', exist_ok=True)
    start_time = time.time()
    
    # Model-Specific Hyperparameters
    param_bounds = (0.5, 0.01) if model_type == 'boltzmann' else (1.0, 0.01)
    alpha_bounds = (0.1, 0.01)
    
    # Inject all configurations into the worker arguments
    args_list = [(i, episodes_per_core, dag, true_win_margin, alpha_bounds, param_bounds, model_type, checkpoint_interval) for i in range(cores)]
        
    with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
        results = pool.map(worker_process, args_list)
        
    final_V_learned = np.frombuffer(shared_array_base, dtype=np.float32).reshape((TOTAL_STATES, 2))
    
    # Save the final result
    np.save(f'data/V_rl_minimax_{model_type}_TEST.npy', final_V_learned)
    np.save(f'data/mse_history_{model_type}_TEST.npy', np.array(results[0]))
    
    end_time = time.time()
    print(f"\n{model_type.upper()} Complete in {round((end_time - start_time), 2)} seconds!")

def run_benchmark(benchmark_episodes=100_000, target_episodes=20_000_000):
    """
    Runs a small number of episodes for all three models to extrapolate total training time.
    """
    print("Loading exact WIN PROBABILITY Nash values and DAG for benchmarking...")
    V_exact = np.load('data/V_nash_win_prob.npy')
    dag = np.load('data/topological_dag.npy')
    true_win_margin = V_exact[0, 0] 
    
    cores = mp.cpu_count()
    episodes_per_core = benchmark_episodes // cores
    multiplier = target_episodes / benchmark_episodes
    
    print(f"===========================================================")
    print(f" BENCHMARKING TIME FOR {target_episodes:,} EPISODES ")
    print(f" Running {benchmark_episodes:,} episodes per model and extrapolating x{int(multiplier)}")
    print(f"===========================================================\n")

    models = ['reward_shape', 'td_lambda', 'boltzmann']
    
    for model_type in models:
        print(f"Benchmarking [{model_type.upper()}]...")
        shared_array_base = mp.Array('f', 1048576 * 2, lock=False)
        
        param_bounds = (0.5, 0.01) if model_type == 'boltzmann' else (1.0, 0.01)
        alpha_bounds = (0.1, 0.01)
        
        # We pass 'target_episodes + 1' as the checkpoint interval so it never triggers a save during benchmark
        args_list = [(i, episodes_per_core, dag, true_win_margin, alpha_bounds, param_bounds, model_type, target_episodes + 1) for i in range(cores)]
        
        start_time = time.time()
        with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
            pool.map(worker_process, args_list)
        end_time = time.time()
        
        duration_seconds = end_time - start_time
        extrapolated_seconds = duration_seconds * multiplier
        extrapolated_minutes = extrapolated_seconds / 60
        extrapolated_hours = extrapolated_seconds / 3600
        
        print(f"  -> Benchmark Time: {duration_seconds:.2f} seconds")
        print(f"  -> Extrapolated Time (20M): ~{extrapolated_minutes:.2f} Minutes (~{extrapolated_hours:.2f} Hours)\n")

if __name__ == '__main__':
    # ==========================================
    # CONFIGURATION BLOCK (Single Source of Truth)
    # ==========================================
    
    # Modes: 'BENCHMARK', 'TEST', or 'PRODUCTION'
    RUN_MODE = 'BENCHMARK'  
    
    # Choose your model (Only applies to TEST or PRODUCTION modes)
    TARGET_MODEL = 'boltzmann' # Options: 'boltzmann', 'td_lambda', 'reward_shape'
    
    if RUN_MODE == 'BENCHMARK':
        run_benchmark(benchmark_episodes=500_000, target_episodes=20_000_000)
        
    elif RUN_MODE == 'TEST':
        print("=== RUNNING IN QUICK TEST MODE ===")
        train_unified(
            model_type=TARGET_MODEL, 
            total_episodes=50_000, 
            checkpoint_interval=25_000
        )
        
    elif RUN_MODE == 'PRODUCTION':
        print("=== RUNNING IN MASSIVE PRODUCTION MODE ===")
        train_unified(
            model_type=TARGET_MODEL, 
            total_episodes=20_000_000, 
            checkpoint_interval=5_000_000 
        )