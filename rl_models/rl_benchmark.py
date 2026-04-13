import os
import numpy as np
import random
import time
import multiprocessing as mp
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv
from solvers.matrix_math import solve_zero_sum_matrix

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]
WHITE_ACTIONS = ['R', 'B', None]
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

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

# --- 1. REWARD SHAPING WORKER ---
def worker_reward_shape(args):
    worker_id, episodes, dag = args
    global shared_V_learned
    start_state = 0
    for episode in range(1, episodes + 1):
        state = random.choice(dag) if random.random() < 0.8 else start_state
        active_player = random.choice([1, 2])
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)): break
            dice = roll_dice()
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    for a_c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                        if term: 
                            s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else:
                            step_reward = 0.0
                            if active_player == 1 and np1_p > p1_p: step_reward -= 0.05
                            elif active_player == 2 and np2_p > p2_p: step_reward += 0.05 
                            val = step_reward + shared_V_learned[ns, 1 if active_player == 1 else 0]
                        if active_player == 1 and val > best_future_val: best_future_val = val
                        elif active_player == 2 and val < best_future_val: best_future_val = val
                    payoff_matrix[w1_idx, w2_idx] = best_future_val

            v_target = solve_zero_sum_matrix(payoff_matrix)
            shared_V_learned[state, 0 if active_player == 1 else 1] += 0.1 * (v_target - shared_V_learned[state, 0 if active_player == 1 else 1])
            state, _ = MiniQwixxEnv.step(state, active_player, dice, random.choice(WHITE_ACTIONS), random.choice(WHITE_ACTIONS), random.choice(COLOR_ACTIONS))
            active_player = 2 if active_player == 1 else 1

# --- 2. TD-LAMBDA WORKER ---
def worker_td_lambda(args):
    worker_id, episodes, dag = args
    global shared_V_learned
    start_state = 0
    for episode in range(1, episodes + 1):
        state = random.choice(dag) if random.random() < 0.8 else start_state
        active_player = random.choice([1, 2])
        eligibility_traces = {}
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)): break
            dice = roll_dice()
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    for a_c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        if term: 
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                            s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else: val = 0.0 + shared_V_learned[ns, 1 if active_player == 1 else 0]
                        if active_player == 1 and val > best_future_val: best_future_val = val
                        elif active_player == 2 and val < best_future_val: best_future_val = val
                    payoff_matrix[w1_idx, w2_idx] = best_future_val

            v_target = solve_zero_sum_matrix(payoff_matrix)
            active_idx = 0 if active_player == 1 else 1
            delta = v_target - shared_V_learned[state, active_idx]
            eligibility_traces[(state, active_idx)] = 1.0
            
            for (trace_state, trace_player) in list(eligibility_traces.keys()):
                shared_V_learned[trace_state, trace_player] += 0.1 * delta * eligibility_traces[(trace_state, trace_player)]
                eligibility_traces[(trace_state, trace_player)] *= 0.9
                if eligibility_traces[(trace_state, trace_player)] < 0.01:
                    del eligibility_traces[(trace_state, trace_player)]
                    
            state, _ = MiniQwixxEnv.step(state, active_player, dice, random.choice(WHITE_ACTIONS), random.choice(WHITE_ACTIONS), random.choice(COLOR_ACTIONS))
            active_player = 2 if active_player == 1 else 1

# --- 3. BOLTZMANN WORKER ---
def worker_boltzmann(args):
    worker_id, episodes, dag = args
    global shared_V_learned
    start_state = 0
    tau = 0.1
    for episode in range(1, episodes + 1):
        state = random.choice(dag) if random.random() < 0.8 else start_state
        active_player = random.choice([1, 2])
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)): break
            dice = roll_dice()
            payoff_matrix = np.zeros((3, 3), dtype=np.float32)
            best_c_actions = {}
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_future_val = -9999.0 if active_player == 1 else 9999.0
                    best_a_c = None
                    for a_c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        if term: 
                            np1_r, np1_b, np1_p, np2_r, np2_b, np2_p = decode_state(ns)
                            s1, s2 = calculate_score(np1_r, np1_b, np1_p), calculate_score(np2_r, np2_b, np2_p)
                            val = 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
                        else: val = 0.0 + shared_V_learned[ns, 1 if active_player == 1 else 0]
                        
                        if active_player == 1 and val > best_future_val: best_future_val = val; best_a_c = a_c
                        elif active_player == 2 and val < best_future_val: best_future_val = val; best_a_c = a_c
                    payoff_matrix[w1_idx, w2_idx] = best_future_val
                    best_c_actions[(a_w1, a_w2)] = best_a_c

            v_target = solve_zero_sum_matrix(payoff_matrix)
            active_idx = 0 if active_player == 1 else 1
            shared_V_learned[state, active_idx] += 0.1 * (v_target - shared_V_learned[state, active_idx])
            
            p1_vals = np.min(payoff_matrix, axis=1)
            p1_exp = np.exp((p1_vals - np.max(p1_vals)) / tau) 
            p1_probs = p1_exp / np.sum(p1_exp)
            p2_vals = np.max(payoff_matrix, axis=0)
            p2_exp = np.exp((-p2_vals - np.max(-p2_vals)) / tau) 
            p2_probs = p2_exp / np.sum(p2_exp)
            
            a_w1 = np.random.choice(WHITE_ACTIONS, p=p1_probs)
            a_w2 = np.random.choice(WHITE_ACTIONS, p=p2_probs)
            
            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c_actions[(a_w1, a_w2)])
            active_player = 2 if active_player == 1 else 1

def run_benchmark():
    BENCHMARK_EPISODES = 100000
    TARGET_EPISODES = 20000000
    MULTIPLIER = TARGET_EPISODES / BENCHMARK_EPISODES
    
    dag = np.load('data/topological_dag.npy')
    cores = mp.cpu_count()
    episodes_per_core = BENCHMARK_EPISODES // cores
    
    print(f"===========================================================")
    print(f" BENCHMARKING TIME FOR {TARGET_EPISODES:,} EPISODES ")
    print(f" Running {BENCHMARK_EPISODES:,} episodes and extrapolating x{int(MULTIPLIER)}")
    print(f"===========================================================\n")

    models = [
        ("Reward Shaping", worker_reward_shape),
        ("TD-Lambda", worker_td_lambda),
        ("Boltzmann", worker_boltzmann)
    ]
    
    for name, worker_func in models:
        print(f"Benchmarking [{name}]...")
        shared_array_base = mp.Array('f', 1048576 * 2, lock=False)
        args_list = [(i, episodes_per_core, dag) for i in range(cores)]
        
        start_time = time.time()
        with mp.Pool(processes=cores, initializer=init_worker, initargs=(shared_array_base,)) as pool:
            pool.map(worker_func, args_list)
        end_time = time.time()
        
        duration_seconds = end_time - start_time
        extrapolated_seconds = duration_seconds * MULTIPLIER
        extrapolated_hours = extrapolated_seconds / 3600
        
        print(f"  -> Benchmark Time: {duration_seconds:.2f} seconds")
        print(f"  -> Extrapolated Time (20M): ~{extrapolated_hours:.2f} Hours\n")

if __name__ == '__main__':
    run_benchmark()