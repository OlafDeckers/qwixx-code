import os
import numpy as np
import random
import time
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv

ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

def calculate_score(r_id, b_id, penalties):
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def get_final_score_diff(state_int):
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    return calculate_score(p1_r, p1_b, p1_p) - calculate_score(p2_r, p2_b, p2_p)

def roll_dice():
    return {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
            'R': random.randint(1, 3), 'B': random.randint(1, 3)}

def train_sparse_rewards(episodes=1000000, alpha_start=0.1, alpha_end=0.01, epsilon_start=1.0, epsilon_end=0.01):
    print("Loading exact Nash values and DAG...")
    V_exact = np.load('data/V_nash.npy')
    dag = np.load('data/topological_dag.npy')
    
    # Extract the true Nash difference from the 3D array to track our error
    true_nash_diff = V_exact[0, 0, 0] - V_exact[0, 0, 1]
    
    V_learned = np.zeros((1048576, 2), dtype=np.float32)
    
    white_actions = ['R', 'B', None]
    color_actions = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]
    
    start_state = 0
    mse_history = []
    
    start_time = time.time()
    print(f"Beginning SPARSE REWARD Q-Learning for {episodes} episodes...")
    print(f"Target Value to Learn: {true_nash_diff:.4f}")
    
    for episode in range(episodes):
        # 80% Exploring Starts to hit all states, 20% natural game progression
        if random.random() < 0.8:
            state = random.choice(dag)
        else:
            state = start_state
            
        active_player = random.choice([1, 2])
        
        # Linear Decay Schedules
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
                        
                        # --- SPARSE REWARD LOGIC ---
                        # The AI only gets a reward if the game is completely over.
                        if is_terminal:
                            future_val = get_final_score_diff(next_state)
                        else:
                            future_val = V_learned[next_state, next_active_idx]
                        
                        # Find the max/min action for the active player
                        if active_player == 1 and future_val > best_future_val: 
                            best_future_val = future_val
                        elif active_player == 2 and future_val < best_future_val: 
                            best_future_val = future_val
                                
                    payoff_matrix[w1_idx, w2_idx] = best_future_val

            # --- NAIVE Q-LEARNING TARGET ---
            # It ignores the Nash Matrix entirely and just blindly takes the max/min
            if random.random() < epsilon:
                v_target = random.choice(payoff_matrix.flatten())
            else:
                if active_player == 1:
                    v_target = np.max(payoff_matrix) # Greedy Max
                else:
                    v_target = np.min(payoff_matrix) # Greedy Min
            
            # Temporal Difference Update
            V_learned[state, active_idx] += alpha * (v_target - V_learned[state, active_idx])
            
            # Take a random action to progress the state (Off-Policy Learning)
            a_w1, a_w2, a_c = random.choice(white_actions), random.choice(white_actions), random.choice(color_actions)
            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
            active_player = 2 if active_player == 1 else 1

        # Logging
        if episode % 500 == 0:
            p1_start_val = V_learned[start_state, 0]
            error = (p1_start_val - true_nash_diff) ** 2
            mse_history.append(error)
            print(f"Episode {episode:07d} | Epsilon: {epsilon:.2f} | Alpha: {alpha:.3f} | P1 Start Value: {p1_start_val:.4f} | MSE: {error:.4f}")

    os.makedirs('data', exist_ok=True)
    np.save('data/V_rl_sparse.npy', V_learned)
    np.save('data/mse_history_sparse.npy', np.array(mse_history))
    
    end_time = time.time()
    print(f"\nSparse Q-Learning Complete in {round((end_time - start_time)/60, 2)} minutes!")

if __name__ == '__main__':
    train_sparse_rewards()