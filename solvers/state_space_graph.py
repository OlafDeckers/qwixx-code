import os
import time
import numpy as np
from collections import deque
from core.state_encoder import encode_state, decode_state
from core.environment import MiniQwixxEnv

# Precomputed lookup table to find the "depth" of a row ID
# Maps Row ID (0-13) to the total number of marks in that row.
# Derived from environment.py's get_row_details()
ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

def get_state_depth(state_int):
    """
    Calculates the topological depth of a state.
    Depth = Total Marks (P1 + P2) + Total Penalties (P1 + P2)
    Since every valid turn strictly increases the depth, sorting by 
    this guarantees a perfect topological order for Backward Induction.
    """
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    
    depth = ROW_ID_TO_COUNT[p1_r] + ROW_ID_TO_COUNT[p1_b] + p1_p + \
            ROW_ID_TO_COUNT[p2_r] + ROW_ID_TO_COUNT[p2_b] + p2_p
            
    return depth

def generate_state_space():
    print("Initializing Breadth-First Search (BFS)...")
    start_time = time.time()
    
    # 1. Generate all possible dice rolls for D3 dice
    dice_combinations = []
    for w1 in [1, 2, 3]:
        for w2 in [1, 2, 3]:
            for r in [1, 2, 3]:
                for b in [1, 2, 3]:
                    dice_combinations.append({'W1': w1, 'W2': w2, 'R': r, 'B': b})
                    
    # 2. Define all valid action inputs
    white_actions = ['R', 'B', None]
    color_actions = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]
    
    # 3. Initialize BFS structures
    start_state = encode_state(0, 0, 0, 0, 0, 0)
    visited = set([start_state])
    queue = deque([start_state])
    
    states_processed = 0
    
    # 4. BFS Loop
    while queue:
        current_state = queue.popleft()
        states_processed += 1
        
        if states_processed % 10000 == 0:
            print(f"Processed {states_processed} states... Current Queue Size: {len(queue)}")
            
        # If the state is terminal, it has no children. Skip expansion.
        # (Termination: 3 penalties for either player, or both rows locked)
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(current_state)
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
        
        if p1_p >= 3 or p2_p >= 3 or (red_locked and blue_locked):
            continue
            
        # To find ALL reachable board configurations, we simulate the state 
        # transitioning when Player 1 is active, and when Player 2 is active.
        for active_player in [1, 2]:
            for dice in dice_combinations:
                for a_w1 in white_actions:
                    for a_w2 in white_actions:
                        for a_c in color_actions:
                            
                            # Apply the environment transition
                            next_state, _ = MiniQwixxEnv.step(
                                current_state, active_player, dice, a_w1, a_w2, a_c
                            )
                            
                            if next_state not in visited:
                                visited.add(next_state)
                                queue.append(next_state)
                                
    print(f"\nBFS Complete! Total Unique Reachable States Found: {len(visited)}")
    
    # 5. Topologically Sort the DAG
    print("Topologically sorting states by depth...")
    sorted_states = sorted(list(visited), key=get_state_depth)
    
    # Convert to a flat, fast numpy array (32-bit integers)
    dag_array = np.array(sorted_states, dtype=np.int32)
    
    # 6. Save to disk so we NEVER have to run BFS again
    os.makedirs('data', exist_ok=True)
    np.save('data/topological_dag.npy', dag_array)
    
    end_time = time.time()
    print(f"State Space Generation saved to 'data/topological_dag.npy'.")
    print(f"Total Time Elapsed: {round(end_time - start_time, 2)} seconds.")

if __name__ == '__main__':
    generate_state_space()