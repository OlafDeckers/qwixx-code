"""
solvers/state_space_graph.py

State Space Reachability and Topological Sorting.
This module empirically maps the complete reachable state space |S| of the 
Mini-Qwixx environment. By performing a forward-reachability Breadth-First Search (BFS) 
from the initial state s_0, we bypass the computationally wasteful Cartesian product 
of all theoretical state combinations (many of which are mathematically unreachable).

Finally, it topologically sorts the state space, providing the strict strict ordering 
necessary to evaluate the Bellman equations via one-pass Backward Induction.

Thesis Reference: Section "Structural Properties: The DAG Advantage"
"""

import os
import time
import numpy as np
from collections import deque
from core.state_encoder import encode_state, decode_state
from core.environment import MiniQwixxEnv, get_state_depth
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS

def generate_state_space():
    """
    Computes the exact connected component of the state space graph starting from s_0.
    Produces an array of topologically sorted state integers.
    """
    print("Initializing Breadth-First Search (BFS)...")
    start_time = time.time()
    
    # 1. Generate the Stochastic Chance Nodes (D)
    # 81 possible permutations for D3 dice rolls (3^4)
    dice_combinations = []
    for w1 in [1, 2, 3]:
        for w2 in [1, 2, 3]:
            for r in [1, 2, 3]:
                for b in [1, 2, 3]:
                    dice_combinations.append({'W1': w1, 'W2': w2, 'R': r, 'B': b})
    
    # 2. Initialize BFS structures
    # s_0: The origin node of the Markov Game (empty board, zero penalties)
    start_state = encode_state(0, 0, 0, 0, 0, 0)
    visited = set([start_state])
    queue = deque([start_state])
    
    states_processed = 0
    
    # 3. BFS Forward-Reachability Loop
    # Expands the frontier of the graph by applying the Transition Function T(s, a, d)
    while queue:
        current_state = queue.popleft()
        states_processed += 1
        
        if states_processed % 10000 == 0:
            print(f"Processed {states_processed} states... Current Queue Size: {len(queue)}")
            
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(current_state)
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
        
        # S_terminal Check: If the state is terminal, it has no outgoing edges.
        # (Termination: 3 penalties for either player, or both rows locked)
        if p1_p >= 3 or p2_p >= 3 or (red_locked and blue_locked):
            continue
            
        # 4. State Expansion
        # To find ALL reachable board configurations, we simulate the state 
        # transitioning when Player 1 is the active player, and when Player 2 is active.
        for active_player in [1, 2]:
            for dice in dice_combinations:
                # Joint Action Space evaluation: Aw1 x Aw2 x Ac
                for a_w1 in WHITE_ACTIONS:
                    for a_w2 in WHITE_ACTIONS:
                        for a_c in COLOR_ACTIONS:
                            
                            # Apply the deterministic environment transition T(s, a, d)
                            next_state, _ = MiniQwixxEnv.step(
                                current_state, active_player, dice, a_w1, a_w2, a_c
                            )
                            
                            # Add strictly novel states to the frontier
                            if next_state not in visited:
                                visited.add(next_state)
                                queue.append(next_state)
                                
    print(f"\nBFS Complete! Total Unique Reachable States Found: {len(visited)}")
    
    # 5. Topologically Sort the DAG
    # Because marks and penalties strictly increase monotonically, the depth of a state
    # (sum of marks + penalties) perfectly stratifies the graph. Sorting by depth ensures 
    # that when evaluating V*(s) in backward induction, V*(s') for all future transitions 
    # has already been computed.
    print("Topologically sorting states by depth...")
    sorted_states = sorted(list(visited), key=get_state_depth)
    
    # Convert to a flat, fast numpy array (32-bit integers) to optimize memory during DP
    dag_array = np.array(sorted_states, dtype=np.int32)
    
    # 6. Cache the Graph
    # We serialize the DAG to disk so the expensive BFS only ever runs once.
    os.makedirs('data', exist_ok=True)
    np.save('data/topological_dag.npy', dag_array)
    
    end_time = time.time()
    print(f"State Space Generation saved to 'data/topological_dag.npy'.")
    print(f"Total Time Elapsed: {round(end_time - start_time, 2)} seconds.")

if __name__ == '__main__':
    generate_state_space()