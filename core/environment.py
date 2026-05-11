"""
core/environment.py

The Markov Decision Process (MDP) Formulation for Mini-Qwixx.
This module strictly enforces the game rules, acting as the transition function 
T(s, a, d) -> s'. It also handles state encoding, scoring (Equation 2), and the 
probability distributions of the stochastic chance nodes.
"""

import numpy as np
import random
from numba import njit
from core.state_encoder import encode_state, decode_state
from core.constants import ROW_ID_TO_COUNT
from collections import Counter

# Convert strictly to a 32-bit NumPy array for blazing fast Numba memory lookups
ROW_ID_TO_COUNT_ARR = np.array(ROW_ID_TO_COUNT, dtype=np.int32)

@njit(nogil=True)
def calculate_score(r_id, b_id, penalties):
    """
    Thesis Reference: Equation 2 (Triangular Number Scoring Rule).
    Computes the terminal score for a player based on their crossed boxes and penalties.
    Score = [ c_red(c_red + 1)/2 ] + [ c_blue(c_blue + 1)/2 ] - 3*p
    """
    count_r, count_b = ROW_ID_TO_COUNT_ARR[r_id], ROW_ID_TO_COUNT_ARR[b_id]
    
    # If the row is locked (ID >= 11), the player receives a +1 cross bonus
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
        
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def roll_dice():
    """Generates the stochastic variable d at a chance node."""
    return {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
            'R': random.randint(1, 3), 'B': random.randint(1, 3)}

def _generate_unique_dice_combinations():
    """
    Computes the exact probability distribution P(d) for the chance nodes.
    A D3 dice roll yields 3^4 = 81 total outcomes. Because the two white dice 
    (W1, W2) are indistinguishable, we collapse symmetrical permutations into 
    54 unique outcomes, weighting them by their combinatorial probability.
    """
    combinations = []
    for w1 in [1, 2, 3]:
        for w2 in [1, 2, 3]:
            for r in [1, 2, 3]:
                for b in [1, 2, 3]:
                    w_tuple = tuple(sorted([w1, w2]))
                    combinations.append((w_tuple[0], w_tuple[1], r, b))
    counts = Counter(combinations)
    return [{'W1': d[0], 'W2': d[1], 'R': d[2], 'B': d[3], 'prob': count / 81.0} for d, count in counts.items()]

# Compute the probability space once globally for the solver algorithms
UNIQUE_DICE = _generate_unique_dice_combinations()

@njit(nogil=True)
def get_state_depth(state_int):
    """
    Calculates the topological depth of state s ∈ S.
    The strict left-to-right marking rule and monotonically increasing penalties 
    guarantee the state space forms a Directed Acyclic Graph (DAG). 
    Depth = Total Marks + Total Penalties.
    """
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    return ROW_ID_TO_COUNT_ARR[p1_r] + ROW_ID_TO_COUNT_ARR[p1_b] + p1_p + \
           ROW_ID_TO_COUNT_ARR[p2_r] + ROW_ID_TO_COUNT_ARR[p2_b] + p2_p

# ==========================================
# 1. STATE SPACE ENCODING (The Tuple representation)
# ==========================================
# To evaluate a row optimally, the MDP only needs two pieces of information: 
# the right-most crossed index (dictating legal future moves) and the total 
# number of crossed boxes (for terminal scoring). 
# We map these (index, count) pairs to a single integer ID [0 to 13].

# Array mapping: ROW_DETAILS_ARRAY[row_id] = [rightmost_index, count]
# Converts O(N) if/elif branches into O(1) CPU pointer lookups
ROW_DETAILS_ARRAY = np.array([
    [-1, 0], [0, 1], [1, 1], [1, 2], [2, 1], [2, 2], [2, 3],
    [3, 1], [3, 2], [3, 3], [3, 4], [4, 4], [4, 5], [4, 6]
], dtype=np.int32)

@njit(nogil=True)
def get_row_id(idx, count):
    """Maps a formal (rightmost_index, count) state to its composite Row ID."""
    if idx == -1: return 0
    if idx == 0: return 1
    if idx == 1: return count + 1
    if idx == 2: return count + 3
    if idx == 3: return count + 6
    if idx == 4: return count + 7  # Lock box (counts 4, 5, 6 map to 11, 12, 13)
    return -1

@njit(nogil=True)
def get_row_details(row_id):
    """Decodes a composite Row ID back to its (rightmost_index, count) state."""
    return ROW_DETAILS_ARRAY[row_id][0], ROW_DETAILS_ARRAY[row_id][1]


# ==========================================
# 2. TRANSITION MATRIX (Row-Level Irreversibility)
# ==========================================
# Precomputes the deterministic transition matrix for a single row.
# This matrix physically enforces the irreversibility of the DAG.
# ROW_TRANSITIONS[current_row_id][target_box_index] = new_row_id (-1 if invalid)
ROW_TRANSITIONS = np.full((14, 5), -1, dtype=np.int32)

for current_id in range(14):
    curr_idx, curr_count = get_row_details(current_id)
    
    for target_idx in range(5):
        # DAG Constraint 1: Marks must strictly progress left-to-right
        if target_idx <= curr_idx:
            continue
            
        # DAG Constraint 2: The final lock box requires a minimum of 2 prior marks
        if target_idx == 4 and curr_count < 2:
            continue
            
        # Calculate new count (marking the lock box gives +1 mark, and +1 bonus)
        new_count = curr_count + 1
        if target_idx == 4:
            new_count += 1 
            
        ROW_TRANSITIONS[current_id][target_idx] = get_row_id(target_idx, new_count)


# ==========================================
# 3. GAME ENVIRONMENT LOGIC (The Transition Function T)
# ==========================================
class MiniQwixxEnv:
    
    @staticmethod
    def is_row_locked(p1_row_id, p2_row_id):
        """Returns True if either player has reached the terminal lock box (ID >= 11)."""
        return (p1_row_id >= 11) or (p2_row_id >= 11)

    @staticmethod
    def step(state_int, active_player, dice, a_w1, a_w2, a_c):
        """
        The formal Transition Function T(s, d, a_w, a_c) -> s'.
        Processes the simultaneous White Phase and sequential Color Phase.
        """
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
        
        # Determine current lock status (Inlined logic for O(1) performance)
        red_locked = (p1_r >= 11) or (p2_r >= 11)
        blue_locked = (p1_b >= 11) or (p2_b >= 11)

        # ------------------------------------------------
        # 1. Resolve White Phase (Simultaneous Decision)
        # ------------------------------------------------
        white_sum = dice['W1'] + dice['W2']
        
        # Resolve Player 1 White Action
        p1_marked_white = False
        if a_w1 == 'R' and not red_locked:
            # Inline Affine Transformation (dice_sum - 2) avoids function call overhead
            new_r = ROW_TRANSITIONS[p1_r][white_sum - 2]
            if new_r != -1: 
                p1_r = new_r
                p1_marked_white = True
        elif a_w1 == 'B' and not blue_locked:
            # Inline Affine Transformation (6 - dice_sum) avoids function call overhead
            new_b = ROW_TRANSITIONS[p1_b][6 - white_sum]
            if new_b != -1: 
                p1_b = new_b
                p1_marked_white = True

        # Resolve Player 2 White Action
        p2_marked_white = False
        if a_w2 == 'R' and not red_locked:
            new_r = ROW_TRANSITIONS[p2_r][white_sum - 2]
            if new_r != -1: 
                p2_r = new_r
                p2_marked_white = True
        elif a_w2 == 'B' and not blue_locked:
            new_b = ROW_TRANSITIONS[p2_b][6 - white_sum]
            if new_b != -1: 
                p2_b = new_b
                p2_marked_white = True

        # Re-evaluate lock constraints immediately after the White Phase
        red_locked = (p1_r >= 11) or (p2_r >= 11)
        blue_locked = (p1_b >= 11) or (p2_b >= 11)

        # ------------------------------------------------
        # 2. Resolve Color Phase (Sequential, Active Player Only)
        # ------------------------------------------------
        active_marked_color = False
        
        if a_c is not None:
            c_color = a_c[0]  # 'R' or 'B'
            w_die = dice['W1'] if a_c[1] == '1' else dice['W2']
            color_sum = w_die + dice[c_color]
            
            # Inline Box Array logic
            box_idx = color_sum - 2 if c_color == 'R' else 6 - color_sum
            
            if active_player == 1:
                if c_color == 'R' and not red_locked:
                    new_r = ROW_TRANSITIONS[p1_r][box_idx]
                    if new_r != -1:
                        p1_r = new_r
                        active_marked_color = True
                elif c_color == 'B' and not blue_locked:
                    new_b = ROW_TRANSITIONS[p1_b][box_idx]
                    if new_b != -1:
                        p1_b = new_b
                        active_marked_color = True
            else:
                if c_color == 'R' and not red_locked:
                    new_r = ROW_TRANSITIONS[p2_r][box_idx]
                    if new_r != -1:
                        p2_r = new_r
                        active_marked_color = True
                elif c_color == 'B' and not blue_locked:
                    new_b = ROW_TRANSITIONS[p2_b][box_idx]
                    if new_b != -1:
                        p2_b = new_b
                        active_marked_color = True

        # ------------------------------------------------
        # 3. Resolve Penalties
        # ------------------------------------------------
        # The Active Player receives an immutable penalty ONLY if they 
        # failed to mark any box during BOTH the White and Color phases.
        if active_player == 1:
            if not p1_marked_white and not active_marked_color:
                p1_p += 1
        else:
            if not p2_marked_white and not active_marked_color:
                p2_p += 1

        # ------------------------------------------------
        # 4. Check Terminal State (s ∈ S_terminal)
        # ------------------------------------------------
        # The game reaches a terminal absorbing state if either player accrues 
        # 3 penalties, or if both color rows become locked.
        is_terminal = False
        if p1_p >= 3 or p2_p >= 3:
            is_terminal = True
        elif ((p1_r >= 11) or (p2_r >= 11)) and ((p1_b >= 11) or (p2_b >= 11)):
            is_terminal = True

        new_state_int = encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p)
        return new_state_int, is_terminal