import numpy as np
import random
from core.state_encoder import encode_state, decode_state
from core.constants import ROW_ID_TO_COUNT
from collections import Counter

def calculate_score(r_id, b_id, penalties):
    """Calculates the exact Qwixx score given row IDs and penalties."""
    count_r, count_b = ROW_ID_TO_COUNT[r_id], ROW_ID_TO_COUNT[b_id]
    if r_id >= 11: count_r += 1
    if b_id >= 11: count_b += 1
    return ((count_r * (count_r + 1)) // 2) + ((count_b * (count_b + 1)) // 2) - (3 * penalties)

def roll_dice():
    """Returns a random dice roll dictionary for Mini-Qwixx."""
    return {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 
            'R': random.randint(1, 3), 'B': random.randint(1, 3)}

def _generate_unique_dice_combinations():
    """Compresses 81 possible D3 rolls down to 54 unique permutations with probabilities."""
    combinations = []
    for w1 in [1, 2, 3]:
        for w2 in [1, 2, 3]:
            for r in [1, 2, 3]:
                for b in [1, 2, 3]:
                    w_tuple = tuple(sorted([w1, w2]))
                    combinations.append((w_tuple[0], w_tuple[1], r, b))
    counts = Counter(combinations)
    return [{'W1': d[0], 'W2': d[1], 'R': d[2], 'B': d[3], 'prob': count / 81.0} for d, count in counts.items()]

# Compute this once globally so other scripts can import it
UNIQUE_DICE = _generate_unique_dice_combinations()

def get_state_depth(state_int):
    """Calculates the topological depth (Marks + Penalties) of a state."""
    p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
    return ROW_ID_TO_COUNT[p1_r] + ROW_ID_TO_COUNT[p1_b] + p1_p + \
           ROW_ID_TO_COUNT[p2_r] + ROW_ID_TO_COUNT[p2_b] + p2_p

# Maps (rightmost_index, count) to a unique Row ID (0 to 13)
def get_row_id(idx, count):
    if idx == -1: return 0
    if idx == 0: return 1
    if idx == 1: return count + 1
    if idx == 2: return count + 3
    if idx == 3: return count + 6
    if idx == 4: return count + 7  # Lock box (counts 4, 5, 6 map to 11, 12, 13)
    return -1

# Decode Row ID back to (rightmost_index, count)
def get_row_details(row_id):
    if row_id == 0: return -1, 0
    if row_id == 1: return 0, 1
    if 2 <= row_id <= 3: return 1, row_id - 1
    if 4 <= row_id <= 6: return 2, row_id - 3
    if 7 <= row_id <= 10: return 3, row_id - 6
    if 11 <= row_id <= 13: return 4, row_id - 7
    return -1, 0

# Build the 14x5 Transition Matrix
# ROW_TRANSITIONS[current_row_id][target_box_index] = new_row_id (-1 if invalid)
ROW_TRANSITIONS = np.full((14, 5), -1, dtype=np.int32)

for current_id in range(14):
    curr_idx, curr_count = get_row_details(current_id)
    
    for target_idx in range(5):
        # Rule 1: Must move left to right
        if target_idx <= curr_idx:
            continue
            
        # Rule 2: To mark the Lock box (index 4), must have >= 2 previous marks
        if target_idx == 4 and curr_count < 2:
            continue
            
        # Calculate new count (marking the lock box gives +1 mark, and +1 bonus)
        new_count = curr_count + 1
        if target_idx == 4:
            new_count += 1 
            
        ROW_TRANSITIONS[current_id][target_idx] = get_row_id(target_idx, new_count)


# ==========================================
# 2. GAME ENVIRONMENT LOGIC
# ==========================================
class MiniQwixxEnv:
    
    @staticmethod
    def is_row_locked(p1_row_id, p2_row_id):
        """Returns True if either player has reached the lock box (ID 11, 12, or 13)"""
        return (p1_row_id >= 11) or (p2_row_id >= 11)

    @staticmethod
    def step(state_int, active_player, dice, a_w1, a_w2, a_c):
        """
        Takes the current state, the dice roll, and the chosen actions.
        Returns the new state integer and a boolean indicating if the game is over.
        
        Actions:
        a_w1, a_w2: 'R', 'B', or None (White Phase)
        a_c: 'R1', 'R2', 'B1', 'B2', or None (Color Phase for active player)
        """
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state_int)
        
        # Check if rows are currently locked
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
        
        # Helper to map dice sum to box index
        def get_box_idx(color, dice_sum):
            if color == 'R': return dice_sum - 2  # Red: 2->0, 3->1, 4->2, 5->3, 6->4
            if color == 'B': return 6 - dice_sum  # Blue: 6->0, 5->1, 4->2, 3->3, 2->4
            return -1

        # ------------------------------------------------
        # 1. Resolve White Phase (Simultaneous)
        # ------------------------------------------------
        white_sum = dice['W1'] + dice['W2']
        
        # Player 1 White Action
        p1_marked_white = False
        if a_w1 == 'R' and not red_locked:
            new_r = ROW_TRANSITIONS[p1_r][get_box_idx('R', white_sum)]
            if new_r != -1: 
                p1_r = new_r
                p1_marked_white = True
        elif a_w1 == 'B' and not blue_locked:
            new_b = ROW_TRANSITIONS[p1_b][get_box_idx('B', white_sum)]
            if new_b != -1: 
                p1_b = new_b
                p1_marked_white = True

        # Player 2 White Action
        p2_marked_white = False
        if a_w2 == 'R' and not red_locked:
            new_r = ROW_TRANSITIONS[p2_r][get_box_idx('R', white_sum)]
            if new_r != -1: 
                p2_r = new_r
                p2_marked_white = True
        elif a_w2 == 'B' and not blue_locked:
            new_b = ROW_TRANSITIONS[p2_b][get_box_idx('B', white_sum)]
            if new_b != -1: 
                p2_b = new_b
                p2_marked_white = True

        # Update Locks immediately in case someone locked a row in the white phase
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)

        # ------------------------------------------------
        # 2. Resolve Color Phase (Sequential, Active Player Only)
        # ------------------------------------------------
        active_marked_color = False
        
        if a_c is not None:
            c_color = a_c[0]  # 'R' or 'B'
            w_die = dice['W1'] if a_c[1] == '1' else dice['W2']
            c_die = dice[c_color]
            color_sum = w_die + c_die
            
            if active_player == 1:
                if c_color == 'R' and not red_locked:
                    new_r = ROW_TRANSITIONS[p1_r][get_box_idx('R', color_sum)]
                    if new_r != -1:
                        p1_r = new_r
                        active_marked_color = True
                elif c_color == 'B' and not blue_locked:
                    new_b = ROW_TRANSITIONS[p1_b][get_box_idx('B', color_sum)]
                    if new_b != -1:
                        p1_b = new_b
                        active_marked_color = True
            else:
                if c_color == 'R' and not red_locked:
                    new_r = ROW_TRANSITIONS[p2_r][get_box_idx('R', color_sum)]
                    if new_r != -1:
                        p2_r = new_r
                        active_marked_color = True
                elif c_color == 'B' and not blue_locked:
                    new_b = ROW_TRANSITIONS[p2_b][get_box_idx('B', color_sum)]
                    if new_b != -1:
                        p2_b = new_b
                        active_marked_color = True

        # ------------------------------------------------
        # 3. Resolve Penalties
        # ------------------------------------------------
        # Active player gets a penalty ONLY if they marked nothing in both phases
        if active_player == 1:
            if not p1_marked_white and not active_marked_color:
                p1_p += 1
        else:
            if not p2_marked_white and not active_marked_color:
                p2_p += 1

        # ------------------------------------------------
        # 4. Check Termination
        # ------------------------------------------------
        # Re-check locks in case the color phase triggered one
        red_locked = MiniQwixxEnv.is_row_locked(p1_r, p2_r)
        blue_locked = MiniQwixxEnv.is_row_locked(p1_b, p2_b)
        
        is_terminal = False
        if p1_p >= 3 or p2_p >= 3:
            is_terminal = True
        elif red_locked and blue_locked:
            is_terminal = True

        new_state_int = encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p)
        return new_state_int, is_terminal