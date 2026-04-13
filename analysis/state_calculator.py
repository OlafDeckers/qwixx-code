import time
import itertools
from collections import Counter, deque
import numpy as np

class QwixxStateCalculator:
    def __init__(self, num_colors=2, dice_sides=3, lock_threshold=2, max_penalties=3, locks_to_end=2):
        self.num_colors = num_colors
        self.dice_sides = dice_sides
        self.lock_threshold = lock_threshold
        self.max_penalties = max_penalties
        self.locks_to_end = locks_to_end
        
        self.num_boxes = (self.dice_sides * 2) - 1
        
        print("="*60)
        print(f" INITIALIZING QWIXX STATE CALCULATOR")
        print("="*60)
        print(f" Parameters:")
        print(f"  - Colors: {self.num_colors}")
        print(f"  - Dice Sides: D{self.dice_sides}")
        print(f"  - Boxes per row: {self.num_boxes} (Sums {2} to {self.dice_sides * 2})")
        print(f"  - Marks needed to lock: {self.lock_threshold}")
        print(f"  - Max Penalties: {self.max_penalties}")
        print(f"  - Locked rows to end game: {self.locks_to_end}")
        print("-" * 60)
        
        self._build_row_transitions()
        self._generate_unique_dice()

    def _build_row_transitions(self):
        """Builds the generalized 1D transition matrix for a single row."""
        self.state_to_id = {}
        self.id_to_state = {}
        current_id = 0
        
        # Define all valid (rightmost_index, mark_count) pairs
        for idx in range(-1, self.num_boxes):
            if idx == -1: # Empty Row
                self.state_to_id[(-1, 0)] = current_id
                self.id_to_state[current_id] = (-1, 0)
                current_id += 1
            elif idx < self.num_boxes - 1: # Standard Boxes
                for count in range(1, idx + 2):
                    self.state_to_id[(idx, count)] = current_id
                    self.id_to_state[current_id] = (idx, count)
                    current_id += 1
            else: # The Lock Box
                for prev_count in range(self.lock_threshold, self.num_boxes):
                    count = prev_count + 2 # +1 for the box, +1 for the lock bonus
                    self.state_to_id[(idx, count)] = current_id
                    self.id_to_state[current_id] = (idx, count)
                    current_id += 1

        self.total_row_states = current_id
        print(f" Distinct mathematical states per single row: {self.total_row_states}")
        
        # Build the transition table
        self.transitions = np.full((current_id, self.num_boxes), -1, dtype=np.int32)
        self.locked_ids = set()
        
        for sid in range(current_id):
            idx, count = self.id_to_state[sid]
            if idx == self.num_boxes - 1:
                self.locked_ids.add(sid)
                
            for target_idx in range(self.num_boxes):
                if target_idx <= idx: continue # Must move right
                if target_idx == self.num_boxes - 1 and count < self.lock_threshold: continue
                
                new_count = count + 1
                if target_idx == self.num_boxes - 1: new_count += 1
                    
                self.transitions[sid][target_idx] = self.state_to_id[(target_idx, new_count)]

    def _generate_unique_dice(self):
        """Generates all mathematically unique permutations of the dice."""
        faces = list(range(1, self.dice_sides + 1))
        combinations = []
        for w_combo in itertools.product(faces, repeat=2):
            w1, w2 = sorted(w_combo) # W1 and W2 are identical
            for c_combo in itertools.product(faces, repeat=self.num_colors):
                combinations.append((w1, w2) + c_combo)
                
        counts = Counter(combinations)
        self.unique_dice = []
        for d in counts.keys():
            dice_dict = {'w1': d[0], 'w2': d[1]}
            for i in range(self.num_colors): dice_dict[f'c{i}'] = d[2 + i]
            self.unique_dice.append(dice_dict)
        print(f" Unique Dice Permutations (Probability Groups): {len(self.unique_dice)}\n")

    def _get_box_idx(self, color_idx, dice_sum):
        """Determines the 0-indexed position of a dice sum (Evens=Ascending, Odds=Descending)"""
        if color_idx % 2 == 0:
            return dice_sum - 2
        else:
            return (self.dice_sides * 2) - dice_sum

    def get_active_moves(self, p_state, dice, locks):
        """Optimized generator for Player 1's sequential White -> Color moves."""
        next_states = set()
        w_sum = dice['w1'] + dice['w2']
        
        w_results = [(p_state, False, locks)] # Format: (state, did_mark_white, current_locks)
        
        # 1. White Phase Options
        for c in range(self.num_colors):
            if not locks[c]:
                idx = self._get_box_idx(c, w_sum)
                new_r = self.transitions[p_state[c]][idx]
                if new_r != -1:
                    new_s, new_l = list(p_state), list(locks)
                    new_s[c] = new_r
                    if idx == self.num_boxes - 1: new_l[c] = True
                    w_results.append((tuple(new_s), True, tuple(new_l)))
                    
        # 2. Color Phase Options
        for w_state, w_marked, w_locks in w_results:
            if not w_marked:
                s = list(w_state)
                s[-1] += 1 # Add Penalty
                next_states.add(tuple(s))
            else:
                next_states.add(w_state)
                
            for c in range(self.num_colors):
                if not w_locks[c]:
                    for w_val in (dice['w1'], dice['w2']):
                        c_sum = w_val + dice[f'c{c}']
                        idx = self._get_box_idx(c, c_sum)
                        new_r = self.transitions[w_state[c]][idx]
                        if new_r != -1:
                            s = list(w_state)
                            s[c] = new_r # No penalty applied because color was marked
                            next_states.add(tuple(s))
        return next_states

    def get_passive_moves(self, p_state, dice, locks):
        """Optimized generator for Player 2's simultaneous White Phase."""
        next_states = set([p_state])
        w_sum = dice['w1'] + dice['w2']
        for c in range(self.num_colors):
            if not locks[c]:
                idx = self._get_box_idx(c, w_sum)
                new_r = self.transitions[p_state[c]][idx]
                if new_r != -1:
                    s = list(p_state)
                    s[c] = new_r
                    next_states.add(tuple(s))
        return next_states

    def run_calculation(self):
        print("Starting Reachability Graph Search (BFS)...")
        start_time = time.time()
        
        start_player_state = tuple([0] * self.num_colors + [0])
        start_state = start_player_state + start_player_state
        
        visited = set([start_state])
        queue = deque([start_state])
        processed = 0
        
        while queue:
            current = queue.popleft()
            processed += 1
            if processed % 50000 == 0:
                print(f"  ... Traversed {processed} states. Queue size: {len(queue)}")
                
            p1 = current[:self.num_colors+1]
            p2 = current[self.num_colors+1:]
            
            # Check Terminations
            locks = tuple((p1[c] in self.locked_ids) or (p2[c] in self.locked_ids) for c in range(self.num_colors))
            if p1[-1] >= self.max_penalties or p2[-1] >= self.max_penalties or sum(locks) >= self.locks_to_end:
                continue

            # Simulate both players being active
            for active in (1, 2):
                for dice in self.unique_dice:
                    if active == 1:
                        p1_next = self.get_active_moves(p1, dice, locks)
                        p2_next = self.get_passive_moves(p2, dice, locks)
                        for s1 in p1_next:
                            for s2 in p2_next:
                                new_s = s1 + s2
                                if new_s not in visited: visited.add(new_s); queue.append(new_s)
                    else:
                        p1_next = self.get_passive_moves(p1, dice, locks)
                        p2_next = self.get_active_moves(p2, dice, locks)
                        for s1 in p1_next:
                            for s2 in p2_next:
                                new_s = s1 + s2
                                if new_s not in visited: visited.add(new_s); queue.append(new_s)

        end_time = time.time()
        print("\n" + "="*60)
        print(f" SEARCH COMPLETE IN {end_time - start_time:.2f} SECONDS")
        print(f" TOTAL REACHABLE UNIQUE STATES: {len(visited)}")
        print("="*60)

if __name__ == '__main__':
    # Using your exact Mini-Qwixx defaults
    calculator = QwixxStateCalculator(
        num_colors=2, 
        dice_sides=4, 
        lock_threshold=2, 
        max_penalties=3,
        locks_to_end=2
    )
    calculator.run_calculation()