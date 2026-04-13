"""
core/constants.py

Single Source of Truth for Mini-Qwixx Formal Environment Variables.
This module defines the boundaries of the state space |S| and the discrete 
action spaces A_w (White Phase) and A_c (Color Phase) used in the Markov 
Decision Process (MDP).
"""

# Precomputed mapping for the Scoring Function.
# Maps the formal Row ID (representing the right-most crossed box) to the 
# total number of valid crosses in that row. This is used strictly for evaluating 
# Equation 2 (Triangular Number Scoring Rule) at terminal states.
ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

# --- The Discrete Action Spaces ---
# A_w: The simultaneous action space for both players during the White Phase.
WHITE_ACTIONS = ['R', 'B', None]

# A_c: The sequential action space for the active player during the Color Phase.
# Formulated as tuples indicating the target color row and the specific white die used.
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

# The absolute upper bound of the encoded integer state space.
# While the true reachable state space |S| is exactly 565,656 (as proven by BFS),
# the bitwise encoding scheme requires a flat memory allocation of 2^20 to guarantee
# O(1) mathematical matrix lookups during backward induction and RL iterations.
TOTAL_STATES = 1048576