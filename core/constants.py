"""
core/constants.py
Single Source of Truth for Mini-Qwixx Environment Variables
"""

# Precomputed mappings for Qwixx row scoring based on state IDs
ROW_ID_TO_COUNT = [0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5]

# Available actions in the Mini-Qwixx environment
WHITE_ACTIONS = ['R', 'B', None]
COLOR_ACTIONS = [('R', '1'), ('R', '2'), ('B', '1'), ('B', '2'), None]

# Total size of the Mini-Qwixx state space (used for shared memory allocation)
TOTAL_STATES = 1048576