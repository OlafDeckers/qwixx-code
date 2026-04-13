"""
core/state_encoder.py

State Space Isomorphism for Matrix Operations.
This module provides a bijective mapping between the formal state tuple s ∈ S
(defined in Equation 1) and a unique integer index. This O(1) mapping is 
computationally necessary to perform array-based value iteration and exact 
Nash equilibrium lookups across millions of episodes.
"""

import numpy as np

def encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p):
    """
    Thesis Reference: Equation 1 -> s = (r_1, b_1, p_1, r_2, b_2, p_2)
    Maps the multi-dimensional game state tuple to a singular 32-bit integer.
    
    By using bitwise left-shifts (<<) and bitwise ORs (|), we tightly pack 
    the variables into a single integer without overlapping memory bounds.
    - Row states (r, b) require 4 bits (values 0-14).
    - Penalties (p) require 2 bits (values 0-3).
    """
    return (p1_r) | \
           (p1_b << 4) | \
           (p1_p << 8) | \
           (p2_r << 10) | \
           (p2_b << 14) | \
           (p2_p << 18)


def decode_state(state_int):
    """
    Inverse mapping: Reconstructs the formal state tuple s from its integer index.
    This allows the Transition Function T(s, a, d) to parse the exact game board.
    
    Uses bitwise right-shifts (>>) and bitwise ANDs (&) as a bitmask to isolate 
    and extract the specific variables.
    - 0xF (binary 1111) isolates a 4-bit integer (Row State).
    - 0x3 (binary 0011) isolates a 2-bit integer (Penalties).
    """
    p1_r = state_int & 0xF           # Player 1 Red Row ID
    p1_b = (state_int >> 4) & 0xF    # Player 1 Blue Row ID
    p1_p = (state_int >> 8) & 0x3    # Player 1 Penalty Count
    
    p2_r = (state_int >> 10) & 0xF   # Player 2 Red Row ID
    p2_b = (state_int >> 14) & 0xF   # Player 2 Blue Row ID
    p2_p = (state_int >> 18) & 0x3   # Player 2 Penalty Count
    
    return p1_r, p1_b, p1_p, p2_r, p2_b, p2_p