import numpy as np

def encode_state(p1_r, p1_b, p1_p, p2_r, p2_b, p2_p):
    """
    p1_r, p1_b, p2_r, p2_b are integers from 0-14.
    p1_p, p2_p are integers from 0-3 (penalties).
    """
    return (p1_r) | \
           (p1_b << 4) | \
           (p1_p << 8) | \
           (p2_r << 10) | \
           (p2_b << 14) | \
           (p2_p << 18)

def decode_state(state_int):
    p1_r = state_int & 0xF           # 0xF is 15 (4 bits)
    p1_b = (state_int >> 4) & 0xF
    p1_p = (state_int >> 8) & 0x3    # 0x3 is 3 (2 bits)
    p2_r = (state_int >> 10) & 0xF
    p2_b = (state_int >> 14) & 0xF
    p2_p = (state_int >> 18) & 0x3
    return p1_r, p1_b, p1_p, p2_r, p2_b, p2_p