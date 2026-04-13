"""
analysis/evaluator.py

The Central Simulation Engine for Mini-Qwixx.
This module executes Monte Carlo simulations to evaluate the policies derived 
from Backward Induction (Exact DP) and Reinforcement Learning. 
It strictly adheres to the mathematical rules defined in the Methodology section.
"""

import random
import time
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing as mp
from core.constants import WHITE_ACTIONS, COLOR_ACTIONS
from core.state_encoder import decode_state
from core.environment import MiniQwixxEnv, calculate_score
from solvers.matrix_math import get_nash_probs

# Centralized Memory Registry to hold the Optimal Value functions V(s)
LOADED_POLICIES = {}

def init_tournament_worker(required_agents, custom_paths=None):
    """
    Initializes memory for multiprocessing workers.
    Because the state space |S| is large (565,656 states), we use memory-mapped 
    files (mmap_mode='r') to share the Exact Nash Equilibrium tables (V_tables) 
    across CPU cores without overloading RAM.
    """
    global LOADED_POLICIES
    
    # Cryptographic seeding ensures that parallel simulations do not experience 
    # the exact same sequence of stochastic dice rolls.
    random.seed(os.getpid() + int(time.time() * 1000))
    np.random.seed((os.getpid() + int(time.time() * 1000)) % 4294967295)
    
    paths = {
        'SOLO': 'data/V_solo.npy',
        'SCORE': 'data/V_nash.npy',
        'WIN': 'data/V_nash_win_prob.npy',
        'HYBRID_5': 'data/V_nash_hybrid_5.npy',
        'HYBRID_10': 'data/V_nash_hybrid_10.npy',
        'HYBRID_25': 'data/V_nash_hybrid_25.npy',
        'HYBRID_50': 'data/V_nash_hybrid_50.npy'
    }
    if custom_paths: paths.update(custom_paths)
        
    for agent in required_agents:
        if agent not in LOADED_POLICIES and agent in paths:
            LOADED_POLICIES[agent] = np.load(paths[agent], mmap_mode='r')


def evaluate_state(state, active_idx, is_term, agent_type, evaluating_player):
    """
    Unified State Evaluator.
    Maps an objective function string to its corresponding mathematical formulation.
    """
    if is_term:
        # --- TERMINAL STATE EVALUATION ---
        # Decode the state and compute final points based on Equation 2 (Triangular Scoring)
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        s1 = calculate_score(p1_r, p1_b, p1_p)
        s2 = calculate_score(p2_r, p2_b, p2_p)
        
        # Solo Mode (No zero-sum dynamics, strictly maximizing own points)
        if agent_type == 'SOLO': return s1 if evaluating_player == 1 else -s2
            
        diff = s1 - s2
        
        # Thesis Eq 4: Score Difference Method (V_SD)
        if agent_type == 'SCORE': 
            return diff
            
        # Thesis Eq 3: Win Probability Method (V_WP) -> Maps to {1.0, 0.5, 0.0}
        elif agent_type == 'WIN' or agent_type == 'RL_AGENT': 
            return 1.0 if s1 > s2 else (-1.0 if s1 < s2 else 0.0)
            
        # Thesis Eq 5: Hybrid Method (V_H) -> Score Diff + Phantom Win Bonus (beta)
        elif agent_type.startswith('HYBRID'):
            bonus = float(agent_type.split('_')[1])
            return (diff + bonus) if diff > 0 else ((diff - bonus) if diff < 0 else 0.0)

    # --- NON-TERMINAL STATE EVALUATION ---
    # Retrieve the exact expected value V(s) computed via Backward Induction
    V = LOADED_POLICIES[agent_type]
    if agent_type == 'SOLO': return V[state, active_idx, 0] if evaluating_player == 1 else -V[state, active_idx, 1]
    elif agent_type == 'SCORE': return V[state, active_idx, 0] - V[state, active_idx, 1]
    else: return V[state, active_idx]


def _nash_matchup_chunk(args):
    """
    Simulates a sequence of games between two Exact DP agents.
    At every turn, it constructs a 3x3 normal-form game for the White Phase
    and solves for the exact Nash Equilibrium mixed strategy probabilities.
    """
    num_games, agent_a_type, agent_b_type = args
    stats = {
        'a_as_p1_wins': 0, 'a_as_p2_wins': 0, 'b_as_p1_wins': 0, 'b_as_p2_wins': 0, 'ties': 0,
        'a_as_p1_pts': 0, 'a_as_p2_pts': 0, 'b_as_p1_pts': 0, 'b_as_p2_pts': 0,
        'a_as_p1_margin_sum': 0, 'a_as_p2_margin_sum': 0, 'b_as_p1_margin_sum': 0, 'b_as_p2_margin_sum': 0,
        'total_welfare': 0
    }

    for i in range(num_games):
        state = 0
        active_player = 1
        a_is_p1 = (i % 2 == 0) # Alternate who goes first to ensure fairness
        agent_p1 = agent_a_type if a_is_p1 else agent_b_type
        agent_p2 = agent_b_type if a_is_p1 else agent_a_type

        while True:
            # 1. Terminal Check
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)):
                s1, s2 = calculate_score(p1_r, p1_b, p1_p), calculate_score(p2_r, p2_b, p2_p) 
                stats['total_welfare'] += (s1 + s2)
                
                # Log final statistics
                if a_is_p1:
                    stats['a_as_p1_pts'] += s1; stats['b_as_p2_pts'] += s2
                    if s1 > s2: stats['a_as_p1_wins'] += 1; stats['a_as_p1_margin_sum'] += (s1 - s2)
                    elif s2 > s1: stats['b_as_p2_wins'] += 1; stats['b_as_p2_margin_sum'] += (s2 - s1)
                    else: stats['ties'] += 1
                else:
                    stats['b_as_p1_pts'] += s1; stats['a_as_p2_pts'] += s2
                    if s1 > s2: stats['b_as_p1_wins'] += 1; stats['b_as_p1_margin_sum'] += (s1 - s2)
                    elif s2 > s1: stats['a_as_p2_wins'] += 1; stats['a_as_p2_margin_sum'] += (s2 - s1)
                    else: stats['ties'] += 1
                break

            # Chance Node: Generate stochastic dice roll
            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_idx = 1 if active_player == 1 else 0
            
            # Construct Payoff Matrices M_p1 and M_p2 for the White Phase
            M_p1, M_p2, best_c_dict = np.zeros((3, 3)), np.zeros((3, 3)), {}

            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_c = None
                    best_val = -9999 if active_player == 1 else 9999
                    
                    # Thesis Eq 7: Collapsing the sequential Color Phase into the simultaneous White Phase
                    # Active player computes U1(s, d, a_w1, a_w2) by maximizing over color actions (a_c)
                    for c in COLOR_ACTIONS:
                        ns, term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, c)
                        current_eval_agent = agent_p1 if active_player == 1 else agent_p2
                        val = evaluate_state(ns, next_idx, term, current_eval_agent, active_player)
                        
                        if active_player == 1 and val > best_val: best_val = val; best_c = c
                        elif active_player == 2 and val < best_val: best_val = val; best_c = c

                    best_c_dict[(w1_idx, w2_idx)] = best_c
                    final_ns, final_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, best_c)
                    
                    # Populate matrix elements
                    M_p1[w1_idx, w2_idx] = evaluate_state(final_ns, next_idx, final_term, agent_p1, 1)
                    M_p2[w1_idx, w2_idx] = evaluate_state(final_ns, next_idx, final_term, agent_p2, 2)

            # Thesis Eq 8: Find Subgame Perfect Equilibrium probabilities (p1*, p2*)
            p1_probs, _ = get_nash_probs(M_p1)
            _, p2_probs = get_nash_probs(M_p2)
            
            # Sample actions stochastically from the calculated probability distributions
            idx_w1 = np.random.choice([0,1,2], p=p1_probs)
            idx_w2 = np.random.choice([0,1,2], p=p2_probs)

            # Transition state
            state, _ = MiniQwixxEnv.step(state, active_player, dice, WHITE_ACTIONS[idx_w1], WHITE_ACTIONS[idx_w2], best_c_dict[(idx_w1, idx_w2)])
            active_player = 2 if active_player == 1 else 1
            
    return stats


def _pure_minmax_chunk(args):
    """
    Worker for RL vs Exact DP Evaluation.
    Unlike DP vs DP (which uses mixed strategies), Standard Q-learning outputs a 
    pure deterministic policy (argmax Q). This function evaluates the RL agent 
    using strict Pure Min-Max selection to test its robustness against perfect play.
    """
    chunk_size, agent_rl, agent_dp = args
    local_rl_wins = 0
    
    for _ in range(chunk_size):
        state = 0
        active_player = random.choice([1, 2]) 
        
        while True:
            p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
            if p1_p >= 3 or p2_p >= 3 or (MiniQwixxEnv.is_row_locked(p1_r, p2_r) and MiniQwixxEnv.is_row_locked(p1_b, p2_b)): break
                
            dice = {'W1': random.randint(1, 3), 'W2': random.randint(1, 3), 'R': random.randint(1, 3), 'B': random.randint(1, 3)}
            next_active_idx = 1 if active_player == 1 else 0
            
            # Formulate Payoff Matrices (RL values vs DP expected values)
            rl_payoff = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_val = -9999.0
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        val = evaluate_state(next_s, next_active_idx, is_term, agent_rl, active_player)
                        if val > best_val: best_val = val
                    rl_payoff[w1_idx, w2_idx] = best_val

            dp_payoff = np.zeros((3, 3), dtype=np.float32)
            for w1_idx, a_w1 in enumerate(WHITE_ACTIONS):
                for w2_idx, a_w2 in enumerate(WHITE_ACTIONS):
                    best_val = 9999.0
                    for a_c in COLOR_ACTIONS:
                        next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1, a_w2, a_c)
                        val = evaluate_state(next_s, next_active_idx, is_term, agent_dp, active_player)
                        if val < best_val: best_val = val
                    dp_payoff[w1_idx, w2_idx] = best_val

            # RL acts deterministically: Maximin criteria for purely adversarial play
            a_w1_chosen = WHITE_ACTIONS[np.argmax(np.min(rl_payoff, axis=1))]
            a_w2_chosen = WHITE_ACTIONS[np.argmin(np.max(dp_payoff, axis=0))]

            # Resolve Sequential Color phase optimally
            best_final_c = None
            if active_player == 1: 
                best_val = -9999.0
                for a_c in COLOR_ACTIONS:
                    next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, a_c)
                    val = evaluate_state(next_s, next_active_idx, is_term, agent_rl, active_player)
                    if val > best_val: best_val = val; best_final_c = a_c
            else: 
                best_val = 9999.0
                for a_c in COLOR_ACTIONS:
                    next_s, is_term = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, a_c)
                    val = evaluate_state(next_s, next_active_idx, is_term, agent_dp, active_player)
                    if val < best_val: best_val = val; best_final_c = a_c

            state, _ = MiniQwixxEnv.step(state, active_player, dice, a_w1_chosen, a_w2_chosen, best_final_c)
            active_player = 2 if active_player == 1 else 1
            
        p1_r, p1_b, p1_p, p2_r, p2_b, p2_p = decode_state(state)
        if calculate_score(p1_r, p1_b, p1_p) > calculate_score(p2_r, p2_b, p2_p): local_rl_wins += 1
            
    return local_rl_wins


class TournamentEngine:
    """
    Standardized orchestrator for empirical analysis.
    Uses Monte Carlo sampling parallelized across available CPU cores 
    to approximate the expected value of strategies in competition.
    """
    
    @staticmethod
    def run_nash_matchup(agent_a, agent_b, num_games):
        cores = mp.cpu_count()
        games_per_core = [num_games // cores] * cores
        for i in range(num_games % cores): games_per_core[i] += 1
        
        args = [(n, agent_a, agent_b) for n in games_per_core]
        
        with mp.Pool(processes=cores, initializer=init_tournament_worker, initargs=([agent_a, agent_b],)) as pool:
            results = pool.map(_nash_matchup_chunk, args)
            
        final_stats = {k: 0 for k in results[0].keys()}
        for r in results:
            for k in final_stats.keys(): final_stats[k] += r[k]
                
        return final_stats

    @staticmethod
    def run_pure_minmax_matchup(agent_rl, agent_dp, custom_paths, num_games):
        cores = mp.cpu_count()
        games_per_core = [num_games // cores] * cores
        for i in range(num_games % cores): games_per_core[i] += 1
        
        args = [(n, agent_rl, agent_dp) for n in games_per_core]
        
        with mp.Pool(processes=cores, initializer=init_tournament_worker, initargs=([agent_rl, agent_dp], custom_paths)) as pool:
            results = pool.map(_pure_minmax_chunk, args)
            
        return sum(results)