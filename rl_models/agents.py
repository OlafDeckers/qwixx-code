"""
rl_models/agents.py

Reinforcement Learning Architectures for Markov Games.
This module defines the specific learning algorithms evaluated in the thesis.
It uses the Strategy Design Pattern to cleanly separate the mathematical learning 
rules (Reward Shaping, TD-Lambda, Boltzmann) from the generalized multiprocessing 
Minimax-Q training loop.
"""

import numpy as np
import random

class BaseAgent:
    """
    Base Strategy class implementing Standard Minimax-Q Learning.
    This acts as the baseline Off-Policy Temporal Difference (TD) algorithm.
    """
    
    def reset_episode(self):
        """Re-initializes episodic memory (used primarily by traces)."""
        pass

    def get_future_value(self, is_term, s1, s2, shared_V, next_state, next_idx, env_info):
        """
        Extracts the bootstrapped value estimate V(s_{t+1}) for the Bellman equation.
        For terminal states, it returns the absolute win/loss scalar {1.0, -1.0, 0.0} 
        (Thesis Eq 3: Win Probability).
        """
        if is_term:
            if s1 > s2: return 1.0    # Player 1 Wins
            elif s1 < s2: return -1.0 # Player 2 Wins
            return 0.0                # Tie
        return float(shared_V[next_state, next_idx])

    def update_value(self, state, active_idx, v_target, alpha, shared_V):
        """
        Standard Off-Policy Temporal Difference (TD) update rule.
        V(s) <- V(s) + alpha * [V_target - V(s)]
        Where V_target is the Minimax value of the next state's subgame.
        """
        shared_V[state, active_idx] += alpha * (v_target - shared_V[state, active_idx])

    def select_actions(self, payoff_matrix, white_actions, best_c_actions, param_val):
        """
        Standard Epsilon-Greedy fallback for action selection.
        (Note: The pure greedy path is handled inside the training loop; this 
        function specifically returns the exploratory random uniform actions).
        """
        return random.choice(white_actions), random.choice(white_actions)


class RewardShapingAgent(BaseAgent):
    """
    Implements Heuristic Reward Shaping (Thesis Equation 15).
    Standard Qwixx has a highly sparse reward structure (rewards only occur at 
    terminal states). This agent modifies the MDP by injecting dense heuristic 
    rewards into the intermediate state transitions to guide the gradient.
    """
    
    def get_future_value(self, is_term, s1, s2, shared_V, next_state, next_idx, env_info):
        if is_term:
            return super().get_future_value(is_term, s1, s2, shared_V, next_state, next_idx, env_info)
        
        # Unpack environment tracking variables to check for state deltas
        active_player, np1_p, p1_p, np2_p, p2_p, c_np1_r, c_p1_r, c_np1_b, c_p1_b, c_np2_r, c_p2_r, c_np2_b, c_p2_b = env_info.values()
        
        step_reward = 0.0
        
        # Heuristic 1: Penalties are highly detrimental to win probability.
        # Apply a +/- 0.05 value shift if a penalty was taken this turn.
        if active_player == 1 and np1_p > p1_p: step_reward -= 0.05
        elif active_player == 2 and np2_p > p2_p: step_reward += 0.05 
        
        # Heuristic 2: Marking boxes generally correlates with higher scores.
        # Apply a +/- 0.01 micro-reward if a successful cross was made this turn.
        if active_player == 1 and (c_np1_r > c_p1_r or c_np1_b > c_p1_b): step_reward += 0.01
        elif active_player == 2 and (c_np2_r > c_p2_r or c_np2_b > c_p2_b): step_reward -= 0.01
        
        # The expected value becomes: Immediate Heuristic Reward + Discounted Future Value
        return step_reward + float(shared_V[next_state, next_idx])


class TDLambdaAgent(BaseAgent):
    """
    Implements Minimax-TD(lambda) with Replacing Eligibility Traces.
    (Thesis Equations 17-21).
    Instead of only updating the immediately preceding state, this agent 
    propagates the Temporal Difference error (delta) backwards through the 
    entire episodic trajectory, exponentially decaying by lambda.
    """
    
    def __init__(self, lambda_decay=0.9):
        self.lambda_decay = lambda_decay
        self.eligibility_traces = {}

    def reset_episode(self):
        """Clears the trajectory memory vector e(s) at the start of a new Markov episode."""
        self.eligibility_traces = {}

    def update_value(self, state, active_idx, v_target, alpha, shared_V):
        # Calculate the TD Error (delta): [Target - Current Estimate]
        delta = v_target - shared_V[state, active_idx]
        
        # Replacing Traces: Set current state trace exactly to 1.0 (Equation 20)
        self.eligibility_traces[(state, active_idx)] = 1.0
        
        # Apply the TD error (delta) backward through all historically visited states
        for (t_state, t_player) in list(self.eligibility_traces.keys()):
            # V(s) <- V(s) + alpha * delta * e(s)
            shared_V[t_state, t_player] += alpha * delta * self.eligibility_traces[(t_state, t_player)]
            
            # Decay the trace: e(s) <- e(s) * lambda
            self.eligibility_traces[(t_state, t_player)] *= self.lambda_decay
            
            # Computational Optimization: Prune traces once they become mathematically negligible
            if self.eligibility_traces[(t_state, t_player)] < 0.01:
                del self.eligibility_traces[(t_state, t_player)]


class BoltzmannAgent(BaseAgent):
    """
    Implements Softmax / Boltzmann Action Selection (Thesis Equation 22).
    Instead of rigidly taking the argmax (greedy), this agent samples actions 
    from a Gibbs distribution parameterized by Temperature (tau).
    This ensures all actions have a non-zero probability of being explored, 
    but heavily favors the mathematically superior actions.
    """
    
    def select_actions(self, payoff_matrix, white_actions, best_c_actions, param_val):
        """
        param_val represents Tau (Temperature). 
        High tau -> Uniform random exploration.
        Low tau -> Approaches pure argmax exploitation.
        """
        tau = param_val
        
        # --- PLAYER 1 (Maximizer) Probabilities ---
        # Evaluate P1's pure actions assuming P2 plays optimally against them
        p1_vals = np.min(payoff_matrix, axis=1)
        
        # Numerically Stable Softmax: Subtracting the maximum value prevents 
        # floating-point overflow during exponentiation (e^x).
        p1_exp = np.exp((p1_vals - np.max(p1_vals)) / tau) 
        p1_probs = p1_exp / np.sum(p1_exp)
        
        # --- PLAYER 2 (Minimizer) Probabilities ---
        p2_vals = np.max(payoff_matrix, axis=0)
        
        # Player 2 minimizes, so we invert the values (-p2_vals) before exponentiating
        p2_exp = np.exp((-p2_vals - np.max(-p2_vals)) / tau) 
        p2_probs = p2_exp / np.sum(p2_exp)
        
        # Stochastically sample the joint action profile based on the computed distributions
        a_w1 = np.random.choice(white_actions, p=p1_probs)
        a_w2 = np.random.choice(white_actions, p=p2_probs)
        
        return a_w1, a_w2