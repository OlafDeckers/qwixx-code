import numpy as np
import random

class BaseAgent:
    """Base Strategy class for RL Agents. Default behavior is standard Minimax-Q."""
    
    def reset_episode(self):
        """Called at the start of every new episode."""
        pass

    def get_future_value(self, is_term, s1, s2, shared_V, next_state, next_idx, env_info):
        """Calculates the bootstrapped future value for the payoff matrix."""
        if is_term:
            if s1 > s2: return 1.0
            elif s1 < s2: return -1.0
            return 0.0
        return float(shared_V[next_state, next_idx])

    def update_value(self, state, active_idx, v_target, alpha, shared_V):
        """Standard Temporal Difference (TD) update."""
        shared_V[state, active_idx] += alpha * (v_target - shared_V[state, active_idx])

    def select_actions(self, payoff_matrix, white_actions, best_c_actions, param_val):
        """Action selection. param_val represents epsilon for standard agents."""
        # Using exact logic from original scripts to preserve results
        return random.choice(white_actions), random.choice(white_actions)


class RewardShapingAgent(BaseAgent):
    """Implements Equation 15: Heuristic Reward Shaping."""
    
    def get_future_value(self, is_term, s1, s2, shared_V, next_state, next_idx, env_info):
        if is_term:
            return super().get_future_value(is_term, s1, s2, shared_V, next_state, next_idx, env_info)
        
        # Unpack env_info dictionary
        active_player, np1_p, p1_p, np2_p, p2_p, c_np1_r, c_p1_r, c_np1_b, c_p1_b, c_np2_r, c_p2_r, c_np2_b, c_p2_b = env_info.values()
        
        step_reward = 0.0
        if active_player == 1 and np1_p > p1_p: step_reward -= 0.05
        elif active_player == 2 and np2_p > p2_p: step_reward += 0.05 
        
        if active_player == 1 and (c_np1_r > c_p1_r or c_np1_b > c_p1_b): step_reward += 0.01
        elif active_player == 2 and (c_np2_r > c_p2_r or c_np2_b > c_p2_b): step_reward -= 0.01
        
        return step_reward + float(shared_V[next_state, next_idx])


class TDLambdaAgent(BaseAgent):
    """Implements Equation 17-21: Replacing Eligibility Traces."""
    
    def __init__(self, lambda_decay=0.9):
        self.lambda_decay = lambda_decay
        self.eligibility_traces = {}

    def reset_episode(self):
        self.eligibility_traces = {}

    def update_value(self, state, active_idx, v_target, alpha, shared_V):
        delta = v_target - shared_V[state, active_idx]
        self.eligibility_traces[(state, active_idx)] = 1.0
        
        # Apply delta backward through traces
        for (t_state, t_player) in list(self.eligibility_traces.keys()):
            shared_V[t_state, t_player] += alpha * delta * self.eligibility_traces[(t_state, t_player)]
            self.eligibility_traces[(t_state, t_player)] *= self.lambda_decay
            if self.eligibility_traces[(t_state, t_player)] < 0.01:
                del self.eligibility_traces[(t_state, t_player)]


class BoltzmannAgent(BaseAgent):
    """Implements Equation 22: Softmax Action Selection."""
    
    def select_actions(self, payoff_matrix, white_actions, best_c_actions, param_val):
        """param_val represents Tau (Temperature) for Boltzmann."""
        tau = param_val
        
        p1_vals = np.min(payoff_matrix, axis=1)
        p1_exp = np.exp((p1_vals - np.max(p1_vals)) / tau) 
        p1_probs = p1_exp / np.sum(p1_exp)
        
        p2_vals = np.max(payoff_matrix, axis=0)
        p2_exp = np.exp((-p2_vals - np.max(-p2_vals)) / tau) 
        p2_probs = p2_exp / np.sum(p2_exp)
        
        a_w1 = np.random.choice(white_actions, p=p1_probs)
        a_w2 = np.random.choice(white_actions, p=p2_probs)
        
        return a_w1, a_w2