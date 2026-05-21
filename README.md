Qwixx AI: Minimax Backward Induction & Reinforcement Learning

This repository contains the complete implementation for solving and evaluating Mini-Qwixx, a reduced-state variant of the dice game Qwixx, using both exact game-theoretic methods and reinforcement learning.

The framework models the game as a finite Directed Acyclic Graph (DAG) with 565,656 unique states, computes exact Subgame Perfect Nash Equilibria through parallelized Backward Induction, and compares these optimal strategies against model-free RL agents under multiple competitive objectives.

✨ Key Features
Exact Dynamic Programming Solver
Computes equilibrium strategies for:
Win Probability
Score Difference
Hybrid utility functions
Solo score maximization
Branchless Environment Architecture
All transitions and rule evaluations are precomputed into compact 32-bit lookup arrays.
Environment steps reduce to branchless O(1) memory accesses.
Numba-Accelerated Compilation
Critical solvers and encoders are JIT-compiled with Numba for high-throughput multi-core execution.
Large-Scale Tournament Engine
Shared-memory Monte Carlo evaluator supporting millions of rollouts.
Tracks:
Win rates
Score margins
Penalties
Marks skipped
Behavioral efficiency metrics
Behavioral Game-Theoretic Analysis
Measures the "Price of Competition"
Evaluates:
Offensive rushing
Defensive stalling
Risk-sensitive equilibrium behavior
⚙️ Installation

Requires Python 3.9+

pip install -r requirements.txt
📂 Project Structure
core/ — Environment & State Representation
environment.py

Defines the complete Markov Decision Process (MDP):

White Phase and Color Phase logic
Legal transitions
Triangular score calculations
Terminal state detection
state_encoder.py

Implements constant-time bijective state encoding:

Multi-dimensional game state → 32-bit integer index
Optimized for dense array access
constants.py

Centralized configuration:

Action spaces
Probability tables
Row limits
Scoring constants
solvers/ — Exact Dynamic Programming
unified_backward_induction.py

Main exact solver:

Traverses the topological DAG backward
Computes equilibrium value functions and policies
matrix_math.py

Custom analytical mixed-strategy solver:

Solves simultaneous 3×3 zero-sum matrices
Avoids heavy LP dependencies
state_space_graph.py

Builds:

Reachability graph
Topological ordering
State depth hierarchy
rl_models/ — Reinforcement Learning Baselines
agents.py

Implements:

Littman Minimax-Q Learning
Competitive tabular RL agents
train_unified.py

Training pipeline supporting:

ε-greedy exploration
Boltzmann exploration
TD(λ)
Reward shaping
Large-scale self-play
analysis/ — Evaluation & Visualization
evaluator.py

High-performance tournament engine:

Parallel rollouts
Cross-play evaluation
Behavioral metric tracking
simulate_round_robin.py

Generates:

Win-rate matrices
Margin heatmaps
First-player advantage analysis
plot_behavioral_metrics.py

Analyzes:

Marks skipped
Penalties taken
Offensive vs. defensive strategic behavior
calculate_spectrum_poc.py

Computes:

Social welfare
Price of Competition (PoC)
Objective tradeoff curves
variance_horizon.py

Visualizes:

Score variance
Horizon effects
Win probability interactions
plot_model_comparison.py

Compares RL agents against exact DP baselines:

Convergence
MSE
Learning efficiency
🚀 Execution Pipeline

Run the following commands sequentially from the repository root.

1. Generate the State-Space DAG
python solvers/state_space_graph.py
2. Compute Exact Equilibrium Policies
python solvers/unified_backward_induction.py
3. Train RL Baselines
python rl_models/train_unified.py
4. Run Evaluation & Generate Figures
python analysis/simulate_round_robin.py
python analysis/plot_behavioral_metrics.py
python analysis/calculate_spectrum_poc.py
python analysis/variance_horizon.py
python analysis/plot_model_comparison.py
📊 Research Focus

This project studies the interaction between:

Exact game-theoretic optimization
Competitive reinforcement learning
Risk-sensitive objectives
Strategic inefficiency in adversarial settings

Key empirical findings include:

Quantification of the Price of Competition
Behavioral transitions between aggressive and conservative play
Convergence gaps between exact DP and RL approximations
📈 Behavioral Metrics Summary

Fig. behavioral_metrics shows relatively modest behavioral differences across objective functions rather than a strict offensive-versus-defensive dichotomy.

The Solo baseline records the fewest skipped marks at approximately 2.30 per player and about 1.05 penalties per player.
The Score (0 Bonus) objective increases skipped marks to roughly 2.51 while reducing penalties to approximately 1.00.
The Hybrid objectives gradually increase skipped marks from about 2.53 to 2.60, with penalties remaining nearly constant around 1.00.
The Win Probability objective produces the highest penalty rate at roughly 1.17 penalties per player while averaging about 2.53 skipped marks.

Overall, the results indicate that equilibrium objectives induce subtle strategic shifts in pacing and risk management rather than extreme offensive rushing or defensive stalling behavior.