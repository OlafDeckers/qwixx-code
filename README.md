# Qwixx AI: Exact Equilibria and Reinforcement Learning in Adversarial Dice Games

This repository contains the complete codebase for the Bachelor's thesis evaluating exact Nash equilibria and Reinforcement Learning (RL) architectures in a reduced variant of the stochastic dice game Qwixx.

## Overview
Qwixx is traditionally played as a parallel solitaire optimization problem. This project formalizes the game as a **zero-sum stochastic Markov Game**. Because the full game contains over 73 million states, we introduce **Mini-Qwixx** (2 rows, 3-sided dice), reducing the state space to 565,656 states. This allows for the exact computation of Nash equilibria via single-pass backward induction, establishing a ground truth to evaluate model-free RL architectures.

## Repository Structure
* `core/`: Core game logic, state encoding, and environment constants.
* `solvers/`: Dynamic Programming algorithms (Backward Induction) to compute exact state values and Nash mixed strategies.
* `rl_models/`: Reinforcement Learning agents (Standard, Reward Shaping, TD-$\lambda$, Boltzmann) and the Hogwild multiprocessing training loop.
* `analysis/`: Scripts for running round-robin tournaments, evaluating the Price of Anarchy, and generating heatmaps/plots.
* `tests/`: Validation scripts to ensure algorithmic correctness and strategic integrity.
* `data/`: Stored value matrices (`.npy`) and RL checkpoints.
* `plots/`: Generated visualization assets.

## Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

Usage GuideThe project is modular. Scripts should be run from the root directory using the -m flag.
1. Compute Exact Game Values to solve the DAG and generate the baseline Dynamic Programming values for all objective functions: Bash python -m solvers.unified_backward_induction
2. Run Strategy Tournaments to simulate the 100,000-game round-robin tournament between DP agents: Bash python -m analysis.simulate_round_robin
3. Calculate Price of Anarchy (PoA) to evaluate the systemic cost of adversarial play and compute Social Welfare: Bash python -m analysis.calculate_spectrum_poa
4. Train Reinforcement Learning Agents to train the model-free RL architectures over 20 million episodes: Bash python -m rl_models.train_unified
5. Generate Visualizations to plot the RL convergence or generate tournament heatmaps: Bash python -m analysis.plot_model_comparison

## Key Findings
1. DAG Topology defeats Sparse Rewards: A standard 1-step TD(0) agent successfully rediscovers optimal zero-sum policies without complex credit assignment due to the environment's strictly forward-moving 
2. structure.The Danger of Shaping & Traces: Heuristic reward shaping introduces strategic bias against defensive sacrifices, while TD-$\lambda$ suffers from stochastic variance bleed.
3. Price of Anarchy (1.30): Rational, self-interested adversarial play reduces overall social welfare by approximately 23%.