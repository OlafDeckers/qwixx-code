```markdown
# Qwixx AI: Minimax Backward Induction & Reinforcement Learning 

This repository contains the full source code for the empirical evaluation and mathematical solving of **Mini-Qwixx**, a reduced-state variant of the popular dice game Qwixx. 

The framework models the game as a finite Directed Acyclic Graph (DAG) containing 565,656 distinct states. It calculates exact Subgame Perfect Nash Equilibria via parallelized Backward Induction and evaluates the "Price of Competition" and behavioral mechanics against baseline Reinforcement Learning (RL) agents.

## ✨ Key Features
* **Branchless Architecture:** Environment transitions and rule evaluations are precomputed into strictly typed 32-bit static arrays, reducing environment steps to branchless O(1) memory lookups.
* **Numba C-Compilation:** Core solvers and state encoders are Just-In-Time (JIT) compiled to bypass the Python Global Interpreter Lock (GIL), enabling massive multi-core throughput.
* **Unified Tournament Engine:** A custom lock-free, shared-memory evaluator that pits Exact DP models against RL agents over millions of Monte Carlo rollouts to track precise behavioral metrics (margins, penalties, and skipped marks).

---

## ⚙️ Installation & Setup

Ensure you have Python 3.9+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt

```

---

## 📂 Project Structure & Core Modules

### 1. `core/` (The Environment & MDP Formulation)

* **`environment.py`**: The central Markov Decision Process (MDP). Enforces legal transitions, resolves the simultaneous White Phase and sequential Color Phase, calculates exact scoring via triangular numbers, and identifies terminal states.
* **`state_encoder.py`**: Contains the O(1) bijective bitwise encoding/decoding functions that map the multi-dimensional game tuple to a singular 32-bit integer for array indexing.
* **`constants.py`**: The single source of truth for the action spaces (A_w, A_c), row limits, and probability constants.

### 2. `solvers/` (Exact Dynamic Programming)

* **`unified_backward_induction.py`**: The core exact solver. Traverses the topological DAG backwards from terminal states to calculate exact expected values (Win Probability, Score Difference, Hybrid models, and the non-adversarial Solo baseline).
* **`matrix_math.py`**: A custom, compiled analytical solver used to extract mixed-strategy Nash equilibria from the 3x3 zero-sum simultaneous White Phase payoff matrices without the overhead of heavy LP libraries.
* **`state_space_graph.py`**: Computes and stores the topological depth sorting required for single-pass backward induction.

### 3. `rl_models/` (Model-Free Baselines)

* **`agents.py`**: Implements Littman's Minimax-Q learning algorithm.
* **`train_unified.py`**: Executes training loops over millions of episodes utilizing Exploring Starts, and contrasts different mechanisms such as Standard epsilon-greedy, Boltzmann exploration, TD(lambda), and Reward Shaping.

### 4. `analysis/` (Simulation & Visualization)

* **`evaluator.py`**: The central simulation engine (`TournamentEngine`). Performs rapid parallelized rollouts for cross-play tournaments while explicitly tracking complex behavioral metrics (points, margins, skips, penalties).
* **`simulate_round_robin.py`**: Executes the 7x7 matrix cross-play tournament. Generates heatmaps visualizing Overall Win Rates, First-Mover Advantage, and Expected Margins.
* **`plot_behavioral_metrics.py`**: Analyzes terminal states under self-play to generate a dual-axis chart comparing **Offensive Rushing** (Marks Skipped) against **Defensive Stalling** (Penalties Taken).
* **`calculate_spectrum_poc.py`**: Calculates total expected Social Welfare and visually maps the Price of Competition (PoC) across the different objective functions.
* **`variance_horizon.py`**: Scans the exact DP table to generate a heatmap demonstrating how Win Probability interacts with the Score Margin as game depth increases.
* **`plot_model_comparison.py`**: Compares the learning efficiency and Mean Squared Error (MSE) convergence of the various RL architectures against the exact mathematical baseline.

---

## 🚀 Execution Pipeline

The framework is designed to be executed sequentially. Run the following commands from the root directory to replicate the full thesis methodology:

**1. Generate the topological state space DAG:**
python -m solvers.state_space_graph

**2. Solve the Exact Nash Equilibria (Populates the `data/` directory):**
python -m solvers.unified_backward_induction

**3. Train the Reinforcement Learning baselines:**
python -m rl_models.train_unified

**4. Generate Empirical Analysis & Plots:**
python -m analysis.simulate_round_robin
python -m analysis.plot_behavioral_metrics
python -m analysis.calculate_spectrum_poc
python -m analysis.variance_horizon
python -m analysis.plot_model_comparison
