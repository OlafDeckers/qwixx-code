import os
import matplotlib.pyplot as plt
from analysis.evaluator import TournamentEngine

def calculate_spectrum_poa():
    """
    Calculates the Price of Anarchy (PoA) spectrum as described in the thesis.
    The PoA measures the degradation of system efficiency (total points scored) 
    caused by agents playing competitively (zero-sum) rather than cooperatively.
    """
    
    # We evaluate 7 distinct policies derived from Backward Induction
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    
    # We use Monte Carlo simulation (100,000 games) to find the Expected Value 
    # of the Social Welfare for each policy under self-play.
    games_per_agent = 100000 
    results_welfare = {}

    print("\n" + "="*75)
    print(f" CALCULATING PRICE OF ANARCHY SPECTRUM ({games_per_agent} Games/Agent)")
    print("="*75)

    # Step 1: Calculate the Expected Social Welfare (W) for each strategy
    for agent in agents:
        print(f"Simulating {games_per_agent} matches for {agent} vs {agent}...")
        
        # DELEGATE TO THE UNIFIED ENGINE:
        # This runs the stochastic Markov game, drawing actions from the exact 
        # Nash equilibrium probabilities calculated during backward induction.
        stats = TournamentEngine.run_nash_matchup(agent, agent, games_per_agent)
        
        # Thesis Reference: Equation 14 -> W(s) = Score_1(s) + Score_2(s)
        # We average the sum of both players' points over all simulated games 
        # to find the Expected Social Welfare.
        results_welfare[agent] = stats['total_welfare'] / games_per_agent

    # Step 2: Establish the Social Optimum (W_solo)
    # The SOLO agent ignores zero-sum dynamics and strictly maximizes its own score.
    # Therefore, self-play between two SOLO agents yields the maximum possible total points.
    w_opt = results_welfare['SOLO']

    print("\n" + "="*75)
    print(" FINAL RESULTS: THE PRICE OF ANARCHY SPECTRUM")
    print("="*75)
    print(f"Optimal Social Welfare (W_opt): {w_opt:.2f} Total Points per game\n")
    print(f"{'Strategy Type':<25} | {'Social Welfare':<15} | {'PoA (W_opt / W_nash)'}")
    print("-" * 75)
    
    plot_labels, welfares_to_plot, poas_to_plot = [], [], []

    # Step 3: Calculate the Price of Anarchy for the adversarial/zero-sum agents
    for agent in ['SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']:
        w_nash = results_welfare[agent]
        
        # Thesis Reference: Equation 15 -> PoA = W_solo / W_nash
        # A PoA of 1.0 means perfectly efficient. > 1.0 indicates value destroyed by defensive play.
        poa = w_opt / w_nash
        
        print(f"{agent:<25} | {w_nash:<15.2f} | {poa:.4f}")
        
        # Map internal variable names to clean labels for the Matplotlib graph
        label_map = {'SCORE': 'Score', 'HYBRID_5': 'Hybrid 5', 'HYBRID_10': 'Hybrid 10', 'HYBRID_25': 'Hybrid 25', 'HYBRID_50': 'Hybrid 50', 'WIN': 'Win Prob'}
        plot_labels.append(label_map[agent]); welfares_to_plot.append(w_nash); poas_to_plot.append(poa)
        
    print("="*75)
    print("\nGenerating visual plot...")
    
    # ========================================================
    # MATPLOTLIB VISUALIZATION
    # Generates a dual-axis chart: Bar chart for Welfare, Line chart for PoA.
    # ========================================================
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Y-Axis: Social Welfare (Bar Chart)
    color_bar = '#4A90E2'
    ax1.set_xlabel('Strategy Objective', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Social Welfare (Combined Points)', color=color_bar, fontsize=12, fontweight='bold')
    ax1.bar(plot_labels, welfares_to_plot, color=color_bar, alpha=0.8, edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    
    # Draw the dashed horizontal line representing the W_solo theoretical maximum
    ax1.axhline(y=w_opt, color='gray', linestyle='--', linewidth=2, label=f'Optimal Baseline ({w_opt:.2f} pts)')
    ax1.legend(loc='upper left')

    # Right Y-Axis: Price of Anarchy Ratio (Line Plot)
    ax2 = ax1.twinx()  
    color_line = '#D0021B'
    ax2.set_ylabel('Price of Anarchy (PoA)', color=color_line, fontsize=12, fontweight='bold')
    ax2.plot(plot_labels, poas_to_plot, color=color_line, marker='o', markersize=8, linewidth=3, label='PoA')
    ax2.tick_params(axis='y', labelcolor=color_line)
    
    # Scale the PoA axis slightly above the highest value for clean visual padding
    ax2.set_ylim([0.9, max(poas_to_plot) + 0.15]) 

    # Plot styling and saving
    plt.title('Price of Anarchy Spectrum in Mini-Qwixx', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout() 

    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/price_of_anarchy_spectrum.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")
    
    # Display the final image to the user
    plt.show()

if __name__ == '__main__':
    calculate_spectrum_poa()