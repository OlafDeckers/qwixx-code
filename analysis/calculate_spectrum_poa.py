import os
import matplotlib.pyplot as plt
from analysis.evaluator import TournamentEngine

def calculate_spectrum_poa():
    agents = ['SOLO', 'SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']
    games_per_agent = 100000 
    results_welfare = {}

    print("\n" + "="*75)
    print(f" CALCULATING PRICE OF ANARCHY SPECTRUM ({games_per_agent} Games/Agent)")
    print("="*75)

    for agent in agents:
        print(f"Simulating {games_per_agent} matches for {agent} vs {agent}...")
        
        # DELEGATE TO THE UNIFIED ENGINE
        stats = TournamentEngine.run_nash_matchup(agent, agent, games_per_agent)
        results_welfare[agent] = stats['total_welfare'] / games_per_agent

    w_opt = results_welfare['SOLO']

    print("\n" + "="*75)
    print(" FINAL RESULTS: THE PRICE OF ANARCHY SPECTRUM")
    print("="*75)
    print(f"Optimal Social Welfare (W_opt): {w_opt:.2f} Total Points per game\n")
    print(f"{'Strategy Type':<25} | {'Social Welfare':<15} | {'PoA (W_opt / W_nash)'}")
    print("-" * 75)
    
    plot_labels, welfares_to_plot, poas_to_plot = [], [], []

    for agent in ['SCORE', 'HYBRID_5', 'HYBRID_10', 'HYBRID_25', 'HYBRID_50', 'WIN']:
        w_nash = results_welfare[agent]
        poa = w_opt / w_nash
        print(f"{agent:<25} | {w_nash:<15.2f} | {poa:.4f}")
        
        label_map = {'SCORE': 'Score', 'HYBRID_5': 'Hybrid 5', 'HYBRID_10': 'Hybrid 10', 'HYBRID_25': 'Hybrid 25', 'HYBRID_50': 'Hybrid 50', 'WIN': 'Win Prob'}
        plot_labels.append(label_map[agent]); welfares_to_plot.append(w_nash); poas_to_plot.append(poa)
        
    print("="*75)
    print("\nGenerating visual plot...")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_bar = '#4A90E2'
    ax1.set_xlabel('Strategy Objective', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Social Welfare (Combined Points)', color=color_bar, fontsize=12, fontweight='bold')
    ax1.bar(plot_labels, welfares_to_plot, color=color_bar, alpha=0.8, edgecolor='black')
    ax1.tick_params(axis='y', labelcolor=color_bar)
    ax1.axhline(y=w_opt, color='gray', linestyle='--', linewidth=2, label=f'Optimal Baseline ({w_opt:.2f} pts)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  
    color_line = '#D0021B'
    ax2.set_ylabel('Price of Anarchy (PoA)', color=color_line, fontsize=12, fontweight='bold')
    ax2.plot(plot_labels, poas_to_plot, color=color_line, marker='o', markersize=8, linewidth=3, label='PoA')
    ax2.tick_params(axis='y', labelcolor=color_line)
    ax2.set_ylim([0.9, max(poas_to_plot) + 0.15]) 

    plt.title('Price of Anarchy Spectrum in Mini-Qwixx', fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout() 

    os.makedirs('plots', exist_ok=True)
    plot_path = 'plots/price_of_anarchy_spectrum.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    calculate_spectrum_poa()