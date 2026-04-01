import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_20m_vtable():
    # Make sure the plots directory exists OUTSIDE the data folder
    os.makedirs('plots', exist_ok=True)
    
    print("="*60)
    print(" ANALYZING 20 MILLION EPISODE V-TABLE")
    print("="*60)
    try:
        # Load the final 20M V-Table snapshot
        file_path = 'data/V_rl_hybrid_20M.npy'
            
        print(f"Loading massive 20M Episode V-Table from {file_path}...")
        V_learned = np.load(file_path)
        
        p1_values = V_learned[:, 0]
        p2_values = V_learned[:, 1]
        
        # --- PLOT 1: THE HISTOGRAM ---
        print("Generating Value Distribution Histogram...")
        plt.figure(figsize=(8, 6))
        plt.hist(p1_values, bins=100, color='#4A90E2', alpha=0.7, label='Player 1')
        plt.hist(p2_values, bins=100, color='#D0021B', alpha=0.5, label='Player 2')
        plt.title("Distribution of Expected Values (20M Iterations)", fontsize=14, fontweight='bold')
        plt.xlabel("Expected Value (Points + Win Bonus)", fontsize=12, fontweight='bold')
        plt.ylabel("Frequency (Log Scale)", fontsize=12, fontweight='bold')
        plt.yscale('log') 
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        hist_save_path = 'plots/v_table_histogram_20M.png'
        plt.savefig(hist_save_path, dpi=300)
        plt.close()
        print(f"Success! Histogram saved to {hist_save_path}")

        # --- PLOT 2: THE SCATTER PLOT ---
        print("Generating State Value Scatter Plot (10,000 random states)...")
        # Sample 10k states so we don't crash matplotlib
        sample_indices = np.random.choice(len(p1_values), size=10000, replace=False)
        p1_sample = p1_values[sample_indices]
        p2_sample = p2_values[sample_indices]

        plt.figure(figsize=(8, 6))
        plt.scatter(p1_sample, p2_sample, alpha=0.3, color='purple', s=4)
        plt.title("P1 vs P2 Expected Values (20M Iterations - 10k Sample)", fontsize=14, fontweight='bold')
        plt.xlabel("Player 1 Expected Value", fontsize=12, fontweight='bold')
        plt.ylabel("Player 2 Expected Value", fontsize=12, fontweight='bold')
        
        min_val = min(np.min(p1_sample), np.min(p2_sample))
        max_val = max(np.max(p1_sample), np.max(p2_sample))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Symmetry (y=x)')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        scatter_save_path = 'plots/v_table_scatter_20M.png'
        plt.savefig(scatter_save_path, dpi=300)
        plt.close()
        print(f"Success! Scatter plot saved to {scatter_save_path}\n")

    except FileNotFoundError:
        print("Error: Could not find the data/V_rl_hybrid_20M.npy file. Make sure the path is correct.\n")

    print("="*60)
    print(" ANALYSIS COMPLETE! ")
    print("="*60)

if __name__ == '__main__':
    analyze_20m_vtable()