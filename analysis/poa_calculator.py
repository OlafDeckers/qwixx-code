import numpy as np

def calculate_exact_poa():
    print("Loading Exact DP Arrays...")
    try:
        V_solo = np.load('data/V_solo.npy')
        V_nash = np.load('data/V_nash.npy')
    except FileNotFoundError:
        print("Error: Could not find the .npy files.")
        return

    # 1. Extract Exact Optimal Social Welfare (Solo Play)
    solo_p1 = V_solo[0, 0, 0]
    solo_p2 = V_solo[0, 0, 1]
    w_opt = solo_p1 + solo_p2

    # 2. Extract Exact Nash Social Welfare (Zero-Sum Play)
    # Because of our update, V_nash now tracks the exact scores just like V_solo!
    nash_p1 = V_nash[0, 0, 0]
    nash_p2 = V_nash[0, 0, 1]
    w_nash = nash_p1 + nash_p2

    # 3. Calculate PoA
    sacrifice = w_opt - w_nash
    poa_ratio = w_opt / w_nash

    print("\n" + "="*65)
    print("             THE EXACT PRICE OF ANARCHY (PoA)")
    print("="*65)
    print(f"[SOLO POLICY] - Cooperative/Greedy Maximums")
    print(f"  P1 Expected Score: {solo_p1:.4f}")
    print(f"  P2 Expected Score: {solo_p2:.4f}")
    print(f"  Optimal Social Welfare (W_opt):  {w_opt:.4f} Total Points\n")

    print(f"[NASH POLICY] - Competitive/Defensive Equilibrium")
    print(f"  P1 Expected Score: {nash_p1:.4f}")
    print(f"  P2 Expected Score: {nash_p2:.4f}")
    print(f"  Nash Social Welfare (W_nash):    {w_nash:.4f} Total Points")
    
    print("\n" + "-"*65)
    print(f"The 'Sacrifice' (Lost Efficiency): {sacrifice:.4f} points destroyed")
    print(f"The Price of Anarchy Ratio:        {poa_ratio:.5f}")
    print("-" * 65)
    
    print("\nCONCLUSION FOR THESIS:")
    print(f"By playing optimally to win (Nash), the players destroy {sacrifice:.2f} points")
    print(f"of potential score. The game is {poa_ratio:.3f}x more 'efficient' when")
    print("players ignore each other, proving the severe cost of defensive play.")
    print("="*65)

if __name__ == '__main__':
    calculate_exact_poa()