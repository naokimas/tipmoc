# Determine the bifurcation point of the mutualistic-interaction model.
# The code reads LSF.csv, the network with N=100 nodes used in this study.

import numpy as np
import pandas as pd

def deterministic_interaction(x, D, dt, edges):
    """
    One step of mutualistic-interaction dynamics without noise.
    """
    # System parameters
    B, C, Di, E, H, K = 0.1, 1, 5, 0.9, 0.1, 5
    N = len(x)
    stress = -2.9
    
    # Deterministic drift term (removed sigma * sqrt(dt) * randn)
    dx = (B + stress + x * (1 - x / K) * (x / C - 1)) * dt

    # Extract i and j indices from edge list (0-indexed)
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1

    # Compute asymmetric interaction terms
    # D * xi * xj / (Di + E*xi + H*xj)
    interaction_term_i = D * x[i_indices] * x[j_indices] / (Di + E * x[i_indices] + H * x[j_indices]) * dt
    interaction_term_j = D * x[i_indices] * x[j_indices] / (Di + E * x[j_indices] + H * x[i_indices]) * dt

    contribution = np.zeros_like(x)
    # Accumulate interactions that the nodes receive in dt
    np.add.at(contribution, i_indices, interaction_term_i)
    np.add.at(contribution, j_indices, interaction_term_j)

    # Update state
    nextx = x + dx + contribution
    nextx[nextx <= 0] = 0
    return nextx

def check_collapse(D, edges, N):
    """
    Runs the simulation for a fixed D and returns True if any node 
    falls below the threshold of 0.1.
    """
    dt = 0.001
    equil_steps = int(10 // dt)  # Transient period
    x = np.full(N, 5.0)          # Initial condition
    
    for _ in range(equil_steps):
        x = deterministic_interaction(x, D, dt, edges)
        
    # Check if any node is in the lower state (collapsed)
    return np.any(x < 0.1)

def find_critical_D(csv_path='LFR.csv', tolerance=1e-4):

    # Load network
    df = pd.read_csv(csv_path)
    edge_list = [(int(u[1:]), int(v[1:])) for u, v in zip(df['from'], df['to'])]
    edges = np.array(edge_list)
    N = edges.max()

    # Set bisection parameters
    D_max = 1.0 # coexistence state is stable (no collapse)
    D_min = 0.0 # likely collapsed
    
    # Initial check to ensure bisection is valid
    if not check_collapse(D_min, edges, N):
        print("Even at D=0, no collapse occurs. Transition point is outside [0, 1].")
        return None
    if check_collapse(D_max, edges, N):
        print("At D=1, the system is already collapsed. Increase D_max.")
        return None

    print(f"Starting bisection to find D_c in [{D_min}, {D_max}]...")
    
    # Bisection Loop
    iteration = 0
    while (D_max - D_min) > tolerance:
        mid_D = (D_max + D_min) / 2.0
        collapsed = check_collapse(mid_D, edges, N)
        
        if collapsed:
            # If collapsed, the critical D is higher than current mid_D
            D_min = mid_D
        else:
            # If stable, the critical D is lower than current mid_D
            D_max = mid_D
            
        iteration += 1
        print(f"Iter {iteration:02d}: D_range = [{D_min:.6f}, {D_max:.6f}]")

    critical_D = (D_max + D_min) / 2.0
    print("-" * 30)
    print(f"Approximate Critical D: {critical_D:.6f}")
    return critical_D

if __name__ == "__main__":
    # Ensure LFR.csv is in the same directory
    try:
        find_critical_D('LFR.csv')
    except FileNotFoundError:
        print("Error: LFR.csv not found.")