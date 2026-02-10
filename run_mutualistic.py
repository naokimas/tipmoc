# import networkx as nx
import numpy as np
import math
# import time
import multiprocessing
import pandas as pd

# one step of mutualistic interaction dynamics
def mutualistic_interaction(x, sigma, D, dt, edges):
    # parameters in the mutualistic interaction dynamics
    B, C, Di, E, H, K = 0.1, 1, 5, 0.9, 0.1, 5
    N = len(x)
    stress = -2.9
    dx = (B + stress + x * (1 - x / K) * (x / C - 1)) * dt + sigma * np.sqrt(dt) * np.random.randn(N)

    # Extract i and j indices from edge list
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1

    # Compute asymmetric interaction terms
    interaction_term_i = D * x[i_indices] * x[j_indices] / (Di + E * x[i_indices] + H * x[j_indices]) * dt
    interaction_term_j = D * x[i_indices] * x[j_indices] / (Di + E * x[j_indices] + H * x[i_indices]) * dt

    contribution = np.zeros_like(x)
    # Accumulate interactions into dx for both i and j nodes
    np.add.at(contribution, i_indices, interaction_term_i)  # Update dx[i]
    np.add.at(contribution, j_indices, interaction_term_j)  # Update dx[j]
    # contribution[i_indices] += interaction_term_i
    # contribution[j_indices] += interaction_term_j

    # Update x
    dx += contribution
    nextx = x + dx
    nextx[nextx <= 0] = 0
    return nextx

def run_mutualistic(sigma):

# From Shilong, and this is for the LFR network: the initial values are u = 0 and D = 1. The number of time steps for is I = 20000. The number of times we change the control parameter is k = 200. If we change u, then du = -0.1. If we change D, then dD = -0.01. 

    # Read network
    df = pd.read_csv('LFR.csv') # N = 100 nodes

    # Parse edge list: strip 'V' and convert to int
    edge_list = [(int(u[1:]), int(v[1:])) for u, v in zip(df['from'], df['to'])] # Removes the "V" and converts to integer.

    # Convert to numpy array
    edges = np.array(edge_list)

    # Optional: get number of nodes if needed
    N = edges.max()    # Assumes nodes are labelled consecutively from 1

    # Parameters
    dt = 0.001
    n_samples = 5 # 100 # 4 # 100
    sample_gap = int(1.0//dt) # simulate 10 steps between samples

    # 'D', the coupling strength, is the bifurcation parameter
    num_u = 50 # We scan 'k' values of 'D'
    org_D_series = np.linspace(1.0, 0.0, num_u) # Initially D=1.0
    final_D_series = []
    var_series = []
    terminate = False
    equil_steps = int(10//dt) # = 10 TU in Masuda et al., Nat Commun (2024)

    for D in org_D_series:
        if terminate:
            break
        # Initialization
        x = np.full(N, 5.0) # each x[i] = 5.0

        # Equilibrate (burn-in)
        for _ in range(equil_steps):
            x = mutualistic_interaction(x, sigma, D, dt, edges)
        # Sample equilibrium x series
        samples = []
        crossed = False
        for _ in range(n_samples):
            for _ in range(sample_gap):
                x = mutualistic_interaction(x, sigma, D, dt, edges)
                if np.min(x) < 0.1:  # collapsed population
                    crossed = True
                    break
            if crossed:
                break
            samples.append(x)
        if crossed:
            terminate = True
            print(f"Terminating further D; collapse of the species population detected at D = {D:.3f}")
            break
        else:
            samples_array = np.array(samples)        # shape (n_samples, N)
            # Compute sample variance for each node (axis=0, ddof=1 for unbiased sample variance)
            variances = np.var(samples_array, axis=0, ddof=1)
            avg_variance = np.mean(variances)        # average over nodes
            var_series.append(float(avg_variance))
#            var_series.append(float(variances[0]))
            final_D_series.append(-float(D)) # sign flipped for preparing for the Vuong test
    return final_D_series, var_series