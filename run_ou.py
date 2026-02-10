import numpy as np

def run_ou(sigma):

    org_u_series = np.linspace(0.01, 2.0, 50)  # change upper end as needed

    dt = 0.001 # time step for Euler-Maruyama
    n_samples = 100 # number of samples per bifurcation parameter value for computing std
    sample_gap = int(10.0//dt) # steps between samples to ensure independence
    
    final_u_series = []
    var_series = []
    
    for u in org_u_series:
        x = 0.0 # start from the equikibrium

        # Equilibrate (transient decay); we do not practically need it, but we do this to be consistent with other models of dynamics
        equil_steps = int(10//dt) # lengthen for stronger separation if needed
        for _ in range(equil_steps):
            noise = sigma * np.sqrt(dt) * np.random.randn()
            x += -x/u * dt + noise

        # Sample n_samples points, well-separated in time
        samples = []
        for _ in range(n_samples):
            for _ in range(sample_gap):
                noise = sigma * np.sqrt(dt) * np.random.randn()
                x += -x/u * dt + noise
            samples.append(x)

        var_series.append(np.var(samples)) # Compute sample standard deviation and add to the existing array
        final_u_series.append(u)

    # dynamics simulations finished

    final_u_series = np.array(final_u_series)
    var_series = np.array(var_series)
    return final_u_series, var_series