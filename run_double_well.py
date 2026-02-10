import numpy as np

def run_double_well(sigma, poisson_u=False, colored_noise=False):
# if poisson_u==1, then u values are not equidistantly but exponentially distributed.

    r1, r2, r3 = 1, 3, 5
    num_u = 50
    if poisson_u:
        org_u_series = np.sort(np.random.uniform(low=0.0, high=3.079, size=num_u))
    else:
        org_u_series = np.linspace(0.0, 3.079, num_u)  # change upper end as needed

    dt = 0.001 # time step for Euler-Maruyama
    n_samples = 100 # number of samples per bifurcation parameter value for computing sample var
    sample_gap = int(1.0//dt) # steps between samples to ensure independence
    if colored_noise:
        tau_col = 1.0
    
    def f(x, u):
        return - (x - r1)*(x - r2)*(x - r3) + u

    final_u_series = []
    var_series = []
    terminate = False
    
    for u in org_u_series:
        if terminate:
            break
        x = r1 # initialization

        # Equilibrate (transient decay)
        equil_steps = int(10//dt)  # lengthen for stronger separation if needed
        if colored_noise:
            eta = 0.0 # colored noise auto-correlation timescale
        for _ in range(equil_steps):
            if colored_noise:
                # Evolve an OU process for noise
                eta += (-eta/tau_col)*dt + sigma * np.sqrt(2*dt)*np.random.randn()
                x += (f(x, u) + eta) * dt
            else:
                x += f(x, u) * dt + sigma * np.sqrt(dt) * np.random.randn()
    
        # Sample n_samples points, well-separated in time
        samples = []
        crossed_r2 = False
        for _ in range(n_samples):
            for _ in range(sample_gap):
                if colored_noise:
                    # Evolve an OU process for noise
                    eta += (-eta/tau_col)*dt + sigma * np.sqrt(2*dt)*np.random.randn()
                    x += (f(x, u) + eta) * dt
                else:
                    noise = sigma * np.sqrt(dt) * np.random.randn()
                    x += f(x, u) * dt + sigma * np.sqrt(dt) * np.random.randn()
                if x > r2:
                    crossed_r2 = True
                    break
            if crossed_r2:
                break
            samples.append(x) # If x > r2 at any point, do not append x and terminate all future simulations

        if crossed_r2:
            terminate = True
            print(f"Terminating all further u at u = {u:.3f}, because x crossed {r2}.")
            break
        else:
            var_series.append(np.var(samples)) # Compute sample standard deviation and add to the existing array
            final_u_series.append(u)

    # dynamics simulations finished


    if poisson_u:
        equi_u_series = np.linspace(0.0, 3.079, num_u)
        final_u_series = equi_u_series[:len(final_u_series)]
    else:
        final_u_series = np.array(final_u_series)

    var_series = np.array(var_series)
    return final_u_series, var_series