import numpy as np

def run_over_harvesting(K, sigma, mult_noise=False, poisson_u=False):

    r = 1
    h = 1
    dt = 0.001 # time step for Euler-Maruyama
    n_samples = 100 # number of samples per bifurcation parameter value for computing sample var
    sample_gap = int(1.0//dt) # steps between samples to ensure independence
    num_u = 50

    if K>5:
        c_max = 2.604 # 2.6771 # 2.604
        if poisson_u:
            org_c_series = np.sort(np.random.uniform(low=1, high=c_max, size=num_u))
        else:
            org_c_series = np.linspace(1, c_max, num_u)  # Cannot be reduced down to 0.02 (causing error)

    else:
        if poisson_u:
            org_c_series = np.sort(np.random.uniform(low=0.05, high=1.5, size=num_u))
        else:
            org_c_series = np.linspace(0.05, 1.5, num_u)        


    def f(x, c):
        if x<0:
            return 0.0
        elif x < 10:
            return r * x * (1-x/K) - c * x**2 / (x**2 + h**2)
        else:
            return r * x * (1-x/K) - c / (1 + (h/x)**2)

    final_c_series = []
    var_series = []
    terminate = False
    
    for c in org_c_series:
        if terminate:
            break
        x = K # initial condition

        # Equilibrate (transient decay)
        equil_steps = int(10//dt) # lengthen for stronger separation if needed
        for _ in range(equil_steps):
            dx = f(x, c) * dt
            if mult_noise:
                dx += sigma * x * np.sqrt(dt) * np.random.randn()
            else:
                dx += sigma * np.sqrt(dt) * np.random.randn()
            x += dx

        # Sample 100 points, well-separated in time
        samples = []
        crossed = False
        for _ in range(n_samples):
            for _ in range(sample_gap):
                dx = f(x, c) * dt
                if mult_noise:
                    dx += sigma * x * np.sqrt(dt) * np.random.randn()
                else:
                    dx += sigma * np.sqrt(dt) * np.random.randn()
                x += dx
                if x < 0:
                    crossed = True
                    break
            if crossed:
                break
            samples.append(x) # If x < 0 at any point, do not append x and terminate all future simulations

        if crossed:
            terminate = True
            print(f"Terminating all further c at c = {c:.3f}, because x crossed {-0.01}.")
            break
        else:
#            print(samples)
            var_series.append(np.var(samples)) # Compute sample standard deviation and add to the existing array
            final_c_series.append(c)

    # dynamics simulations finished

    if poisson_u:
        equi_c_series = np.linspace(1, c, len(equi_c_series)) # this is probably one off, so should be remedied if I use the poisson_u option.
        final_c_series = equi_c_series[:len(final_c_series)]
    else:
        final_c_series = np.array(final_c_series)

    var_series = np.array(var_series)
    return final_c_series, var_series

def run_linear_grazing(sigma, poisson_u=False):
# Kefi et al., Oikos (2013), model 2, showing transcritical bifurcation

    r = 1
    K = 10
    dt = 0.001 # time step for Euler-Maruyama
    n_samples = 100 # number of samples per bifurcation parameter value for computing sample var
    sample_gap = int(1.0//dt) # steps between samples to ensure independence
    num_u = 50

    c_min = 0.0
    c_max = 1.0
    if poisson_u:
        org_c_series = np.sort(np.random.uniform(low=c_min, high=c_max, size=num_u))
    else:
        org_c_series = np.linspace(c_min, c_max, num_u)  # Cannot be reduced down to 0.02 (causing error)

    def f_Kefi2013_model2(x, c):
        if x<0:
            return 0.0
        else:
            return r * x * (1-x/K) - c * x

    final_c_series = []
    var_series = []
    terminate = False
    
    for c in org_c_series:
        if terminate:
            break
        x = (r-c)/r * K # Start from the equilibrium in the absence of noise

        # Equilibrate (transient decay)
        equil_steps = int(10//dt) # lengthen for stronger separation if needed
        for _ in range(equil_steps):
            dx = f_Kefi2013_model2(x, c) * dt + sigma * np.sqrt(dt) * np.random.randn()
            x += dx

        # Sample 100 points, well-separated in time
        samples = []
        crossed = False
        for _ in range(n_samples):
            for _ in range(sample_gap):
                dx = f_Kefi2013_model2(x, c) * dt + sigma * np.sqrt(dt) * np.random.randn()
                x += dx
                if x < 0:
                    crossed = True
                    break
            if crossed:
                break
            samples.append(x) # If x < 0 at any point, do not append x and terminate all future simulations

        if crossed:
            terminate = True
            print(f"Terminating all further c at c = {c:.3f}, because x crossed {-0.01}.")
            break
        else:
#            print(samples)
            var_series.append(np.var(samples)) # Compute sample standard deviation and add to the existing array
            final_c_series.append(c)

    # dynamics simulations finished

    if poisson_u:
        equi_c_series = np.linspace(c_min, c, len(equi_c_series)) # this is probably one off, so should be remedied if I use the poisson_u option.
        final_c_series = equi_c_series[:len(final_c_series)]
    else:
        final_c_series = np.array(final_c_series)

    var_series = np.array(var_series)
    return final_c_series, var_series

def run_Kefi_model3(sigma, poisson_u=False):
# Kefi et al., Oikos (2013), model 3, showing Hopf bifurcation

    r = 0.5
    g = 0.4
    h = 0.6
    e = 0.6
    m = 0.15
    dt = 0.001 # time step for Euler-Maruyama
    n_samples = 100 # number of samples per bifurcation parameter value for computing sample var
    sample_gap = int(10.0//dt) # steps between samples to ensure independence
    num_u = 50

    K_min = 1.1 # 1.5 # 2.1 # 1.5 # Kefi et al. (2013) starts from K=0.1, 
    # but here we avoid transcritical bifurcation at K \approx 1
    K_max = h*(e*g+m)/(e*g-m) # Hopf bifurcation occurs at K = 2.6
    if poisson_u:
        org_K_series = np.sort(np.random.uniform(low=K_min, high=K_max, size=num_u))
    else:
        org_K_series = np.linspace(K_min, K_max, num_u)  # Cannot be reduced down to 0.02 (causing error)

    final_K_series = []
    var_series = []
    terminate = False
    
    for K in org_K_series:
        if terminate:
            break

        # start from the analytically obtained equilibrium
        x = m*h/(e*g-m)
        y = r/g * (1-x/K) * (x+h)

        # Equilibrate (transient decay)
        equil_steps = int(10//dt) # lengthen for stronger separation if needed
        for _ in range(equil_steps):
            dx = (r * x * (1-x/K) - g * x / (x+h) * y) * dt + sigma * np.sqrt(dt) * np.random.randn()
            dy = (e * g * x / (x+h) * y - m * y) * dt + sigma * np.sqrt(dt) * np.random.randn()
            x += dx
            y += dy

        # Sample 100 points, well-separated in time
        samples = []
        crossed = False
        for _ in range(n_samples):
#            abs_dev_x = 0
            for _ in range(sample_gap):
                dx = (r * x * (1-x/K) - g * x / (x+h) * y) * dt + sigma * np.sqrt(dt) * np.random.randn()
                dy = (e * g * x / (x+h) * y - m * y) * dt + sigma * np.sqrt(dt) * np.random.randn()
                x += dx
                y += dy
                if x < 0:
#                    crossed = True
#                    break
                    x = 0
                if y < 0:
#                    crossed = True
#                    break
                    y = 0
#                abs_dev_x += abs(x-1)
#            abs_dev_x /= sample_gap
#            if abs_dev_x > 0.1:
#                crossed = True
#                break
            samples.append(x) # If x < 0 at any point, do not append x and terminate all future simulations

        if crossed:
            terminate = True
            print(f"Terminating all further K at K = {K:.3f}, because abs(x-1) exceeded {0.1}.")
            break
        else:
#            print(samples)
            var_series.append(np.var(samples)) # Compute sample standard deviation and add to the existing array
            final_K_series.append(K)

    # dynamics simulations finished

    if poisson_u:
        equi_K_series = np.linspace(K_min, K, len(equi_K_series)) # this is probably one off, so should be remedied if I use the poisson_u option.
        final_K_series = equi_K_series[:len(final_K_series)]
    else:
        final_K_series = np.array(final_K_series)

    var_series = np.array(var_series)
    return final_K_series, var_series