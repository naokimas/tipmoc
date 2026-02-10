"""
Fits the power-law model y - b = a / (hatu_c - u)^{gamma}
Returns best-fit parameters a, hatu_c, gamma, b and Pearson correlation coefficient.
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr

def fit_powerlaw(x, y):
    # x and y are numpy arrays. x = u, y = hat{V}

    # Validate input
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError("x and y must be numpy arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length.")

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    epsilon = 1e-4

    # hatuc's range
    hatuc_lower = max_x + epsilon   # strictly greater than max_x
    hatuc_upper = max_x + (max_x - min_x) * 10
    if hatuc_lower + epsilon > hatuc_upper: # + epsilon is not necessary, but to safeguard
        raise ValueError("hatuc_lower < hatuc_upper must hold.")

    # b's range
    b_lower = 0 # min_y - (max_y - min_y) / 2
    b_upper = 0.9999 * min_y

    bounds = [(hatuc_lower, hatuc_upper), (b_lower, b_upper)]

    # Objective: Pearson correlation closest to -1 --> minimize abs(corr + 1)
    def objective(params):
        hatuc, b = params

        if hatuc <= max_x or b >= min_y:
            return 1e6  # Large penalty

        lnX = np.log(hatuc-x)
        lnY = np.log(y-b)

        corr, _ = pearsonr(lnX, lnY)

        # We want corr close to -1 (i.e., as small as possible), so minimize corr+1
        return corr + 1

    # Run global optimization
    result = differential_evolution(objective, bounds, tol=1e-5)

    hatuc_opt, b_opt = result.x
    if hatuc_opt <= max_x:
        raise ValueError("hatuc must be larger than max_x.")

    lnX = np.log(hatuc_opt - x)
    lnY = np.log(y - b_opt)

    # Linear fit: lnY = ln(a) - gamma * lnX
    coeffs = np.polyfit(lnX, lnY, 1)
    a = np.exp(coeffs[1]) # coeffs[1]: intercept obtained from linear regression
    gamma = -coeffs[0] # coeffs[0]: slope obtained from linear regression

    # Compute best Pearson correlation coefficient (should be ideally close to -1)
    best_corr, _ = pearsonr(lnX, lnY)

    return a, b_opt, hatuc_opt, gamma, best_corr