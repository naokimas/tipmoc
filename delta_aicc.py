"""
Compares the linear and power-law fits in terms of AIC_c
Returns Delta AIC_c, best linear fit parameters (2-dim vector popt_lin), best power-law fit parameters (a, hatu_c, gamma, b), and Pearson correlation coefficient for the best power-law fit.
"""

import numpy as np
from scipy.optimize import curve_fit
import fit_powerlaw as fitpow

def delta_AICc(x_values, y_values):
    # x and y are numpy arrays. x = u, y = hat{V}

    # Convert to numpy arrays (if not yet)
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Linear model
    def f_lin(x, a, b):
        return a * x + b

    try:
        popt_lin, _ = curve_fit(f_lin, x_values, y_values)
    except RuntimeError as e:
        return None, None, None, None, 1  # Non-converge

    y_values_hat_lin = f_lin(x_values, *popt_lin)
    resid_lin = y_values - y_values_hat_lin
    rss_lin = np.sum(resid_lin**2)
    n = len(y_values)
    k_lin = 2  # number of parameters for linear
    AICc_lin = n * np.log(rss_lin / n) + 2 * k_lin + (2*k_lin*(k_lin+1))/(n - k_lin - 1)
    # rss_lin / n = \hat{\sigma}^2 of the normal distribution of error

    a, b, hatuc, gamma, best_corr = fitpow.fit_powerlaw(x_values, y_values)
    y_values_hat_powerlaw = a / np.power(hatuc - x_values, gamma) + b
    resid_powerlaw = y_values - y_values_hat_powerlaw
    rss_powerlaw = np.sum(resid_powerlaw**2)
    k_powerlaw = 4  # number of parameters for the power-law fit
    AICc_powerlaw = n * np.log(rss_powerlaw / n) + 2 * k_powerlaw + (2*k_powerlaw*(k_powerlaw+1))/(n - k_powerlaw - 1)

    # Negative values favor power-law fit, positive favor linear fit.
    delta_AICc = AICc_powerlaw - AICc_lin

    return delta_AICc, popt_lin, a, b, hatuc, gamma, best_corr