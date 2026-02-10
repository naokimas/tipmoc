"""
Plot hat{V} versus u for three simulation runs (by setting trials = 3) with different line colors.
Produces Fig. 2(c) (by setting dynamics = 1) and Fig. 4 (by setting dynamics = 4, 5, 6, and 7 for panels (a), (b), (c), and (d), respectively).

Output:
    sample_variance_trials.csv --> I renamed this by adding the dynamical system's name.
    detection_summary.csv --> Same as above.
    variance_trials_plot.pdf --> However, we fine-tune its appearance of this figure by running sample-plot-using-given-data.py, producing the final figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import run_double_well as dw
import run_over_harvesting as oh
import run_mutualistic as mu
import run_ou as ou
import delta_aicc as dAICc    
from scipy.stats import kendalltau
import matplotlib.ticker as ticker # adjust ytics number labels

trials = 3
i0 = 8
threshold_AICc = -10
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
num_u = 50
dynamics = 7

# --- Setup parameters ---
if dynamics == 1: # double-well
    sigma = 0.05
    u_c = 3.079
    u_values_sample = np.linspace(0.0, u_c, num_u)
elif dynamics == 4: # over-harvesting with K=10
    sigma = 0.05
    u_c = 2.604
    u_values_sample = np.linspace(1, u_c, num_u)
elif dynamics == 5: # linear grazing
    sigma = 0.05
    u_c = 1.0
    u_values_sample = np.linspace(0.0, u_c, num_u)
elif dynamics==6: # Rosenzweig-MacArthur
    sigma = 0.01
    u_c = 2.6
    u_values_sample = np.linspace(1.1, u_c, num_u)
elif dynamics==7: # mutualistic-interaction
    sigma = 0.15
    u_c = 0.0
    u_values_sample = np.linspace(1.0, u_c, num_u)

variance_matrix = np.full((len(u_values_sample), trials), np.nan)
summary_data = []

# --- Run simulations ---
for tr in range(trials): # number of simulation runs
    if dynamics==1:
        u_values, var_values = dw.run_double_well(sigma=sigma)
    elif dynamics==4: # over-harvesting with saddle-node bifurcation
        u_values, var_values = oh.run_over_harvesting(K=10, sigma=sigma)
    elif dynamics==5: # linear grazing
        u_values, var_values = oh.run_linear_grazing(sigma=sigma)
    elif dynamics==6: # Rosenzweig-MacArthur
        u_values, var_values = oh.run_Kefi_model3(sigma=sigma)
    elif dynamics==7: # mutualistic-interaction
        u_values, var_values = mu.run_mutualistic(sigma=sigma) # u_values is sign-flipped
       
    # Fill the matrix safely
    variance_matrix[:len(var_values), tr] = var_values 
        
    # AICc Logic
    current_num_u = len(u_values)
    delta_AICc_values = []
    hatuc_raw = []
        
    for i in range(i0, current_num_u + 1):
        window_u = u_values[:i]
        window_var = var_values[:i]
        res = dAICc.delta_AICc(window_u, window_var)
        delta_AICc_values.append(res[0])
        hatuc_raw.append(res[4])
        
    detected = False
    u_det = np.nan
    hatuc = np.nan
        
    d_AICc = np.array(delta_AICc_values)
    for j in range(len(d_AICc) - 2):
        if np.all(~np.isnan(d_AICc[j:j+3])) and np.all(d_AICc[j:j+3] <= threshold_AICc):
            detected = True
            det_idx = i0 + j + 2 
            u_det = u_values[det_idx - 1]
            hatuc = hatuc_raw[j + 2]
            if dynamics==7: # mutualistic interaction
                u_det = -u_det
                hatuc = -hatuc
            break
        
    summary_data.append({
        'trial': tr + 1,
        'detected': detected,
        'u_at_detection': u_det,
        'hatuc': hatuc
    })

# --- Save data ---
var_df = pd.DataFrame(variance_matrix, columns=[f'Trial_{i+1}' for i in range(trials)])
var_df.insert(0, 'u', u_values_sample)
var_df.to_csv('sample_variance_trials.csv', index=False)
pd.DataFrame(summary_data).to_csv('detection_summary.csv', index=False)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(7, 5))

# Set fixed margins (values are fractions of the figure size 0 to 1)
# left=0.15 gives enough room for the scientific notation
fig.subplots_adjust(left=0.12, right=0.98, top=0.9, bottom=0.1)

ax.set_ylim(bottom=0)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) # Force the y-axis to use scientific notation for 10^-X

# ---- Dynamics-specific y-limits & formatting ----
if dynamics == 1:
    ax.set_ylim(top=0.0016)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.0005)) # Set the tick positions every 0.0005
elif dynamics==4 or dynamics==5 or dynamics==6:
    if dynamics == 4 or dynamics==6:
        ax.set_ylim(top=0.05)
    elif dynamics == 5:
        ax.set_ylim(top=0.03)
    formatter.set_powerlimits((-2, -2)) # Forces 10^-2 specifically
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
elif dynamics==7:
    ax.set_ylim(top=0.01)
    formatter.set_powerlimits((-3, -3)) # Forces 10^-3 specifically
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.002))

ax.yaxis.set_major_formatter(formatter)
y_limit = ax.get_ylim()[1]

all_x_points = list(u_values_sample) + [u_c]

# Draw theoretical critical point
ax.axvline(x=u_c, color='black', linestyle='--', linewidth=1.5)

# Dictionary to track how many triangles are at the same x-position
u_counts = {}

for tr in range(trials):
    color = colors[tr % len(colors)]
    ax.plot(u_values_sample, variance_matrix[:, tr], color=color, linewidth=1.5, alpha=0.8)
        
    u_det = summary_data[tr]['u_at_detection']
    hatuc = summary_data[tr]['hatuc']
        
    if summary_data[tr]['detected']:
        # --- Marker stacking logic ---
        # Use rounding to ensure floating point u-values are grouped correctly
        u_key = round(u_det, 5) 
        count = u_counts.get(u_key, 0)
            
        # Base y is 2% of axis height. Shifts up by 2.5% of axis height.
        y_pos = (0.02 * y_limit) + (count * 0.025 * y_limit)
            
        ax.scatter(u_det, y_pos, marker='^', color=color, s=70, zorder=5)
        u_counts[u_key] = count + 1 # Increment for next trial
        all_x_points.append(u_det)
            
        # --- Dotted line logic ---
        if not np.isnan(hatuc):
            # Fixed at 15% and 3.75% of the total box height
            top_y = 0.15 * y_limit
            bottom_y = 0.0375 * y_limit 
            ax.vlines(x=hatuc, ymin=bottom_y, ymax=top_y, 
                      color=color, linestyle=':', linewidth=2.0)
            all_x_points.append(hatuc)

# --- Axis limit logic ---
if dynamics == 7: # mutualistic-interaction dynamics
    # Extract all hatuc values, ignoring NaNs
    hatuc_values = [d['hatuc'] for d in summary_data if not np.isnan(d['hatuc'])]
        
    # If we have hatuc values, use the minimum of those; otherwise, fallback to sample min
    if hatuc_values:
        u_min = min(min(hatuc_values), np.min(u_values_sample))
    else:
        u_min = np.min(u_values_sample)
else:
    u_min = np.min(u_values_sample)

u_max_base = np.max(u_values_sample)
global_x_max = max(all_x_points)
    
x_range = u_max_base - u_min
ax.set_xlim(left=u_min - x_range * 0.0125, right=global_x_max + x_range * 0.02)

ax.set_xlabel(r'$u$', fontsize=18)
ax.set_ylabel(r'$\hat{V}$', fontsize=18)
ax.grid(True, linestyle=':', alpha=0.6)

if dynamics==1:
    ax.annotate(
        '(c)',
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left', va='top'
    )
elif dynamics==4:
    ax.annotate(
        '(a)',
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left', va='top'
    )
elif dynamics==5:
    ax.annotate(
        '(b)',
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left', va='top'
    )
elif dynamics==6:
    ax.annotate(
        '(c)',
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left', va='top'
    )
elif dynamics==7:
    ax.annotate(
        '(d)',
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left', va='top'
    )

plt.savefig('variance_trials_plot.pdf')
plt.show()

print(f"Success: Files generated. Black dashed line at u={u_c} added.")