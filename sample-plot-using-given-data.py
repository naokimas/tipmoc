"""
Plot hat{V} versus u for three simulation runs (by setting trials = 3) with different line colors.
Produces Figure 2(c) (by setting dynamics = 1) and Figure 4 (by setting dynamics = 4, 5, 6, and 7 for panels (a), (b), (c), and (d), respectively).

Differently from sample-plot.py, this code does not run simulations but uses the saved simulation data, allowing fine tuning of the appearance of figures.
This code was used for producing final Figures 2(c) and 4.

Input:

    sample_variance_trials-X.csv --> X is the name of the dynamical system. This file is renamed from sample_variance_trials.csv returned by sample-plot.py

    detection_summary-X.csv --> Same as above.

Output:

    variance_trials_plot.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

dynamics = 7
trials = 3 # number of runs
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

if dynamics == 1: # double well
    u_c = 3.079 # theoretical critical value
    var_df = pd.read_csv('sample_variance_trials-dw.csv')
    summary_df = pd.read_csv('detection_summary-dw.csv')
elif dynamics == 4: # over-grazing with K=10, showing saddle-node bifurcation
    u_c = 2.604
    var_df = pd.read_csv('sample_variance_trials-overharvestingK10.csv')
    summary_df = pd.read_csv('detection_summary-overharvestingK10.csv')
elif dynamics == 5: # linear grazing
    u_c = 1.0
    var_df = pd.read_csv('sample_variance_trials-lineargrazing.csv')
    summary_df = pd.read_csv('detection_summary-lineargrazing.csv')
elif dynamics == 6: # Rosenzweig-MacArthur
    u_c = 2.6
    var_df = pd.read_csv('sample_variance_trials-rosenzweig.csv')
    summary_df = pd.read_csv('detection_summary-rosenzweig.csv')
elif dynamics == 7: # mutualistic-interaction
    u_c = 0.047
    var_df = pd.read_csv('sample_variance_trials-mutualistic.csv')
    summary_df = pd.read_csv('detection_summary-mutualistic.csv')

# -----------------------------
# Load data
# -----------------------------

u_values_sample = var_df['u'].values
variance_matrix = var_df.drop(columns=['u']).values

summary_data = summary_df.to_dict(orient='records')

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 5))
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

u_counts = {}

# -----------------------------
# Plot
# -----------------------------
for tr in range(trials):
    color = colors[tr % len(colors)]

    ax.plot(
        u_values_sample,
        variance_matrix[:, tr],
        color=color,
        linewidth=1.5,
        alpha=0.8
    )

    u_det = summary_data[tr]['u_at_detection']
    hatuc = summary_data[tr]['hatuc']
    detected = summary_data[tr]['detected']

    if detected:
        u_key = round(u_det, 5)
        count = u_counts.get(u_key, 0)

        y_pos = (0.02 * y_limit) + (count * 0.025 * y_limit)
        ax.scatter(
            u_det, y_pos,
            marker='^',
            color=color,
            s=70,
            zorder=5
        )

        u_counts[u_key] = count + 1
        all_x_points.append(u_det)

        if not np.isnan(hatuc):
            top_y = 0.15 * y_limit
            bottom_y = 0.0375 * y_limit
            ax.vlines(
                x=hatuc,
                ymin=bottom_y,
                ymax=top_y,
                color=color,
                linestyle=':',
                linewidth=2.0
            )
            all_x_points.append(hatuc)

# -----------------------------
# Axis limits
# -----------------------------
if dynamics == 7:
    hatuc_vals = [
        d['hatuc'] for d in summary_data
        if not np.isnan(d['hatuc'])
    ]
    u_min = min(hatuc_vals) if hatuc_vals else np.min(u_values_sample)
else:
    u_min = np.min(u_values_sample)

u_max_base = np.max(u_values_sample)
global_x_max = max(all_x_points)
x_range = u_max_base - u_min

ax.set_xlim(
    left=u_min - x_range * 0.0125,
    right=global_x_max + x_range * 0.02
)

# -----------------------------
# Labels and grid
# -----------------------------
ax.set_xlabel(r'$u$', fontsize=18, labelpad=-3)
ax.set_ylabel(r'$\hat{V}$', fontsize=18, labelpad=-3)
ax.grid(True, linestyle=':', alpha=0.6)
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.yaxis.get_offset_text().set_fontsize(14)

# -----------------------------
# Panel label
# -----------------------------
panel_labels = {1: '(c)', 4: '(a)', 5: '(b)', 6: '(c)', 7: '(d)'}
if dynamics in panel_labels:
    ax.annotate(
        panel_labels[dynamics],
        xy=(-0.08, 1.09),
        xycoords='axes fraction',
        fontsize=20,
        ha='left',
        va='top'
    )

plt.savefig('variance_trials_plot.pdf')
plt.show()

print("Success: Figure generated from saved CSV files.")