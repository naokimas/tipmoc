# Generate a schematic figure to explain TIPMOC (figure 1 in the manuscript)

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters of the power-law function to be fitted to data
# -----------------------------
b = 0.6
u_c_hat = 1.0
gamma = 0.45
a = 0.3

# -----------------------------
# Artificial observed points
# -----------------------------
u_data = np.linspace(0.68, 0.96, 6)

# Base power-law values
V_base = a * (u_c_hat - u_data) ** (-gamma) + b

# Artificial offsets for natural look
offsets = np.array([-0.10, 0.08, +0.06, -0.14, -0.05, 0.16])
V_data = V_base + offsets

# -----------------------------
# Power-law curve
# -----------------------------
u_curve = np.linspace(0.65, u_c_hat - 0.001, 500)
V_curve = a * (u_c_hat - u_curve) ** (-gamma) + b

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 5))

# Observed points
ax.scatter(u_data, V_data, color='black', s=50, zorder=3, label=r'Observed $\hat{V}(u)$')

# Power-law curve
ax.plot(u_curve, V_curve, color='black', linewidth=2, label='Power-law fit')

# Vertical dotted line for \hat{u}_c
ax.axvline(u_c_hat, color='blue', linestyle=':', linewidth=2)

# Horizontal dashed line for b
ax.axhline(b, color='magenta', linestyle='--', linewidth=2)

# -----------------------------
# Annotations
# -----------------------------
# Label for \hat{u}_c
ax.annotate(
    r'$\hat{u}_{\mathrm{c}}$',
    xy=(u_c_hat, 0), xycoords='data',
    xytext=(0, -4), textcoords='offset points',  # below x-axis label
    color='blue',
    fontsize=16,
    ha='center', va='top'
)

# Label for b
ax.annotate(
    r'$b$',
    xy=(0.655, b), xycoords='data',
    xytext=(-10, 0), textcoords='offset points',
    color='magenta',
    fontsize=16,
    ha='right', va='center'
)

# -----------------------------
# Axes styling
# -----------------------------
ax.set_xlim(0.65, 1.02)
ax.set_ylim(0, 5)
ax.set_xlabel(r'$u$', fontsize=16)
ax.set_ylabel(r'$\hat{V}(u)$', fontsize=16)

# Remove tick labels and marks
ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(axis='both', which='both', length=0)

# Clean schematic look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(frameon=False, fontsize=14)

fig.tight_layout()
fig.savefig('schematic-powerlaw-fit.pdf')
plt.show()
