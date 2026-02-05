# Generate a scattergram between u_{det} and \hat{u}_c (Figure 3(a) in the manuscript) and one between \tau and \hat{u}_c (Figure 3(b)) from the summary data already obtained

# Usage: python plot_hatuc_scatter_dw.py ews_result_dw.csv
# Output: scatter_udet_hatuc_dw.pdf and scatter_tau_hatuc_dw.pdf

import argparse
import sys
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def parse_args():
    p = argparse.ArgumentParser(description="Generate specific scatterplot for detection diagnostics.")
    p.add_argument("csvfile", help="Path to input CSV file")
    p.add_argument("--out-prefix", default="scatter", help="Prefix for output figure")
    return p.parse_args()

def to_bool_like(x):
    """Robustly map typical True/False values to Python bool."""
    if pd.isnull(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return np.nan

def create_diagnostic_plot(csvfile, out_prefix):
    # Load data
    try:
        df = pd.read_csv(csvfile)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Identify "True" rows based on 'detected' column.
    # "True" means that an impending bifurcation is detected.
    col_name = "detected" if "detected" in df.columns else "detected_AIC"
    detected_bool = df[col_name].apply(to_bool_like)
    
    # Filter for True rows and necessary columns
    df_true = df[detected_bool == True].copy()
    
    # Ensure numeric types
    x_col = "c_at_detection" # u_{det}
    y_col = "rdiv_at_detection" # \hat{u}_c
    z_col = "tau" # \tau
    data = df_true[[x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        print("No valid data points found where detected is True. Check column names/values.")
        return
    x = data[x_col].values # u_{det}
    y = data[y_col].values # \hat{u}_c
    z = data[z_col].values # \tau

    # Generate the u_{det} vs \hat{u}_c Figure (Figure 3(a))

    x_range = x.max() - x.min()
    y_range = y.max() - x.min()
    padding_x = x_range * 0.025
    padding_y = y_range * 0.025
    #
    # Force the PHYSICAL box to be square
    # We use the padded ranges to ensure the box remains a perfect square
    padded_x_range = (x.max() + padding_x) - (x.min() - padding_x)
    padded_y_range = (y.max() + padding_x) - (x.min() - padding_x)

    # Plotting u_{det} vs \hat{u}_c
    plt.figure(figsize=(7, 6))
    
    # Scatter plot: small circles
    # s: size
    # alpha: transparency (1 = completely opaque, 0 = completely transparent)
    plt.scatter(x, y, s=20, facecolors='none', edgecolors='blue', alpha=0.7, label='Data points')

    # Horizontal red dashed line at u_c = 3.0792
    plt.axhline(y=3.0792, color='red', linestyle='--', linewidth=1.5, label='Bifurcation point')

    # Draws a diagonal line that extends to the axes edges
    plt.axline((0, 0), slope=1, color='black', linestyle='-', linewidth=1, zorder=0)

    # Configure axes
    ax = plt.gca()

    ax.set_xlim(x.min() - padding_x, x.max() + padding_x)
    ax.set_ylim(x.min() - padding_x, y.max() + padding_x)

    if padded_x_range != 0:
        ax.set_aspect(padded_x_range / padded_y_range)    

    # Pearson correlation
    r_val, _ = stats.pearsonr(x, y)
    plt.text(0.75, 0.1, f"$r$ = {r_val:.3f}", transform=ax.transAxes, 
             verticalalignment='top', fontsize=16)
    plt.text(0.05, 0.95, "(a)", transform=ax.transAxes, 
             verticalalignment='top', fontsize=20)

    plt.xlabel(r"$u$ at detection, $u_{\text{det}}$", fontsize=18)
    plt.ylabel(r"predicted bifurcation point, $\hat{u}_{\text{c}}$", fontsize=18)

    plt.tight_layout()

    # Determine filename
    base = os.path.basename(csvfile)
    m = re.match(r"^ews_result_(.+)\.csv$", base, flags=re.IGNORECASE)
    suffix = m.group(1) if m else os.path.splitext(base)[0]
    output_fn = f"scatter_udet_hatuc_{suffix}.pdf"

    plt.subplots_adjust(right=0.99, left=0.13, top=0.99, bottom=0.1)
    plt.savefig(output_fn, dpi=300)
    print(f"Plot saved to: {output_fn}")
    plt.close()

    # Figure 3(a) done

    # Generate the \tau vs \hat{u}_c Figure (Figure 3(b))

    plt.figure(figsize=(7, 6))
    
    # Scatter plot: small circles
    # s: size
    # alpha: transparency (1 = completely opaque, 0 = completely transparent)
    plt.scatter(z, y, s=20, facecolors='none', edgecolors='blue', alpha=0.7, label='Data points')

    # Horizontal red dashed line at u_c = 3.0792
    plt.axhline(y=3.0792, color='red', linestyle='--', linewidth=1.5, label='Bifurcation point')

    # Configure axes
    ax = plt.gca()

    z_range = z.max() - z.min()
    padding_z = z_range * 0.025
    padded_z_range = (z.max() + padding_z) - (z.min() - padding_z)

    ax.set_xlim(z.min() - padding_z, z.max() + padding_z)
    ax.set_ylim(x.min() - padding_x, y.max() + padding_x)

    if padded_z_range != 0:
        ax.set_aspect(padded_z_range / padded_y_range)

    # Pearson correlation
    r_val, _ = stats.pearsonr(z, y)
    plt.text(0.75, 0.1, f"$r$ = {r_val:.3f}", transform=ax.transAxes, 
             verticalalignment='top', fontsize=16)
    plt.text(0.05, 0.95, "(b)", transform=ax.transAxes, 
             verticalalignment='top', fontsize=20)

    plt.xlabel(r"$\tau$", fontsize=18)
    plt.ylabel(r"predicted bifurcation point, $\hat{u}_{\text{c}}$", fontsize=18)

    plt.tight_layout()

    # Determine filename
    base = os.path.basename(csvfile)
    m = re.match(r"^ews_result_(.+)\.csv$", base, flags=re.IGNORECASE)
    suffix = m.group(1) if m else os.path.splitext(base)[0]
    output_fn = f"scatter_tau_hatuc_{suffix}.pdf"

    plt.subplots_adjust(right=0.99, left=0.13, top=0.99, bottom=0.1)
    plt.savefig(output_fn, dpi=300)
    print(f"Plot saved to: {output_fn}")
    plt.close()

    # Figure 3(b) done

if __name__ == "__main__":
    args = parse_args()
    create_diagnostic_plot(args.csvfile, args.out_prefix)