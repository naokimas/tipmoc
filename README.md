# TIPMOC (TIpping via Power‑law fits and MOdel Comparison)

This repository lists the code to run TIPMOC, including the code that is necessary to produce the figures and table in the paper.
When you use the code, please cite the following paper:

xxx

## Code

`explain_tipmoc.py`  Produce a schematic figure (Figure 1 in the manuscript).

`plot_hatuc_scatter_dw.py` Produce diagnostic scattergrams (Figure 3(a) and 3(b)). See the first few lines of the code for the usage.

`bifu_pt_double_well.py` Determine the lower saddle-node bifurcation point of the double-well dynamical system. The obtained bifurcation point value (i.e., 3.079) is referred to in the manuscript.

`bifu_pt_mutualistic.py` Determine the collapse point of the mutualistic-interaction dynamical system. The obtained collapse point value (i.e., 0.0470) is referred to in the manuscript.

## Data

`LFR.csv` The LFT network used in the simulation of the mutualistic-interaction dynamics. This network is identical to the one used in [Maclaren, Barzel & Masuda, Nature Communications, 2025](https://doi.org/10.1038/s41467-025-64975-x)
