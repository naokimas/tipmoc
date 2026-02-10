"""
Compute the saddle-node bifurcation point (u_c) for the double-well dynamical system by solving
    u = g(x) = (x - r1) * (x - r2) * (x - r3)
by solving g'(x) = 0 and evaluating g at the left critical point (the one in (r1, r2)).
"""

import math

"""
Return (xcrit, ucrit) for the saddle-node bifurcation.
- r1, r2, r3: real numbers (not necessarily sorted). The function sorts them ascending.
- xcrit: critical x in (r1, r2) where g'(x) = 0
- ucrit: g(xcrit) -- the bifurcation parameter value
"""

r1 = 1
r2 = 3
r3 = 5

# Coefficients for g(x) = (x-r1)(x-r2)(x-r3)
S1 = r1 + r2 + r3
S2 = r1*r2 + r1*r3 + r2*r3

# Solve g'(x) = 3 x^2 - 2 S1 x + S2 = 0
disc = S1*S1 - 3.0 * S2 # discriminant
if disc < 0.0:
    raise ValueError("g'(x) has no real roots (disc < 0). Cannot find real saddle nodes.")

sqrt_disc = math.sqrt(disc)
xcrit = (S1 - sqrt_disc) / 3.0
ucrit = (xcrit - r1) * (xcrit - r2) * (xcrit - r3)

print("x_critical     u_critical")
print(f"{xcrit:.4g}  {ucrit:10.4f}")