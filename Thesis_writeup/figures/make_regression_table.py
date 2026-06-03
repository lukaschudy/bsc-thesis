"""Saturated 2^3 factorial OLS on the N=1000 cell means.

With dummy coding L, sigma, T in {0,1}, the design is fully saturated (8
cells, 8 parameters). Coefficients are exact linear contrasts of cell
means; standard errors come from the cross-session SDs scaled by N.

Prints a LaTeX table body to stdout (paste into results_factorial.tex).
"""
import numpy as np
from scipy.stats import norm

# (cell, L, sigma, T, mean_Delta, sd_Delta) at N = 1000 sessions per cell
CELLS = [
    ('F000', 0, 0, 0, 0.848, 0.109),
    ('F100', 1, 0, 0, 0.210, 0.122),
    ('F010', 0, 1, 0, 0.500, 0.298),
    ('F001', 0, 0, 1, 0.383, 0.111),
    ('F110', 1, 1, 0, 0.059, 0.080),
    ('F101', 1, 0, 1, 0.346, 0.174),
    ('F011', 0, 1, 1, 0.280, 0.103),
    ('F111', 1, 1, 1, 0.537, 0.135),
]
N = 1000  # sessions per cell

# Build saturated design matrix X (8 x 8) with columns:
# 1, L, sigma, T, L*sigma, L*T, sigma*T, L*sigma*T
rows, y, var_y = [], [], []
for name, L, s, T, mean, sd in CELLS:
    rows.append([1, L, s, T, L*s, L*T, s*T, L*s*T])
    y.append(mean)
    var_y.append(sd**2 / N)         # variance of the cell mean

X      = np.array(rows, dtype=float)
y      = np.array(y, dtype=float)
SigmaY = np.diag(var_y)             # diagonal cov-matrix of cell means

# Saturated OLS: beta = X^{-1} y, Var(beta) = X^{-1} Sigma X^{-T}
Xinv      = np.linalg.inv(X)
beta      = Xinv @ y
var_beta  = Xinv @ SigmaY @ Xinv.T
se_beta   = np.sqrt(np.diag(var_beta))
z         = beta / se_beta
p_two     = 2 * (1 - norm.cdf(np.abs(z)))

NAMES = ['Intercept', r'L', r'\sigma', r'T',
         r'L \cdot \sigma', r'L \cdot T', r'\sigma \cdot T',
         r'L \cdot \sigma \cdot T']

def stars(p):
    if p < 0.001:
        return r'\textsuperscript{***}'
    if p < 0.01:
        return r'\textsuperscript{**}'
    if p < 0.05:
        return r'\textsuperscript{*}'
    return ''

print('Coefficient table:')
print(f"{'Term':25s}  {'Estimate':>10s}  {'SE':>10s}  {'z':>8s}  {'p':>10s}")
for n, b, se, zv, pv in zip(NAMES, beta, se_beta, z, p_two):
    print(f"{n:25s}  {b:+10.4f}  {se:10.4f}  {zv:+8.2f}  {pv:10.3e} {stars(pv)}")

print()
print('=== LaTeX table body (booktabs) ===')
for n, b, se, zv, pv in zip(NAMES, beta, se_beta, z, p_two):
    print(f'${n}$ & ${b:+.3f}${stars(pv)} & ${se:.3f}$ & ${zv:+.1f}$ & ${pv:.3g}$ \\\\')

# Joint chi-squared test for "all interactions = 0"
# Indices 4,5,6,7 are L*s, L*T, s*T, L*s*T
J = np.zeros((4, 8))
for i, idx in enumerate([4, 5, 6, 7]):
    J[i, idx] = 1.0
beta_J     = J @ beta
var_beta_J = J @ var_beta @ J.T
chi2       = beta_J.T @ np.linalg.inv(var_beta_J) @ beta_J
from scipy.stats import chi2 as chi2_dist
p_joint    = 1 - chi2_dist.cdf(chi2, df=4)
print()
print(f'Joint Wald test (all four interactions = 0):')
print(f'  chi2(4) = {chi2:.2f},  p = {p_joint:.3e}')
