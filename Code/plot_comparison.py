"""Combined latency vs noise figure for the thesis / supervisor.

Two panels side by side so the x-axis units (periods vs index-sigma) aren't
conflated. Same y-axis on both panels makes shape comparison direct.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ----------------------------------------------------------------------
# Data (A_convResults.txt / avgPrGain, 250 sessions each)
# ----------------------------------------------------------------------
latency = [0,       1,       2,       3,       5,       10,      20]
delta_L = [0.84280, 0.27957, 0.21490, 0.19281, 0.17365, 0.15786, 0.15503]

sigma   = [0.0,     0.5,     1.0,     2.0,     3.0,     5.0,     10.0]
delta_N = [0.84280, 0.67049, 0.49269, 0.24244, 0.14261, 0.01326, -0.00258]

# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# --- Latency panel ----------------------------------------------------
ax1.plot(latency, delta_L, color='steelblue', linewidth=2,
         marker='o', markersize=7, markerfacecolor='white',
         markeredgecolor='steelblue', markeredgewidth=2,
         label='Observation latency')
ax1.axhline(y=delta_L[0], color='green', linestyle='--', linewidth=1.2,
            label=f'Baseline:  Δ = {delta_L[0]:.3f}')
ax1.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2,
            label='Nash:  Δ = 0')
for l, d in zip(latency, delta_L):
    ax1.annotate(f'{d:.3f}', xy=(l, d),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=8, color='steelblue')
ax1.set_xlabel('Observation Latency  L  (periods)', fontsize=12)
ax1.set_ylabel('Average Profit Gain  Δ', fontsize=12)
ax1.set_title('Friction 1: Observation Latency', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.5, 21)
ax1.set_ylim(-0.1, 1.0)
ax1.set_xticks(latency)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# --- Noise panel ------------------------------------------------------
ax2.plot(sigma, delta_N, color='darkorange', linewidth=2,
         marker='s', markersize=7, markerfacecolor='white',
         markeredgecolor='darkorange', markeredgewidth=2,
         label='Noisy monitoring')
ax2.axhline(y=delta_N[0], color='green', linestyle='--', linewidth=1.2,
            label=f'Baseline:  Δ = {delta_N[0]:.3f}')
ax2.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2,
            label='Nash:  Δ = 0')
for s, d in zip(sigma, delta_N):
    ax2.annotate(f'{d:.3f}', xy=(s, d),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=8, color='darkorange')
ax2.set_xlabel('Observation Noise  σ  (price-index units)', fontsize=12)
ax2.set_title('Friction 2: Noisy Monitoring', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.4, 10.5)
ax2.set_xticks(sigma)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)

fig.suptitle(
    'Observation Latency vs Noisy Monitoring — Qualitatively Different Effects on Algorithmic Collusion\n'
    '(Calvano et al. baseline: n=2, m=15, α=0.15, β=4×10⁻⁶, δ=0.95, logit demand, 250 sessions)',
    fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig('friction_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: friction_comparison.png")
