import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Results from A_convResults.txt (avgPrGain column), 250 sessions each
sigma = [0.0,    0.5,     1.0,     2.0,     3.0,     5.0,     10.0]
delta = [0.84280, 0.67049, 0.49269, 0.24244, 0.14261, 0.01326, -0.00258]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(sigma, delta, color='darkorange', linewidth=2,
        marker='s', markersize=7, markerfacecolor='white',
        markeredgecolor='darkorange', markeredgewidth=2)

# Reference lines
ax.axhline(y=delta[0], color='green', linestyle='--', linewidth=1.2,
           label=f'Baseline (σ=0):  Δ = {delta[0]:.3f}')
ax.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2,
           label='Nash (full competition):  Δ = 0')

# Annotate each point
for s, d in zip(sigma, delta):
    ax.annotate(f'{d:.3f}', xy=(s, d),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, color='darkorange')

ax.set_xlabel('Observation Noise  σ  (price-index units, Gaussian std)', fontsize=12)
ax.set_ylabel('Average Profit Gain  Δ', fontsize=12)
ax.set_title('Effect of Noisy Monitoring on Algorithmic Collusion\n'
             '(Calvano et al. baseline: n=2, m=15, α=0.15, β=4×10⁻⁶, δ=0.95, logit demand)',
             fontsize=11)

ax.set_xlim(-0.4, 10.5)
ax.set_ylim(-0.1, 1.0)
ax.set_xticks(sigma)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('noise_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: noise_results.png")
