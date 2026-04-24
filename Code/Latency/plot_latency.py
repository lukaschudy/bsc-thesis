import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Results from A_convResults.txt (avgPrGain column), 250 sessions each
latency = [0,      1,       2,       3,       5,       10,      20]
delta   = [0.84280, 0.27957, 0.21490, 0.19281, 0.17365, 0.15786, 0.15503]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(latency, delta, color='steelblue', linewidth=2,
        marker='o', markersize=7, markerfacecolor='white',
        markeredgecolor='steelblue', markeredgewidth=2)

# Reference lines
ax.axhline(y=delta[0], color='green', linestyle='--', linewidth=1.2,
           label=f'Baseline (L=0):  Δ = {delta[0]:.3f}')
ax.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2,
           label='Nash (full competition):  Δ = 0')

# Annotate each point
for l, d in zip(latency, delta):
    ax.annotate(f'{d:.3f}', xy=(l, d),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, color='steelblue')

ax.set_xlabel('Observation Latency  L  (periods)', fontsize=12)
ax.set_ylabel('Average Profit Gain  Δ', fontsize=12)
ax.set_title('Effect of Observation Latency on Algorithmic Collusion\n'
             '(Calvano et al. baseline: n=2, m=15, α=0.15, β=4×10⁻⁶, δ=0.95, logit demand)',
             fontsize=11)

ax.set_xlim(-0.5, 21)
ax.set_ylim(-0.05, 1.0)
ax.set_xticks(latency)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latency_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: latency_results.png")
