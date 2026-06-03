"""Generate every thesis figure from the raw avgPrGain values.

Run from the figures/ directory. Outputs PNG + PDF for each figure.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# -----------------------------------------------------------------------
# Data — copied verbatim from each experiment's A_convResults.txt
# (avgPrGain column, 250 sessions per latency/noise/async cell;
#  factorial replicated at 250/500/1000 sessions)
# -----------------------------------------------------------------------

# Latency sweep (250 sessions per L)
latency_L     = [0,       1,       2,       3,       5,       10,      20]
latency_delta = [0.84280, 0.27957, 0.21490, 0.19281, 0.17365, 0.15786, 0.15503]

# Noise sweep (250 sessions per sigma)
noise_sigma  = [0.0,     0.5,     1.0,     2.0,     3.0,     5.0,     10.0]
noise_delta  = [0.84280, 0.67049, 0.49269, 0.24244, 0.14261, 0.01326, -0.00258]

# Async sweep (250 sessions per T)
async_T      = [0,       1,       2,       5,       10,      20]
async_delta  = [0.84280, 0.37833, 0.41783, 0.42379, 0.41084, 0.40259]

# Factorial: cells in fixed order, three replication sizes
factorial_cells   = ['F000','F100','F010','F001','F110','F101','F011','F111']
factorial_labels  = ['baseline\n(0,0,0)', 'L only\n(2,0,0)', r'$\sigma$ only'+'\n(0,1,0)',
                     'T only\n(0,0,1)',  'L+'+r'$\sigma$'+'\n(2,1,0)', 'L+T\n(2,0,1)',
                     r'$\sigma$+T'+'\n(0,1,1)', 'all three\n(2,1,1)']
factorial_250  = [0.84280, 0.21490, 0.49269, 0.37833, 0.06339, 0.32274, 0.26403, 0.55103]
factorial_500  = [0.84223, 0.21222, 0.49622, 0.38452, 0.06216, 0.34309, 0.27672, 0.53863]
factorial_1000 = [0.84808, 0.20975, 0.49951, 0.38250, 0.05907, 0.34623, 0.27956, 0.53745]
factorial_se_1000 = [0.10924, 0.12228, 0.29796, 0.11070, 0.08014, 0.17392, 0.10299, 0.13462]

# -----------------------------------------------------------------------
# Common style
# -----------------------------------------------------------------------
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def save(fig, name):
    fig.savefig(f'{name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(f'{name}.pdf', bbox_inches='tight')
    print(f'  saved {name}.png and {name}.pdf')


# -----------------------------------------------------------------------
# Figure 1 — Latency sweep
# -----------------------------------------------------------------------
def fig_latency():
    print('Figure 1: latency sweep')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(latency_L, latency_delta, color='steelblue', linewidth=2,
            marker='o', markersize=7, markerfacecolor='white',
            markeredgecolor='steelblue', markeredgewidth=2)
    ax.axhline(latency_delta[0], color='green', linestyle='--', linewidth=1.2,
               label=fr'Baseline ($L=0$): $\Delta = {latency_delta[0]:.3f}$')
    ax.axhline(0.0, color='red', linestyle=':', linewidth=1.2,
               label=r'Nash benchmark: $\Delta = 0$')
    for x, y in zip(latency_L, latency_delta):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='steelblue')
    ax.set_xlabel(r'Observation latency $L$ (periods)')
    ax.set_ylabel(r'Profit gain $\Delta$')
    ax.set_xlim(-0.5, 21); ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(latency_L)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'fig_latency')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 2 — Noise sweep
# -----------------------------------------------------------------------
def fig_noise():
    print('Figure 2: noise sweep')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(noise_sigma, noise_delta, color='darkorange', linewidth=2,
            marker='s', markersize=7, markerfacecolor='white',
            markeredgecolor='darkorange', markeredgewidth=2)
    ax.axhline(noise_delta[0], color='green', linestyle='--', linewidth=1.2,
               label=fr'Baseline ($\sigma=0$): $\Delta = {noise_delta[0]:.3f}$')
    ax.axhline(0.0, color='red', linestyle=':', linewidth=1.2,
               label=r'Nash benchmark: $\Delta = 0$')
    for x, y in zip(noise_sigma, noise_delta):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='darkorange')
    ax.set_xlabel(r'Noise standard deviation $\sigma$ (price-index units)')
    ax.set_ylabel(r'Profit gain $\Delta$')
    ax.set_xlim(-0.4, 10.5); ax.set_ylim(-0.1, 1.0)
    ax.set_xticks(noise_sigma)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'fig_noise')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 3 — Async sweep
# -----------------------------------------------------------------------
def fig_async():
    print('Figure 3: async sweep')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(async_T, async_delta, color='purple', linewidth=2,
            marker='D', markersize=7, markerfacecolor='white',
            markeredgecolor='purple', markeredgewidth=2)
    ax.axhline(async_delta[0], color='green', linestyle='--', linewidth=1.2,
               label=fr'Baseline ($T=0$): $\Delta = {async_delta[0]:.3f}$')
    ax.axhline(0.0, color='red', linestyle=':', linewidth=1.2,
               label=r'Nash benchmark: $\Delta = 0$')
    for x, y in zip(async_T, async_delta):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=9, color='purple')
    ax.set_xlabel(r'Lock-in length $T$ (periods)')
    ax.set_ylabel(r'Profit gain $\Delta$')
    ax.set_xlim(-0.5, 21); ax.set_ylim(-0.05, 1.0)
    ax.set_xticks(async_T)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save(fig, 'fig_async')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 4 — Three-friction comparison (3-panel)
# -----------------------------------------------------------------------
def fig_three_friction_comparison():
    print('Figure 4: three-friction comparison')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    panels = [
        ('Friction 1: Observation latency',  latency_L,    latency_delta,  'steelblue', 'o',
         r'$L$ (periods)'),
        ('Friction 2: Noisy monitoring',     noise_sigma,  noise_delta,    'darkorange', 's',
         r'$\sigma$ (price-index units)'),
        ('Friction 3: Asynchronous actions', async_T,      async_delta,    'purple',     'D',
         r'$T$ (periods)'),
    ]
    for ax, (title, xs, ys, color, marker, xlab) in zip(axes, panels):
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)
        ax.axhline(0.84280, color='green', linestyle='--', linewidth=1.0,
                   label='Baseline')
        ax.axhline(0.0, color='red', linestyle=':', linewidth=1.0,
                   label='Nash')
        for x, y in zip(xs, ys):
            ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 9),
                        textcoords='offset points', ha='center',
                        fontsize=8, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlab); ax.set_xticks(xs)
        ax.set_ylim(-0.1, 1.0); ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    axes[0].set_ylabel(r'Profit gain $\Delta$')
    fig.suptitle(
        'Three frictions, three qualitatively different shapes  '
        '(Calvano et al. baseline; $n=2$, $m=15$; 250 sessions per cell)',
        fontsize=12, y=1.02)
    plt.tight_layout()
    save(fig, 'fig_three_friction_comparison')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 5 — Factorial: observed vs predictions (forest, N=1000)
# -----------------------------------------------------------------------
def fig_factorial_forest():
    print('Figure 5: factorial forest (N=1000)')
    base   = factorial_1000[0]
    L_only = factorial_1000[1]
    s_only = factorial_1000[2]
    T_only = factorial_1000[3]
    rL = L_only / base
    rs = s_only / base
    rT = T_only / base
    dL = base - L_only
    ds = base - s_only
    dT = base - T_only

    def add(*args): return max(0.0, base - sum(args))
    def mul(*ratios):
        v = base
        for r in ratios: v *= r
        return v
    def mn(*deltas): return min(deltas)

    rows = [
        ('F000  baseline',       0,    factorial_1000[0], factorial_se_1000[0], None,             None,                  None),
        ('F100  L only',         1,    factorial_1000[1], factorial_se_1000[1], None,             None,                  None),
        ('F010  $\\sigma$ only', 2,    factorial_1000[2], factorial_se_1000[2], None,             None,                  None),
        ('F001  T only',         3,    factorial_1000[3], factorial_se_1000[3], None,             None,                  None),
        ('F110  L+$\\sigma$',    5,    factorial_1000[4], factorial_se_1000[4], add(dL,ds),       mul(rL,rs),            mn(L_only,s_only)),
        ('F101  L+T',            6,    factorial_1000[5], factorial_se_1000[5], add(dL,dT),       mul(rL,rT),            mn(L_only,T_only)),
        ('F011  $\\sigma$+T',    7,    factorial_1000[6], factorial_se_1000[6], add(ds,dT),       mul(rs,rT),            mn(s_only,T_only)),
        ('F111  all three',      8,    factorial_1000[7], factorial_se_1000[7], add(dL,ds,dT),    mul(rL,rs,rT),         mn(L_only,s_only,T_only)),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_positions = [r[1] for r in rows]
    labels      = [r[0] for r in rows]

    # Observed Δ with cross-session SD as horizontal range bar
    for label, y, obs, sd, addv, mulv, minv in rows:
        ax.errorbar(obs, y, xerr=sd, color='black', fmt='o',
                    markersize=8, markerfacecolor='black',
                    markeredgecolor='black', capsize=4, linewidth=1.6,
                    label='Observed $\\Delta$ (mean $\\pm$ cross-session SD)' if y == 0 else None)
        if addv is not None:
            ax.scatter(addv, y, color='red', marker='v', s=70,
                       label='Additive prediction' if y == 5 else None)
            ax.scatter(mulv, y, color='blue', marker='s', s=55,
                       label='Multiplicative prediction' if y == 5 else None)
            ax.scatter(minv, y, color='green', marker='^', s=70,
                       label='Minimum prediction' if y == 5 else None)

    ax.axvline(base, color='gray', linestyle='--', linewidth=1.0, alpha=0.6,
               label=f'Baseline $\\Delta={base:.3f}$')
    ax.axvline(0.0, color='red', linestyle=':', linewidth=1.0, alpha=0.6,
               label='Nash')

    ax.set_yticks(y_positions); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r'Profit gain $\Delta$')
    ax.set_xlim(-0.1, 1.05)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(
        'Combined-frictions $2{\\times}2{\\times}2$ factorial — observed vs three benchmark predictions  '
        '(N=1000 sessions per cell)', fontsize=12)
    plt.tight_layout()
    save(fig, 'fig_factorial_forest')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 6 — Factorial across N (stability check)
# -----------------------------------------------------------------------
def fig_factorial_acrossN():
    print('Figure 6: factorial Delta across N')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    Ns = [250, 500, 1000]
    cells = factorial_cells
    matrix = np.array([factorial_250, factorial_500, factorial_1000])  # shape (3, 8)
    cmap = plt.cm.tab10
    for i, c in enumerate(cells):
        ys = matrix[:, i]
        ax.plot(Ns, ys, marker='o', linewidth=1.6, color=cmap(i),
                label=f'{c} ({factorial_labels[i].replace(chr(10), " ")})',
                markersize=6)
    ax.set_xlabel('Number of sessions per cell')
    ax.set_ylabel(r'Profit gain $\Delta$')
    ax.set_xticks(Ns)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        'Factorial $\\Delta$ values are stable across replication sizes  '
        '(differences within sampling noise)', fontsize=12)
    ax.legend(loc='center right', bbox_to_anchor=(1.32, 0.5), fontsize=9)
    plt.tight_layout()
    save(fig, 'fig_factorial_acrossN')
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 7 — Grid robustness (Delta vs m, baseline only)
# -----------------------------------------------------------------------
def fig_grid_robustness():
    print('Figure 7: grid robustness (Delta vs m)')
    grid_m_str = ['15', '25', '50', '100']
    grid_delta = [0.848, 0.806, 0.707, 0.706]
    grid_sd    = [0.109, 0.076, 0.023, 0.016]
    x_pos      = list(range(len(grid_m_str)))   # categorical evenly-spaced

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(x_pos, grid_delta, yerr=grid_sd, color='teal', linewidth=2,
                marker='o', markersize=8, markerfacecolor='white',
                markeredgecolor='teal', markeredgewidth=2,
                capsize=4, ecolor='teal', alpha=0.85)
    ax.axhline(y=grid_delta[0], color='green', linestyle='--', linewidth=1.2,
               label=fr'Calvano baseline ($m=15$): $\Delta = {grid_delta[0]:.3f}$')
    ax.axhline(y=0.0, color='red', linestyle=':', linewidth=1.2,
               label=r'Nash benchmark: $\Delta = 0$')

    for x, y in zip(x_pos, grid_delta):
        ax.annotate(f'{y:.3f}', xy=(x, y),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', fontsize=9, color='teal')

    ax.set_xlabel(r'Price-grid size $m$')
    ax.set_ylabel(r'Profit gain $\Delta$')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'$m={m}$' for m in grid_m_str])
    ax.set_xlim(-0.4, len(x_pos) - 0.6)
    ax.set_ylim(-0.05, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(
        r'Grid robustness: $\Delta$ falls modestly with finer grids and asymptotes by $m\sim 50$',
        fontsize=11)

    plt.tight_layout()
    save(fig, 'fig_grid_robustness')
    plt.close(fig)


if __name__ == '__main__':
    fig_latency()
    fig_noise()
    fig_async()
    fig_three_friction_comparison()
    fig_factorial_forest()
    fig_factorial_acrossN()
    fig_grid_robustness()
    print('Done.')
