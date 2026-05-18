"""Render the off-path deviation test figure.

Two outputs:

  fig_offpath_factorial.{png,pdf}
      2x4 grid of panels, one per factorial cell. Each panel shows the
      mean Delta trajectory across the 30 playback periods, separately
      for the two deviation types (Low, BR), with the pre-shock steady
      state as a horizontal reference.

  fig_offpath_retal_heatmaps.{png,pdf}
      3x2 grid. Rows = pairs (LxN, LxT, SxT). Columns = retaliation depth
      (delta_pre - delta_min) for each deviation type. Each heatmap is a
      5x5 grid of cells across the heatmap parameter pair.
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from analyze_playback import (load_playback, mean_trajectory,
                              retaliation_stats, DEV_LABELS)

HERE = pathlib.Path(__file__).resolve().parent

# Grid (must match setup_heatmap_folders.py)
L_VALUES = [0, 1, 2, 3, 5]
S_VALUES = [0, 0.5, 1, 2, 3]
T_VALUES = [0, 1, 2, 5, 10]


def fmt_L(L): return f'L{L}'
def fmt_T(T): return f'T{T}'
def fmt_S(s): return 'S' + (str(int(s)) if float(s).is_integer() else str(s)).replace('.', 'p')


# -----------------------------------------------------------------------
# Figure 1: factorial 2x4 grid of Delta trajectories
# -----------------------------------------------------------------------
FACTORIAL_ORDER = ['F000', 'F100', 'F010', 'F001',
                   'F110', 'F101', 'F011', 'F111']
FACTORIAL_LABELS = {
    'F000': 'F000 baseline (no friction)',
    'F100': 'F100 latency only (L=2)',
    'F010': 'F010 noise only (\\sigma=1)',
    'F001': 'F001 async only (T=1)',
    'F110': 'F110 latency + noise',
    'F101': 'F101 latency + async',
    'F011': 'F011 noise + async',
    'F111': 'F111 all three',
}


def make_factorial_figure():
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    plt.rcParams.update({'font.size': 10})

    for i, cell in enumerate(FACTORIAL_ORDER):
        ax = axes[i // 4, i % 4]
        folder = HERE / 'Factorial' / cell
        df = load_playback(folder)
        if df is None:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            ax.set_title(FACTORIAL_LABELS[cell], fontsize=10)
            continue

        traj = mean_trajectory(df)
        delta_pre = traj.loc[0].mean()  # avg of two devTypes at period 0

        # Plot both deviation curves
        ax.plot(traj.index, traj['Low'], color='tab:blue',
                linewidth=1.8, marker='o', markersize=3.5, label='Low')
        ax.plot(traj.index, traj['BR'],  color='tab:orange',
                linewidth=1.8, marker='s', markersize=3.5, label='BR')
        ax.axhline(delta_pre, color='gray', linestyle='--', linewidth=1.0,
                   alpha=0.7, label='steady state')
        ax.axhline(0.0, color='red', linestyle=':', linewidth=0.9, alpha=0.6)

        # Adapt x-limits to whatever playback length each cell has
        max_period = int(traj.index.max())
        ax.set_xlim(-1, max_period + 1)
        ax.set_ylim(-0.1, 1.0)
        ax.set_title(FACTORIAL_LABELS[cell].replace('\\sigma', r'$\sigma$'),
                     fontsize=10)
        ax.set_xlabel('Period after deviation')
        if i % 4 == 0:
            ax.set_ylabel(r'Mean $\Delta$')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(
        'Off-path deviation test: $\\Delta$ trajectory over 30 periods after '
        'a forced agent-1 deviation\n'
        'Two deviation types (Low = lowest grid price, BR = static-Bertrand best response). '
        'Retaliation looks like a temporary $\\Delta$ dip and recovery.',
        fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_offpath_factorial.png', dpi=150, bbox_inches='tight')
    fig.savefig(HERE / 'fig_offpath_factorial.pdf', bbox_inches='tight')
    print('Saved fig_offpath_factorial.png and .pdf')


# -----------------------------------------------------------------------
# Figure 2: retaliation-depth heatmaps for the three pairs
# -----------------------------------------------------------------------
def _retal_grid(pair_folder, row_vals, col_vals, fmt_r, fmt_c):
    """Return (depth_Low, depth_BR) 2D arrays for each pair."""
    nrows, ncols = len(row_vals), len(col_vals)
    depth_low = np.full((nrows, ncols), np.nan)
    depth_br  = np.full((nrows, ncols), np.nan)
    for i, r in enumerate(row_vals):
        for j, c in enumerate(col_vals):
            folder = pair_folder / f'{fmt_r(r)}_{fmt_c(c)}'
            df = load_playback(folder)
            if df is None:
                continue
            stats = retaliation_stats(df)
            if 'Low' in stats: depth_low[i, j] = stats['Low']['retal_depth']
            if 'BR'  in stats: depth_br[i, j]  = stats['BR']['retal_depth']
    return depth_low, depth_br


def _heatmap(ax, mat, row_labels, col_labels, row_name, col_name, title):
    finite = mat[np.isfinite(mat)]
    vmax = max(0.05, np.nanmax(np.abs(finite))) if len(finite) else 0.3
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = '--' if np.isnan(v) else f'{v:.2f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='black')
    return im


def make_retal_heatmap_figure():
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))
    plt.rcParams.update({'font.size': 10})

    # Row 1: Latency x Noise
    low, br = _retal_grid(HERE / 'LxN', L_VALUES, S_VALUES, fmt_L, fmt_S)
    _heatmap(axes[0,0], low,
             [f'L={v}' for v in L_VALUES], [f'$\\sigma$={v}' for v in S_VALUES],
             'Latency $L$', 'Noise $\\sigma$',
             'L x $\\sigma$: retaliation depth (Low deviation)')
    _heatmap(axes[0,1], br,
             [f'L={v}' for v in L_VALUES], [f'$\\sigma$={v}' for v in S_VALUES],
             'Latency $L$', 'Noise $\\sigma$',
             'L x $\\sigma$: retaliation depth (BR deviation)')

    # Row 2: Latency x Async
    low, br = _retal_grid(HERE / 'LxT', L_VALUES, T_VALUES, fmt_L, fmt_T)
    _heatmap(axes[1,0], low,
             [f'L={v}' for v in L_VALUES], [f'T={v}' for v in T_VALUES],
             'Latency $L$', 'Async $T$',
             'L x T: retaliation depth (Low)')
    _heatmap(axes[1,1], br,
             [f'L={v}' for v in L_VALUES], [f'T={v}' for v in T_VALUES],
             'Latency $L$', 'Async $T$',
             'L x T: retaliation depth (BR)')

    # Row 3: Noise x Async
    low, br = _retal_grid(HERE / 'SxT', S_VALUES, T_VALUES, fmt_S, fmt_T)
    _heatmap(axes[2,0], low,
             [f'$\\sigma$={v}' for v in S_VALUES], [f'T={v}' for v in T_VALUES],
             'Noise $\\sigma$', 'Async $T$',
             '$\\sigma$ x T: retaliation depth (Low)')
    _heatmap(axes[2,1], br,
             [f'$\\sigma$={v}' for v in S_VALUES], [f'T={v}' for v in T_VALUES],
             'Noise $\\sigma$', 'Async $T$',
             '$\\sigma$ x T: retaliation depth (BR)')

    fig.suptitle(
        'Retaliation depth across the heatmap interiors\n'
        'Depth = (steady-state $\\Delta$) $-$ (minimum $\\Delta$ during periods 1-30).\n'
        'Darker = stronger retaliation = stronger evidence of genuine reward-punishment collusion.',
        fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_offpath_retal_heatmaps.png', dpi=150, bbox_inches='tight')
    fig.savefig(HERE / 'fig_offpath_retal_heatmaps.pdf', bbox_inches='tight')
    print('Saved fig_offpath_retal_heatmaps.png and .pdf')


if __name__ == '__main__':
    make_factorial_figure()
    make_retal_heatmap_figure()
