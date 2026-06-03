"""Render the three pairwise heatmaps after the simulations have finished.

For each pair (L x sigma, L x T, sigma x T) produces two panels:
    Left:  observed Delta heatmap (YlGnBu)
    Right: Delta minus multiplicative-benchmark prediction (RdBu divergent)

Output:
    fig_heatmaps.png / fig_heatmaps.pdf  (3 rows x 2 cols)

The multiplicative benchmark for a cell at (level_a, level_b) is:
    Delta_pred = Delta(0,0) * (Delta(a,0)/Delta(0,0)) * (Delta(0,b)/Delta(0,0))
                = Delta(a,0) * Delta(0,b) / Delta(0,0)
i.e. the product of single-friction ratios applied to baseline.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# -----------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------
L_VALUES = [0, 1, 2, 3, 5]
S_VALUES = [0, 0.5, 1, 2, 3]
T_VALUES = [0, 1, 2, 5, 10]

HEATMAPS = pathlib.Path(__file__).resolve().parent

# Cell naming functions (must match setup_heatmap_folders.py exactly)
def fmt_L(L): return f'L{L}'
def fmt_T(T): return f'T{T}'
def fmt_S(s): return 'S' + (str(int(s)) if float(s).is_integer() else str(s)).replace('.', 'p')

# -----------------------------------------------------------------------
# Helper to read avgPrGain from a cell's A_convResults.txt
# -----------------------------------------------------------------------
def read_delta(folder: pathlib.Path):
    f = folder / 'A_convResults.txt'
    if not f.exists():
        return np.nan
    lines = f.read_text().splitlines()
    if len(lines) < 2:
        return np.nan
    header = lines[0].split()
    data   = lines[1].split()
    try:
        i = header.index('avgPrGain')
        return float(data[i])
    except (ValueError, IndexError):
        return np.nan


def load_grid(pair: str, axis_a, axis_b, fmt_a, fmt_b):
    """Load a 2D Delta grid for `pair` (folder name) with axis_a on rows,
    axis_b on cols. Returns (Delta_obs, Delta_pred) arrays of shape
    (len(axis_a), len(axis_b))."""
    obs = np.full((len(axis_a), len(axis_b)), np.nan)
    for i, a in enumerate(axis_a):
        for j, b in enumerate(axis_b):
            folder = HEATMAPS / pair / f'{fmt_a(a)}_{fmt_b(b)}'
            obs[i, j] = read_delta(folder)

    # Multiplicative benchmark from the axes (zero-row / zero-column).
    delta0 = obs[0, 0]
    pred = np.full_like(obs, np.nan)
    for i in range(len(axis_a)):
        for j in range(len(axis_b)):
            di = obs[i, 0]    # single-friction A at level axis_a[i]
            dj = obs[0, j]    # single-friction B at level axis_b[j]
            if not (np.isnan(di) or np.isnan(dj) or delta0 == 0):
                pred[i, j] = di * dj / delta0
    return obs, pred


# -----------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------
def _text_color_for_value(value, cmap, norm):
    if np.isnan(value):
        return 'black'
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'white' if luminance < 0.48 else 'black'


def annotate(ax, mat, fmt='{:.2f}', size=8, cmap=None, norm=None):
    nrows, ncols = mat.shape
    for i in range(nrows):
        for j in range(ncols):
            v = mat[i, j]
            if np.isnan(v):
                txt = '--'
            else:
                txt = fmt.format(v)
            color = _text_color_for_value(v, cmap, norm) if cmap is not None and norm is not None else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=size, color=color)


def add_cell_grid(ax, nrows, ncols):
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.7, alpha=0.65)
    ax.tick_params(which='minor', bottom=False, left=False)


def heatmap_obs(ax, mat, row_labels, col_labels, row_name, col_name, title):
    cmap = plt.get_cmap('YlGnBu')
    norm = mcolors.Normalize(vmin=-0.05, vmax=1.0)
    im = ax.imshow(mat, cmap=cmap, aspect='auto', norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title, fontsize=11)
    add_cell_grid(ax, mat.shape[0], mat.shape[1])
    annotate(ax, mat, cmap=cmap, norm=norm)
    return im


def heatmap_dev(ax, dev_mat, row_labels, col_labels, row_name, col_name, title):
    # Symmetric divergent colormap centred at 0.
    finite = dev_mat[np.isfinite(dev_mat)]
    if len(finite) == 0:
        vmax = 0.3
    else:
        vmax = max(0.05, np.nanmax(np.abs(finite)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap('RdBu')
    im = ax.imshow(dev_mat, cmap=cmap, aspect='auto', norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title, fontsize=11)
    add_cell_grid(ax, dev_mat.shape[0], dev_mat.shape[1])
    annotate(ax, dev_mat, fmt='{:+.2f}', cmap=cmap, norm=norm)
    return im


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))
    plt.rcParams.update({'font.size': 10})

    # Row 1: Latency x Noise
    obs, pred = load_grid('LxN', L_VALUES, S_VALUES, fmt_L, fmt_S)
    dev = obs - pred
    im1 = heatmap_obs(axes[0, 0], obs,
                     [f'L={v}' for v in L_VALUES], [f'$\\sigma$={v}' for v in S_VALUES],
                     'Latency $L$', 'Noise $\\sigma$',
                     'Latency x Noise: observed $\\Delta$')
    im2 = heatmap_dev(axes[0, 1], dev,
                     [f'L={v}' for v in L_VALUES], [f'$\\sigma$={v}' for v in S_VALUES],
                     'Latency $L$', 'Noise $\\sigma$',
                     'Observed $-$ multiplicative prediction\n(red = below prediction, blue = above)')

    # Row 2: Latency x Async
    obs, pred = load_grid('LxT', L_VALUES, T_VALUES, fmt_L, fmt_T)
    dev = obs - pred
    heatmap_obs(axes[1, 0], obs,
               [f'L={v}' for v in L_VALUES], [f'T={v}' for v in T_VALUES],
               'Latency $L$', 'Async $T$',
               'Latency x Async: observed $\\Delta$')
    heatmap_dev(axes[1, 1], dev,
               [f'L={v}' for v in L_VALUES], [f'T={v}' for v in T_VALUES],
               'Latency $L$', 'Async $T$',
               'Observed $-$ multiplicative prediction')

    # Row 3: Noise x Async
    obs, pred = load_grid('SxT', S_VALUES, T_VALUES, fmt_S, fmt_T)
    dev = obs - pred
    heatmap_obs(axes[2, 0], obs,
               [f'$\\sigma$={v}' for v in S_VALUES], [f'T={v}' for v in T_VALUES],
               'Noise $\\sigma$', 'Async $T$',
               'Noise x Async: observed $\\Delta$')
    heatmap_dev(axes[2, 1], dev,
               [f'$\\sigma$={v}' for v in S_VALUES], [f'T={v}' for v in T_VALUES],
               'Noise $\\sigma$', 'Async $T$',
               'Observed $-$ multiplicative prediction')

    # Shared colorbars (one for the left column, one for the right)
    cbar_ax_left  = fig.add_axes([0.05, 0.04, 0.42, 0.012])
    cbar_ax_right = fig.add_axes([0.55, 0.04, 0.42, 0.012])
    plt.colorbar(im1, cax=cbar_ax_left,  orientation='horizontal', label='Observed $\\Delta$')
    plt.colorbar(im2, cax=cbar_ax_right, orientation='horizontal', label='Deviation from multiplicative prediction')

    plt.subplots_adjust(left=0.07, right=0.97, top=0.96, bottom=0.10, hspace=0.45, wspace=0.20)
    fig.savefig(HEATMAPS / 'fig_heatmaps.png', dpi=150, bbox_inches='tight')
    fig.savefig(HEATMAPS / 'fig_heatmaps.pdf', bbox_inches='tight')
    print('Saved fig_heatmaps.png and fig_heatmaps.pdf')


if __name__ == '__main__':
    main()
