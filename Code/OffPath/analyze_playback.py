"""Analysis utilities for the off-path deviation tests.

Each cell folder contains Playback.txt with columns:
    session  devType  period  p1Idx  p2Idx

devType is 0 for the "Low" deviation (force lowest grid index) and 1 for
the "BR" deviation (static-Bertrand best response against agent 2's true
price). Period 0 is the steady state pre-shock; periods 1..30 are post-
shock greedy playback.

This module converts indices into actual prices using the Calvano price
grid, computes the symmetric profit gain Delta per period, and exposes
helpers for aggregation across sessions.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

# Calvano baseline demand / cost parameters
A0, A1, A2, MU = 0.0, 2.0, 2.0, 0.25
C1, C2         = 1.0, 1.0

# Calvano baseline price grid (m=15, extended 10% beyond Nash/Coop)
PRICE_GRID = np.array([
    1.4277250, 1.4664721, 1.5052193, 1.5439664, 1.5827136,
    1.6214607, 1.6602079, 1.6989550, 1.7377021, 1.7764493,
    1.8151964, 1.8539436, 1.8926907, 1.9314379, 1.9701850
])
NASH_PRICE = 1.47293
COOP_PRICE = 1.92498

DEV_LABELS = {0: 'Low', 1: 'BR'}


def _profits(p1: np.ndarray, p2: np.ndarray):
    e0 = np.exp(A0 / MU)
    e1 = np.exp((A1 - p1) / MU)
    e2 = np.exp((A2 - p2) / MU)
    total = e0 + e1 + e2
    q1 = e1 / total
    q2 = e2 / total
    return (p1 - C1) * q1, (p2 - C2) * q2


def _avg_profit_scalar(p1: float, p2: float) -> float:
    pi1, pi2 = _profits(np.array([p1]), np.array([p2]))
    return float((pi1 + pi2) / 2)[0] if pi1.shape else float((pi1 + pi2) / 2)

# Use exact computation for scalar benchmarks
_pi1_n, _pi2_n = _profits(np.array([NASH_PRICE]), np.array([NASH_PRICE]))
NASH_AVG_PI    = float((_pi1_n + _pi2_n)[0] / 2)
_pi1_c, _pi2_c = _profits(np.array([COOP_PRICE]), np.array([COOP_PRICE]))
COOP_AVG_PI    = float((_pi1_c + _pi2_c)[0] / 2)


def delta_from_indices(p1_idx: np.ndarray, p2_idx: np.ndarray) -> np.ndarray:
    """Vectorised Delta for arrays of price indices (1-based)."""
    p1 = PRICE_GRID[p1_idx - 1]
    p2 = PRICE_GRID[p2_idx - 1]
    pi1, pi2 = _profits(p1, p2)
    avg = (pi1 + pi2) / 2
    return (avg - NASH_AVG_PI) / (COOP_AVG_PI - NASH_AVG_PI)


def load_playback(folder: pathlib.Path) -> pd.DataFrame | None:
    """Load Playback.txt and append a Delta column. Returns None if missing."""
    f = folder / 'Playback.txt'
    if not f.exists():
        return None
    df = pd.read_csv(f, sep=' ', header=0)
    df['Delta'] = delta_from_indices(df['p1Idx'].values, df['p2Idx'].values)
    return df


def mean_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame: index = period (0..30), columns = ['Low', 'BR'],
    values = mean Delta across sessions."""
    g = df.groupby(['devType', 'period'])['Delta'].mean().unstack(level=0)
    return g.rename(columns=DEV_LABELS)


def retaliation_stats(df: pd.DataFrame) -> dict:
    """Per-cell summary statistics.

    Works with any playback length: post-shock window is inferred from the
    maximum period present in the data. Recovery time = first period >= 2
    at which mean Delta is within 0.05 of pre-shock Delta. -1 if never
    recovers within the available window.
    """
    out = {}
    max_period = int(df['period'].max())
    for dt, label in DEV_LABELS.items():
        sub = df[df['devType'] == dt]
        traj_delta = sub.groupby('period')['Delta'].mean()
        traj_p2    = sub.groupby('period')['p2Idx'].mean()
        if 0 not in traj_delta.index:
            continue
        delta_pre = traj_delta.loc[0]
        post = traj_delta.loc[1:max_period]
        delta_min = post.min()
        rec = -1
        for t in range(2, max_period + 1):
            if t in traj_delta.index and abs(traj_delta.loc[t] - delta_pre) < 0.05:
                rec = t
                break
        out[label] = {
            'delta_pre':   float(delta_pre),
            'delta_min':   float(delta_min),
            'retal_depth': float(delta_pre - delta_min),
            'p2_pre':      float(traj_p2.loc[0]),
            'p2_min':      float(traj_p2.loc[1:max_period].min()),
            'p2_drop':     float(traj_p2.loc[0] - traj_p2.loc[1:max_period].min()),
            'recover_t':   int(rec),
            'window':      max_period,
        }
    return out


# Convenience for command-line poking
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python analyze_playback.py <cell-folder>')
        sys.exit(1)
    folder = pathlib.Path(sys.argv[1])
    df = load_playback(folder)
    if df is None:
        print(f'No Playback.txt in {folder}')
        sys.exit(1)
    print(f'Loaded {len(df)} rows from {folder}')
    print()
    print('Mean Delta trajectory:')
    print(mean_trajectory(df).round(3))
    print()
    print('Summary stats:')
    for label, stats in retaliation_stats(df).items():
        print(f'  {label}:', {k: round(v, 3) if isinstance(v, float) else v
                              for k, v in stats.items()})
