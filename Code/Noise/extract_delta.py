"""Extract avgPrGain (Delta) from A_convResults.txt for each Noise folder."""
import os
from pathlib import Path

root = Path(__file__).parent
folders = [('N0', 0.0), ('N0p5', 0.5), ('N1', 1.0), ('N2', 2.0),
           ('N3', 3.0), ('N5', 5.0), ('N10', 10.0)]

rows = []
for name, sigma in folders:
    path = root / name / 'A_convResults.txt'
    if not path.exists():
        print(f'{name:>6}  MISSING')
        continue
    with path.open() as f:
        header = f.readline().split()
        lines = [ln.split() for ln in f if ln.strip()]
    try:
        idx = header.index('avgPrGain')
        idx_se = header.index('sePrGain')
    except ValueError:
        print(f'{name:>6}  no avgPrGain column')
        continue
    # Take the FIRST data row (aggregated across sessions)
    row = lines[0]
    delta = float(row[idx])
    se    = float(row[idx_se])
    rows.append((sigma, delta, se))
    print(f'sigma={sigma:>5}  Delta={delta:.5f}  SE={se:.5f}')

print()
print('sigmas =', [r[0] for r in rows])
print('deltas =', [round(r[1], 5) for r in rows])
