"""Smart resume of the off-path sweep.

Walks every cell folder under Code/OffPath/{Factorial,LxN,LxT,SxT}, decides
whether the cell is COMPLETE / INCOMPLETE / NOT-STARTED based on the
presence and row count of Playback.txt, and re-runs only the unfinished
cells with ogcode_combined.exe.

A cell is considered COMPLETE if:
  - Playback.txt exists, AND
  - It has exactly 250 * 2 * 31 + 1 = 15501 lines
    (250 sessions x 2 deviation types x 31 logged periods, plus header).

Run from anywhere:
    python resume_offpath.py            # check + run all unfinished
    python resume_offpath.py --dry-run  # only check, don't run
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]   # ...\Thesis
HERE = ROOT / 'Code' / 'OffPath'
EXE  = ROOT / 'Code' / 'ogcode' / 'cpp conversion' / 'ogcode_combined.exe'

EXPECTED_LINES = 250 * 2 * 31 + 1   # 15501 including header


def cell_status(folder: pathlib.Path) -> str:
    """Return one of: COMPLETE, INCOMPLETE, NOT_STARTED."""
    pb = folder / 'Playback.txt'
    if not pb.exists():
        return 'NOT_STARTED'
    try:
        with open(pb, 'rb') as f:
            n = sum(1 for _ in f)
    except Exception:
        return 'INCOMPLETE'
    if n == EXPECTED_LINES:
        return 'COMPLETE'
    return 'INCOMPLETE'


def all_cells():
    """List every cell folder under OffPath/{Factorial,LxN,LxT,SxT}."""
    cells = []
    for sub in ('Factorial', 'LxN', 'LxT', 'SxT'):
        d = HERE / sub
        if not d.exists():
            continue
        for cell_dir in sorted(d.iterdir()):
            if cell_dir.is_dir() and (cell_dir / 'A_InputParameters.txt').exists():
                cells.append(cell_dir)
    return cells


def run_one(folder: pathlib.Path):
    """Invoke ogcode_combined.exe in `folder` with stdout captured to run.log."""
    log = folder / 'run.log'
    with open(log, 'w') as f:
        return subprocess.run([str(EXE)], cwd=str(folder), stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='only check status, do not run any cells')
    args = ap.parse_args()

    cells = all_cells()
    if not cells:
        print(f'No cells found under {HERE}.')
        return

    by_status = {'COMPLETE': [], 'INCOMPLETE': [], 'NOT_STARTED': []}
    for c in cells:
        by_status[cell_status(c)].append(c)

    print('=' * 60)
    print(f'Cells scanned:           {len(cells)}')
    print(f'  COMPLETE              {len(by_status["COMPLETE"])}')
    print(f'  INCOMPLETE            {len(by_status["INCOMPLETE"])}')
    print(f'  NOT_STARTED           {len(by_status["NOT_STARTED"])}')
    print('=' * 60)

    to_run = by_status['INCOMPLETE'] + by_status['NOT_STARTED']
    if not to_run:
        print('Nothing to do, all cells complete.')
        return

    print(f'\nCells to run ({len(to_run)}):')
    for c in to_run:
        print(f'  {c.relative_to(HERE)}')

    if args.dry_run:
        print('\n[dry-run] Not actually running anything.')
        return

    if not EXE.exists():
        print(f'\nERROR: ogcode_combined.exe not found at {EXE}')
        sys.exit(1)

    print()
    t0 = time.time()
    for i, c in enumerate(to_run, 1):
        ts = time.strftime('%H:%M:%S')
        elapsed = (time.time() - t0) / 60
        print(f'[{ts}]  ({i}/{len(to_run)}, +{elapsed:.1f}min)  '
              f'running {c.relative_to(HERE)} ...', flush=True)
        run_one(c)
        # Re-check status; if still INCOMPLETE, flag it
        st = cell_status(c)
        if st != 'COMPLETE':
            print(f'   WARNING: cell finished with status {st}. Check {c}/run.log.')

    total = (time.time() - t0) / 60
    print(f'\nAll done in {total:.1f} minutes.')
    print(f'Next step:  python make_offpath_figure.py')


if __name__ == '__main__':
    main()
