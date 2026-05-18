"""Mirror the factorial + heatmap cell folders into Code/OffPath/.

For each cell we copy the four input files
    A_InputParameters.txt, A_Latency.txt, A_NoiseStd.txt, A_LockIn.txt
from the existing experiment folders. The off-path-test version of
ogcode_combined.exe will pick them up and additionally write Playback.txt
in each cell.
"""
import pathlib
import shutil

ROOT     = pathlib.Path(__file__).resolve().parents[2]   # ...\Thesis
SRC_FACT = ROOT / 'Code' / 'Combined'                    # 8 factorial cells
SRC_HM   = ROOT / 'Code' / 'Heatmaps'                    # 3 pairs of 25 cells
DST      = ROOT / 'Code' / 'OffPath'

PARAM_FILES = ['A_InputParameters.txt', 'A_Latency.txt',
               'A_NoiseStd.txt',         'A_LockIn.txt']

def copy_cell(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    for fn in PARAM_FILES:
        s = src / fn
        if not s.exists():
            raise FileNotFoundError(f'missing {s}')
        shutil.copyfile(s, dst / fn)

# Factorial corners
factorial_cells = ['F000','F100','F010','F001','F110','F101','F011','F111']
for cell in factorial_cells:
    copy_cell(SRC_FACT / cell, DST / 'Factorial' / cell)
print(f'Copied {len(factorial_cells)} factorial cells -> Factorial/')

# Heatmap interiors
for pair in ['LxN', 'LxT', 'SxT']:
    src_pair = SRC_HM / pair
    n = 0
    for cell_dir in sorted(src_pair.iterdir()):
        if cell_dir.is_dir():
            copy_cell(cell_dir, DST / pair / cell_dir.name)
            n += 1
    print(f'Copied {n} heatmap cells -> {pair}/')

print(f'Total cells set up under Code/OffPath/.')
