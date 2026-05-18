"""Build the 75-cell folder structure for the pairwise heatmap sweeps.

Three pairs, 5x5 each:
    LxN  --  vary L, sigma; T = 0
    LxT  --  vary L, T;     sigma = 0
    SxT  --  vary sigma, T; L = 0

Each cell folder receives:
    A_InputParameters.txt  -- copied from Code/Latency/L0/ (the canonical
                              250-session input file)
    A_Latency.txt          -- one integer  L
    A_NoiseStd.txt         -- one real     sigma
    A_LockIn.txt           -- one integer  T

Run from anywhere; paths are resolved relative to the project root.
"""
import pathlib
import shutil

# --- experiment grid -----------------------------------------------------
L_VALUES = [0, 1, 2, 3, 5]
S_VALUES = [0, 0.5, 1, 2, 3]
T_VALUES = [0, 1, 2, 5, 10]

# --- paths ---------------------------------------------------------------
ROOT     = pathlib.Path(__file__).resolve().parents[2]   # ...\Thesis
HEATMAPS = ROOT / 'Code' / 'Heatmaps'
TEMPLATE = ROOT / 'Code' / 'Latency' / 'L0' / 'A_InputParameters.txt'

assert TEMPLATE.exists(), f'Template A_InputParameters.txt not found at {TEMPLATE}'


# --- naming --------------------------------------------------------------
def fmt_L(L):  return f'L{L}'
def fmt_T(T):  return f'T{T}'
def fmt_S(s):  return 'S' + (str(int(s)) if float(s).is_integer() else str(s)).replace('.', 'p')


# --- writers -------------------------------------------------------------
def write_cell(folder: pathlib.Path, L, sigma, T):
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE, folder / 'A_InputParameters.txt')
    (folder / 'A_Latency.txt').write_text(f'{L}\n')
    (folder / 'A_NoiseStd.txt').write_text(f'{sigma}\n')
    (folder / 'A_LockIn.txt').write_text(f'{T}\n')


# --- generate ------------------------------------------------------------
cells_LxN = []
for L in L_VALUES:
    for s in S_VALUES:
        name = f'{fmt_L(L)}_{fmt_S(s)}'
        write_cell(HEATMAPS / 'LxN' / name, L, s, 0)
        cells_LxN.append(name)

cells_LxT = []
for L in L_VALUES:
    for T in T_VALUES:
        name = f'{fmt_L(L)}_{fmt_T(T)}'
        write_cell(HEATMAPS / 'LxT' / name, L, 0, T)
        cells_LxT.append(name)

cells_SxT = []
for s in S_VALUES:
    for T in T_VALUES:
        name = f'{fmt_S(s)}_{fmt_T(T)}'
        write_cell(HEATMAPS / 'SxT' / name, 0, s, T)
        cells_SxT.append(name)

print(f'Created {len(cells_LxN)} cells under LxN/')
print(f'Created {len(cells_LxT)} cells under LxT/')
print(f'Created {len(cells_SxT)} cells under SxT/')
print(f'Total: {len(cells_LxN) + len(cells_LxT) + len(cells_SxT)} cells')

# Also emit a manifest for the .bat runners
with open(HEATMAPS / 'cells_LxN.txt', 'w') as f: f.write(' '.join(cells_LxN))
with open(HEATMAPS / 'cells_LxT.txt', 'w') as f: f.write(' '.join(cells_LxT))
with open(HEATMAPS / 'cells_SxT.txt', 'w') as f: f.write(' '.join(cells_SxT))
print('Wrote cell name manifests for the batch runners.')
