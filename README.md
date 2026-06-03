# Robustness of Algorithmic Collusion to Information and Timing Frictions

This repository contains the code, simulation outputs, figures, and LaTeX source for my bachelor's thesis on algorithmic collusion in Q-learning pricing games.

The project starts from the pricing environment of Calvano et al. (2020): two independent tabular Q-learning agents repeatedly choose prices in a differentiated-products Bertrand game. The thesis asks whether more realistic information and timing frictions make algorithmic collusion less concerning, or whether those frictions can interact in ways that preserve collusion.

The three frictions studied are:

- **Observation latency**: agents observe stale rival prices.
- **Noisy monitoring**: agents observe rival prices with Gaussian index noise.
- **Asynchronous price setting**: agents move on staggered pricing schedules rather than simultaneously.

Collusion is measured using the profit-gain index `Delta`, where `Delta = 0` corresponds to competitive Nash profits and `Delta = 1` corresponds to joint-monopoly profits.

## Main Findings

- Single-friction experiments produce three different shapes:
  - latency creates a sharp initial drop,
  - noise creates a smoother decline,
  - asynchrony creates a step down followed by a plateau.
- Latency and noise compound: together they disrupt learned collusion more strongly than a multiplicative benchmark predicts.
- Asynchrony has an opposite-sign interaction: although it reduces collusion on its own, it partially restores collusion when combined with latency or noise.
- Off-path deviation tests show that the high-collusion combined cells are not just passive focal points; the agents still display punishment and recovery dynamics.

The central lesson is that information and timing frictions cannot be evaluated one at a time. Realistic combinations can change the qualitative effect of a policy instrument.

## Repository Structure

```text
.
+-- Code/
|   +-- ogcode/
|   |   +-- cpp conversion/        # C++17 simulator and friction executables
|   +-- Latency/                   # latency sweep outputs and plotting
|   +-- Noise/                     # noise sweep outputs and plotting
|   +-- Async/                     # asynchrony sweep outputs
|   +-- Combined*/                 # 2x2x2 combined-friction runs
|   +-- Heatmaps/                  # pairwise parameter-grid heatmaps
|   +-- OffPath/                   # forced-deviation playback tests
|   +-- Grid Robustness/           # price-grid robustness runs
+-- Thesis_writeup/
|   +-- main.tex                   # LaTeX entry point
|   +-- sections/                  # thesis sections
|   +-- figures/                   # generated figures used in the PDF
|   +-- references.bib
|   +-- main.pdf                   # compiled thesis PDF
+-- Papers used/                   # local copies of papers used while writing
```

## Quick Start

To read the final thesis, open:

```text
Thesis_writeup/main.pdf
```

To inspect the main C++ implementation, start with:

```text
Code/ogcode/cpp conversion/src/main_combined.cpp
Code/ogcode/cpp conversion/src/LearningSimulationCombined.cpp
```

The generated figures used in the thesis are in:

```text
Thesis_writeup/figures/
```

## Reproducing the Thesis PDF

The thesis is built with LaTeX/BibTeX. From `Thesis_writeup/`:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled output is `Thesis_writeup/main.pdf`.

## Rebuilding Figures

Most figures are generated from already-computed text outputs. From the relevant folders:

```powershell
python Thesis_writeup/figures/make_all_figures.py
python Code/Heatmaps/make_heatmaps.py
python Code/OffPath/make_offpath_figure.py
```

After regenerating heatmap or off-path figures, copy the updated PDFs/PNGs into `Thesis_writeup/figures/` if you want the thesis PDF to use them.

## Building the C++ Simulator

The simulator is in `Code/ogcode/cpp conversion/` and uses C++17. It defines five executables:

- `ogcode_cpp`: baseline Calvano-style learner
- `ogcode_latency`: observation-latency friction
- `ogcode_noise`: noisy-monitoring friction
- `ogcode_async`: asynchronous-pricing friction
- `ogcode_combined`: unified combined-friction learner

With CMake:

```powershell
cd "Code/ogcode/cpp conversion"
cmake -S . -B build
cmake --build build --config Release
```

On the original Windows setup, I also used the included direct `g++` build script:

```powershell
cd "Code/ogcode/cpp conversion"
.\build_frictions.bat
```

That script expects MSYS2 `g++` at `C:\msys64\ucrt64\bin\g++.exe`; adjust the `GXX` path if your compiler is elsewhere.

## Reproducibility Notes

The repository includes the simulation outputs used to create the thesis figures. Re-running all simulations from scratch is possible but computationally expensive, especially the combined-friction and off-path playback runs.

The key design choices are:

- baseline pricing environment follows Calvano et al. (2020),
- memory is fixed at one previous-period joint price state,
- price grid uses 15 discrete prices,
- frictions preserve the baseline state size so that results remain comparable,
- combined-friction cells are replicated at `N = 250`, `500`, and `1000` sessions.

See the methodology and appendix in `Thesis_writeup/main.pdf` for the exact economic setup and implementation details.

## Requirements

For figure generation:

- Python 3
- `numpy`
- `pandas`
- `matplotlib`

For the simulator:

- C++17 compiler
- OpenMP support recommended
- CMake 3.16+ if using the CMake build path

For the thesis PDF:

- LaTeX distribution such as MiKTeX or TeX Live
- BibTeX

## Thesis Status

This repository corresponds to the final thesis write-up and supervisor-feedback revisions. The current compiled PDF is:

```text
Thesis_writeup/main.pdf
```

## Citation

If you use this repository or build on it, please cite the thesis/repository and the original Calvano et al. (2020) paper:

> Calvano, E., Calzolari, G., Denicolo, V., & Pastorello, S. (2020). Artificial Intelligence, Algorithmic Pricing, and Collusion. *American Economic Review*, 110(10), 3267-3297.
