# Energy Wall: Coherent Analog Computing Simulation

## Abstract

Filamentary memristors (RRAM) are widely considered for analog in‑memory computing, but they suffer from an “energy wall” due to Random Telegraph Noise (RTN), device‑to‑device variability, and relatively high write currents.

This repository contains a **simulation study** of a possible alternative: coherent analog computing using Charge Density Waves (CDW).

In the current toy model, we compare:
- A filamentary, RRAM‑like device with RTN and drift.
- A hypothetical CDW‑based device operated in a resonant, non‑filamentary regime.

The goal is **not** to claim experimental results, but to explore — on simple tasks such as MNIST — how such a coherent regime *might* change the accuracy–vs–energy–vs–noise trade‑off if it can be realized in hardware.

## Key Findings (Simulation Only)

![Efficiency Graph](figures/comparison_chart.png)

Under the assumed device parameters and noise models, the simulations suggest that a resonant CDW‑like regime could:

- Reduce the effective energy per update by **up to ~100×** compared to the filamentary baseline.
- Maintain weight stability compatible with **≳8‑bit equivalent resolution** in a small MNIST classifier.

These numbers are **model‑dependent** and should be interpreted as hypotheses to be tested on real devices, **not** as measured hardware performance.

## Repository Structure

- `sim/`: Core simulation logic (physics models, layers).
- `experiments/`: Training scripts (MNIST, etc.).
- `data/`: Dataset storage (MNIST).
- `figures/`: Generated plots and visualizations.
- `runs/`: CSV logs of simulation runs.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

   python sim/main.py
See experiments/ for more detailed training scripts.

## Citation & License

This repository implements a simple model of a “resonant drive”
protocol for CDW‑based devices, first proposed by Robert Paulig.

Unlike standard purely thermal/voltage switching (e.g. Sci. Rep. 7,
10851), the simulated device is driven by frequency‑matched pulses
to mimic a more coherent, non‑filamentary phase transition and to
suppress stochastic noise in the model.

The code is released under the MIT license.

If you use this simulator or build on the resonant‑drive idea in
academic work, please cite this repository (see CITATION.cff).
