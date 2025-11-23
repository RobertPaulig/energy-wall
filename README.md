# Energy Wall: Coherent Analog Computing Simulation

## Abstract

Filamentary memristors (RRAM) are widely considered for analog in‑memory
computing, but they suffer from an “energy wall” due to Random Telegraph
Noise (RTN), device‑to‑device variability and relatively high write
currents.

This repository contains a **simulation study** of a possible alternative:
coherent analog computing using charge density waves (CDW).

In the current toy model we compare:

- a filamentary, RRAM‑like device with RTN and drift;
- a hypothetical CDW‑based device operated in a resonant, non‑filamentary regime.

The goal is **not** to claim experimental results, but to explore — on
simple tasks such as MNIST — how such a coherent regime *might* change
the accuracy–vs–energy–vs–noise trade‑off if it can be realized in
hardware.

## Key observations (simulation only)

![Efficiency Graph](figures/comparison_chart.png)

Under the assumed device parameters and noise models, the simulations
suggest that a resonant CDW‑like regime could:

- reduce the effective energy per update by **up to ~100×** compared to
  the filamentary baseline;
- maintain weight stability compatible with **≳8‑bit equivalent
  resolution** in a small MNIST classifier.

These numbers are **model‑dependent** and should be interpreted as
hypotheses to be tested on real devices, **not** as measured hardware
performance.

## Repository Structure

- `sim/`: core simulation logic (physics models, layers).
- `experiments/`: training scripts (MNIST, etc.).
- `data/`: dataset storage (MNIST).
- `figures/`: generated plots and visualizations.
- `runs/`: CSV logs of simulation runs.

## Usage

To reproduce the basic comparison plot (RRAM vs CDW‑like device):

```bash
python sim/main.py
