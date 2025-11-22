# energy-wall-cdw-memristor-sim

**Coherent, non‑filamentary analog memory** vs **filamentary RRAM**: simulation of accuracy‑vs‑energy and weight stability.

## Quick start
`ash
python -m venv .venv && . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
python experiments/train_mnist.py --physics RRAM --epochs 1
python experiments/train_mnist.py --physics CDW_COHERENT --epochs 1
`
Outputs go to uns/ and igures/.