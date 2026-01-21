# ORFT — 1+1D Corridor-Locked Phase Field Simulation 

This folder contains a minimal, reproducible simulation that demonstrates the ORFT-style behavior described in the ORFT whitepaper appendix:
- convergence into a narrow coherence corridor (target center near 0.605)
- localized, stable structures under sustained noise injection ("soliton-ish" behavior)
- early warning proxy via coherence-gradient growth

---

## Quick start

```bash
pip install -r requirements.txt
python orft_1p1d_simulation.py --steps 8000 --dt 0.02 --noise 0.12 --out out/
python analyze_metrics.py --metrics out/metrics.csv
```

---

## Outputs

The run creates:

* `out/config.json` — run config (sim + corridor params)
* `out/metrics.csv` — time series:

  * global coherence
  * local coherence statistics
  * corridor-hit indicator for global coherence in [0.59, 0.62]
  * gradient energy
  * mean corridor energy
  * early-warning proxy (RMS coherence gradient)
* `out/snapshot_*.npz` — optional θ-field snapshots (disable with `--snapshot-every 0`)
* `out/summary.json` — final metrics summary

---

## Notes on model choice

This is a **minimal corridor-locking demonstration** consistent with the ORFT framing:

* θ(x,t) is a 1D phase field on a periodic lattice
* local circular coherence r(x) = |mean exp(iθ)| is treated as the order parameter
* a corridor potential V_band(r) centered at ~0.605 shapes stability
* noise injection tests resilience and locking dynamics

This is intended for reproducibility and interpretability, not maximal physical completeness.
