# oclab

A unified, notebook-friendly codebase that reproduces the two prior pipelines:

1) Geometry → stiffness integrals → calibrate `x` with EW lock → predict gauge couplings.
2) RG running from `mu0` to target scale.
3) FRG background-field toy flows to extract a freeze scale `k_*` and OC ratio `C = k_*/m_tau`.

## Quickstart (editable install)

```bash
pip install -e /mnt/data/oclab
```

Then open the starter notebook in `notebooks/01_end_to_end.ipynb`.
