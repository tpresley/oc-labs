#!/usr/bin/env python3
from oclab.config import GeometryConfig, EWAnchor, OCParams, RGRun, FRGScan
from oclab.pipeline import geometry_to_couplings, oc_gap_from_frg
from oclab.frg.flows import solve_alpha_target_for_kstar, expected_kstar_toy
import yaml, argparse, json
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.yaml')
p.add_argument("--kstar-desired", type=float, default=None,
               help="If set, compute alpha_target to freeze at this k* (GeV).")
p.add_argument('--scan-frg', type=str, default=None,
               help='Sweep growth_c over "start:stop:step" (e.g., 1.3:1.9:0.05) and write a CSV. '
                    'Prefers η-based freeze (alpha_target set inert).')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g   = GeometryConfig(**cfg['geometry'])
ew  = EWAnchor(**cfg['ew_anchor'])
oc  = OCParams(**cfg['oc_params'])
rg  = RGRun(**cfg['rg'])
frg = FRGScan(**cfg['frg_scan'])

calib, hist = geometry_to_couplings(g, ew, oc, rg)
alpha0 = calib.alphas_mu0['SU3']

# ---------- FRG sweep mode ----------
if args.scan_frg:
    # prefer η-freeze: make α-target inert
    frg.alpha_target = 1e9
    # parse "start:stop:step"
    try:
        s, t, d = [float(x) for x in args.scan_frg.split(":")]
    except Exception:
        raise SystemExit('Invalid --scan-frg format. Use "start:stop:step", e.g., 1.3:1.9:0.05')
    vals = []
    # include stop (within floating tolerance)
    gc_list = list(np.arange(s, t + 0.5*abs(d), d))
    print(f"[FRG sweep] growth_c ∈ {gc_list}")
    for gc in gc_list:
        frg.growth_c = float(gc)
        frg_res_i, m_tau_i, C_i = oc_gap_from_frg(frg, alpha0, ew.mu0, oc, calib.x)
        exp_k_i = expected_kstar_toy(alpha0, frg.growth_c, frg.kmax, frg.eta_freeze, model=frg.model)
        ln_err_i = (abs(np.log(exp_k_i / frg_res_i.k_star))
                    if frg_res_i.k_star else float('inf'))
        vals.append([
            frg.growth_c,
            frg.eta_freeze,
            frg.alpha_cap,
            frg_res_i.k_star,
            frg_res_i.reason,
            frg_res_i.reached_alpha,
            frg_res_i.reached_eta,
            exp_k_i,
            ln_err_i,
            m_tau_i,
            C_i,
        ])
    from oclab.io.datasets import write_csv
    headers = [
        "growth_c","eta_freeze","alpha_cap","k_star_numeric","reason",
        "reached_alpha","reached_eta","k_star_expected","ln_error",
        "m_tau","C_ratio"
    ]
    path = write_csv(vals, headers, "frg_sweep.csv")
    print(f"[FRG sweep] wrote {path}")
    raise SystemExit(0)

# ---------- Single FRG run ----------
frg_res, m_tau, C = oc_gap_from_frg(frg, alpha0, ew.mu0, oc, calib.x)

if args.kstar_desired is not None:
    atgt, used_k = solve_alpha_target_for_kstar(
        calib.alphas_mu0["SU3"], frg.growth_c, frg.kmax, args.kstar_desired, snap=True
    )
    frg.alpha_target = atgt
    print(f"[FRG] solved alpha_target={frg.alpha_target:.6g} for k*≈{used_k:.5g}")

if frg.alpha_target <= frg.alpha_cap * 1.05:
    frg.alpha_target = 1e9
    print("[FRG] alpha_target set inert (1e9) to prefer η-based freeze.")

exp_k = expected_kstar_toy(calib.alphas_mu0["SU3"], frg.growth_c, frg.kmax, frg.eta_freeze, model=frg.model)
ln_err = abs(np.log(exp_k / frg_res.k_star)) if frg_res.k_star else float("inf")

# Assertions (leave them on by default; comment out if you truly need to bypass)
assert frg_res.reason == "eta_freeze", f"FRG stopped by '{frg_res.reason}', expected 'eta_freeze'."
assert frg_res.k_star and ln_err < 0.15, f"Numeric k*={frg_res.k_star:g} vs analytic ~{exp_k:g} (ln error={ln_err:.3f})"

bundle = {
  'x': calib.x,
  'Ia': calib.Ia,
  'Ia_err': calib.Ia_err,
  'alphas_mu0': calib.alphas_mu0,
  'final_mu': float(hist.mu[-1]),
  'alpha3_final': float(hist.alpha3[-1]),
  'k_star': frg_res.k_star,
  'm_tau': m_tau,
  'C': C,
  'frg': {
      'k_star': frg_res.k_star,
      'reason': frg_res.reason,
      'reached_alpha': frg_res.reached_alpha,
      'reached_eta': frg_res.reached_eta,
      'expected_k_star': exp_k,
      'ln_error_expected_vs_numeric': ln_err,
  }
}
print(json.dumps(bundle, indent=2))
