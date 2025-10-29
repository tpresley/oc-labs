#!/usr/bin/env python3
import numpy as np
from oclab.config import GeometryConfig, EWAnchor, OCParams, RGRun, FRGScan
from oclab.pipeline import geometry_to_couplings, oc_gap_from_frg
from oclab.frg.flows import (
    solve_growth_c_for_kstar,
    solve_alpha_target_for_kstar,
    expected_kstar_toy,
)
from oclab.io.datasets import bundle_results
import yaml, argparse, json

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.yaml')
p.add_argument('--kstar-desired', type=float, default=None,
               help='If set, adjusts growth_c (or alpha_target with --retune-alpha) to aim for this k*.')
p.add_argument('--retune-alpha', action='store_true',
               help='Use alpha_target tuning instead of growth_c for --kstar-desired.')
p.add_argument('--no-assert', action='store_true',
               help='Disable analytic vs numeric FRG consistency assertions.')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g   = GeometryConfig(**cfg['geometry'])
ew  = EWAnchor(**cfg['ew_anchor'])
oc  = OCParams(**cfg['oc_params'])
rg  = RGRun(**cfg['rg'])
frg = FRGScan(**cfg['frg_scan'])

calib, hist = geometry_to_couplings(g, ew, oc, rg)
alpha0 = calib.alphas_mu0['SU3']
if args.kstar_desired:
    if args.retune_alpha:
        atgt, used_k = solve_alpha_target_for_kstar(alpha0, frg.growth_c, frg.kmax, args.kstar_desired, snap=True)
        frg.alpha_target = atgt
    else:
        frg.growth_c = solve_growth_c_for_kstar(alpha0, frg.alpha_target, frg.kmax, args.kstar_desired)

frg_res, m_tau, C = oc_gap_from_frg(frg, alpha0, ew.mu0, oc, calib.x)

# ---------- Consistency checks & analytics ----------
exp_k = expected_kstar_toy(alpha0, frg.growth_c, frg.kmax, frg.eta_freeze, model=frg.model)
ln_err = None
ok_reason = getattr(frg_res, "reason", None) == "eta_freeze"
try:
    ln_err = float(abs(np.log(exp_k / frg_res.k_star))) if frg_res.k_star else float('inf')
except Exception:
    ln_err = float('inf')

if not args.no_assert:
    # 1) we should freeze by eta, not alpha target
    assert ok_reason, f"FRG stopped by '{frg_res.reason}', expected 'eta_freeze'."
    # 2) numeric k* should match analytic k* within ~15% in log-space for the toy ODE
    assert frg_res.k_star and ln_err < 0.15, f"Numeric k*={frg_res.k_star:g} vs analytic ~{exp_k:g} (ln error={ln_err:.3f})"

out = {
  'config': {'frg': vars(frg), 'ew': vars(ew), 'oc': vars(oc), 'rg': vars(rg)},
  'calibration': {'x': calib.x, 'alphas_mu0': calib.alphas_mu0, 'Ia': calib.Ia, 'Ia_err': calib.Ia_err},
  'frg': {'k_star': frg_res.k_star, 'reason': frg_res.reason, 'reached_alpha': frg_res.reached_alpha,
          'reached_eta': frg_res.reached_eta,
          'expected_k_star': exp_k, 'ln_error_expected_vs_numeric': ln_err},
  'units': {'m_tau': m_tau},
  'C_ratio': C
}
print(json.dumps(out, indent=2))
path = bundle_results(out, prefix="oc_endtoend")
print(f"Saved bundle: {path}")