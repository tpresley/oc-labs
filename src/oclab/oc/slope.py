# src/oclab/oc/slope.py (new)
from __future__ import annotations
import numpy as np
from copy import deepcopy
from .stiffness import estimate_stiffness
from .geometry import potential  # only to ensure module import; not used here
from ..config import GeometryConfig, EWAnchor, OCParams, RGRun

def _scale_widths(centers, lam: float):
    # s -> s/lam; leave xyz, A unchanged
    return [[x, y, z, A, float(s)/float(lam)] for x,y,z,A,s in centers]

def slope_ln_xI3_vs_ln_inv_s(
    gcfg: GeometryConfig,
    ew: EWAnchor,
    ocp: OCParams,
    lams=(0.85, 0.9, 0.95, 1.05, 1.1, 1.15),
    seeds=(0,1,2)
) -> float:
    """
    u = d ln(x*I3) / d ln(1/s) at λ=1; width tightening proxy (s->s/λ).
    Recomputes x from EW lock at each λ (x depends on I1, I2).
    """
    xs, ys = [], []
    for lam in lams:
        vals = []
        for bump in seeds:
            g = deepcopy(gcfg)
            g.centers = _scale_widths(gcfg.centers, lam)
            # decorrelate MC noise
            g.rng_seed = None if gcfg.rng_seed is None else int(gcfg.rng_seed) + int(bump)
            stiff = estimate_stiffness(g)

            # recompute x from U(1), SU(2) at this lam (no SU(3) info needed)
            alpha_em = ew.alpha_em_mu0; s2 = ew.sin2_thetaW_mu0; c2 = 1.0 - s2
            alpha1 = (5.0/3.0) * alpha_em / c2
            alpha2 = alpha_em / s2

            # same Ka() you use in calibration for U1 and SU2
            kap = ocp.kappa; W = ocp.weights; n = ocp.n_color_discrete or 1
            K1 = ocp.Ka_prefactor * (abs(kap.get("U1",1))**ocp.Ka_power_kappa) * W.get("U1",1.0) / (n if ocp.Ka_divide_by_n else 1.0)
            K2 = ocp.Ka_prefactor * (abs(kap.get("SU2",1))**ocp.Ka_power_kappa) * W.get("SU2",1.0) / (n if ocp.Ka_divide_by_n else 1.0)

            x1 = (1.0/alpha1) / (K1 * stiff.I["U1"])
            x2 = (1.0/alpha2) / (K2 * stiff.I["SU2"])
            x  = float(np.sqrt(max(x1,1e-30)*max(x2,1e-30)))

            vals.append(x * stiff.I["SU3"])

        xs.append(np.log(float(lam)))                  # ln λ  (= ln(1/s) change)
        ys.append(np.log(float(np.mean(vals))))        # ln (x I3)

    xs = np.asarray(xs); ys = np.asarray(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]   # u = d ln(xI3)/d ln λ
    return float(slope)
