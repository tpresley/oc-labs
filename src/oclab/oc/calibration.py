from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
from ..config import _to_float
import numpy as np

@dataclass
class CalibrationResult:
    x: float
    alphas_mu0: Dict[str,float]
    Ia: Dict[str,float]
    Ia_err: Dict[str,float]

def calibrate_x_and_couplings(gcfg, ew, ocp) -> CalibrationResult:
    # Lazily import to avoid circulars
    from .stiffness import estimate_stiffness
    from .stiffness import scaling_exponent_I3
    stiff = estimate_stiffness(gcfg)

    # GUT-normalized α1 from α_em, θ_W
    alpha_em = _to_float(ew.alpha_em_mu0)
    s2       = _to_float(ew.sin2_thetaW_mu0)
    c2 = 1.0 - s2
    alpha1 = (5.0/3.0) * alpha_em / c2
    alpha2 = alpha_em / s2

    # --- Build Ka(a) = pref * (|kappa_a|^p) * W_a * group_scale[a] / (n_color_discrete if enabled) ---
    def Ka(group: str) -> float:
        kap = abs(ocp.kappa.get(group, 1))
        W   = ocp.weights.get(group, 1.0)
        gs  = 1.0 if not ocp.Ka_group_scale else float(ocp.Ka_group_scale.get(group, 1.0))
        base = (kap ** ocp.Ka_power_kappa) * W * gs
        if ocp.Ka_divide_by_n and ocp.n_color_discrete:
            base /= float(ocp.n_color_discrete)
        return ocp.Ka_prefactor * base

    K1 = Ka("U1")
    K2 = Ka("SU2")
    x1 = (1.0/alpha1) / (K1 * stiff.I["U1"])
    x2 = (1.0/alpha2) / (K2 * stiff.I["SU2"])
    x = math.sqrt(max(x1,1e-30)*max(x2,1e-30))

    # Predict α3 at μ0
    # --- SU(3) normalization ---
    if getattr(ocp, "Ka_mode", "fixed") == "slope":
        from .stiffness import scaling_exponent_I3_widths

        beta0 = 11.0 - 2.0*5.0/3.0  # nf=5 for 150→MZ
        t = scaling_exponent_I3_widths(gcfg)  # t = d ln I3 / d ln(1/s)
        print(f"[slope-widths] t(dln I3/dln 1/s)={t:.6g}, I3={stiff.I['SU3']:.6g}, x={x:.6g}")

        if not np.isfinite(t) or abs(t) < 1e-3:
            raise ValueError(f"Width-slope too small/unstable (t={t}); increase geometry.n_samples or widen λ grid.")
        K3 = (beta0 / (2.0*np.pi)) / (abs(t) * stiff.I["SU3"] * x)  # positive by construction
        inv_alpha3 = K3 * stiff.I["SU3"] * x
        alpha3 = 1.0 / inv_alpha3

    else:
        K3 = Ka("SU3")


    inv_alpha3 = K3 * stiff.I["SU3"] * x
    alpha3 = 1.0 / inv_alpha3 if inv_alpha3>0 else float("nan")

    alphas_mu0 = {"U1": alpha1, "SU2": alpha2, "SU3": alpha3}
    return CalibrationResult(x=x, alphas_mu0=alphas_mu0, Ia=stiff.I, Ia_err=stiff.err)
