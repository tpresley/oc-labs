from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from copy import deepcopy

from .stiffness import estimate_stiffness
from .slope import slope_ln_xI3_vs_ln_inv_s   # the raw, width-slope of ln(x I3)
from ..pipeline import geometry_to_couplings
from ..config import GeometryConfig, EWAnchor, OCParams, RGRun

@dataclass
class ActionMatchSpec:
    trace_convention: Literal["tr(TT)=1/2", "tr(TT)=1"] = "tr(TT)=1/2"
    # YM vs α normalization choice (already handled in slope formula, keep as switch if you want variants)
    use_beta0_over_2pi: bool = True
    # Fisher→spacetime Jacobian choice for overall constant
    jacobian: Literal["unity", "detG_sqrt"] = "detG_sqrt"
    # Optional explicit multiplier if you know your mapping
    extra_factor: float = 1.0

def compute_K0_from_components(spec: ActionMatchSpec, g: GeometryConfig | None = None) -> float:
    # trace factor
    c_trace = 1.0 if spec.trace_convention == "tr(TT)=1/2" else 0.5
    c_beta  = 1.0  # β0/(2π) already in slope-mode formula
    # Jacobian: if requested and geometry provided, compute; else 1.0
    if spec.jacobian == "detG_sqrt" and g is not None:
        c_jac = estimate_fisher_jacobian_K0(g)
    else:
        c_jac = 1.0
    return float(c_trace * c_beta * c_jac * spec.extra_factor)

def compute_K0_by_calibration(
    g: GeometryConfig, ew: EWAnchor, oc: OCParams, rg: RGRun,
    alpha3_target_MZ: float = 0.1181,
    slope_lams = (0.85, 0.9, 0.95, 1.05, 1.1, 1.15),
    slope_seeds = (0,1,2)
) -> float:
    """
    One-time, parameter-free K0 via Route-B slope → compare to target at MZ → K0.
    This *does not* fit per-run; it produces a single universal constant.
    """
    # 1) Measure slope u = d ln(x I3)/d ln(1/s) with raw path
    u = slope_ln_xI3_vs_ln_inv_s(g, ew, oc, lams=slope_lams, seeds=slope_seeds)
    if not np.isfinite(u) or abs(u) < 1e-3:
        raise ValueError(f"Unstable slope for ln(xI3): u={u}. Increase MC or widen λ grid.")

    # 2) Baseline x and I3 (raw) at λ = 1
    stiff0 = estimate_stiffness(g, mode="raw")
    # recompute baseline x from EW lock (same as in calibrate_x_and_couplings)
    alpha_em, s2 = float(ew.alpha_em_mu0), float(ew.sin2_thetaW_mu0)
    c2 = 1.0 - s2
    alpha1 = (5.0/3.0) * alpha_em / c2
    alpha2 = alpha_em / s2

    def Ka(group: str):
        kap = abs(oc.kappa.get(group, 1))
        W   = oc.weights.get(group, 1.0)
        n   = oc.n_color_discrete or 1
        base = (kap ** oc.Ka_power_kappa) * W
        if oc.Ka_divide_by_n: base /= n
        return oc.Ka_prefactor * base

    x1 = (1.0/alpha1) / (Ka("U1") * stiff0.I["U1"])
    x2 = (1.0/alpha2) / (Ka("SU2") * stiff0.I["SU2"])
    x0 = float(np.sqrt(max(x1,1e-30)*max(x2,1e-30)))
    I30 = float(stiff0.I["SU3"])

    # 3) Slope-mode K3 without K0 (pure Route-B)
    #    K3_slope = (β0 / (2π)) / (|u| * x0 * I30)
    #    Here choose nf from the window (e.g., 150→MZ => nf=5), or read a thresholds fn if preferred.
    nf = 5
    beta0 = 11.0 - 2.0*nf/3.0
    K3_noK0 = (beta0 / (2.0*np.pi)) / (abs(u) * x0 * I30)

    # 4) Predict α3(μ0) and run to MZ (no K0 yet)
    oc_tmp = deepcopy(oc)
    oc_tmp.Ka_mode = "fixed"     # we will set SU3 from K3_noK0 * K0 later
    oc_tmp.Ka_group_scale = dict(oc.Ka_group_scale or {})  # no effect here

    # Build a temporary SU3 normalization with K0=1:
    # 1/α3(μ0) = K3 * x0 * I30 => α3(μ0) = 1/(K3 * x0 * I30)
    alpha3_mu0_noK0 = 1.0 / (K3_noK0 * x0 * I30)
    # To use existing pipeline we can just proceed to RG by injecting this α3 at μ0;
    # but the simplest: run the geometry→RG once and replace α3 at μ0 in history is overkill.
    # We can do a one-step short run: reuse pipeline to get hist for α1,2 and then overwrite α3 start.

    # Use pipeline to get hist container shapes
    calib0, hist0 = geometry_to_couplings(g, ew, oc, rg)
    # overwrite α3 start and re-run RG with that start (light re-run):
    from ..sm.rgrun import run_couplings
    alphas_mu0 = dict(U1=calib0.alphas_mu0["U1"], SU2=calib0.alphas_mu0["SU2"], SU3=alpha3_mu0_noK0)
    hist_noK0 = run_couplings(alphas_mu0, ew.mu0, rg, ew.scheme)
    alpha3_MZ_noK0 = float(hist_noK0.alpha3[-1])

    # 5) Universal K0 needed so that α3(MZ) matches the target:
    # α ∝ 1/K0  (since K3 -> K0*K3_noK0 and α3 = 1/(K3 * x0 * I30))
    K0_needed = float(alpha3_MZ_noK0 / alpha3_target_MZ)
    return K0_needed


def estimate_fisher_jacobian_K0(g: GeometryConfig, *, samples: int | None = None) -> float:
    """
    Estimate a universal K0 from the baseline Fisher measure:
      J0 = mean_{y in domain} sqrt(det G(y))  (at λ=1),
      K0 = 1 / J0.
    Uses whatever metric model g has (prefer Hessian-logV for normalization).
    """
    from copy import deepcopy
    import numpy as np
    from ..compute.backend import get_backend, to_device
    from .stiffness import _centers_to_backend, _I3_backend, _vec_V_grad_hess

    # Decide the sample count
    N = int(samples or g.n_samples or 160_000)

    # CPU NumPy is fine here; this runs once
    import numpy as xp
    lows  = xp.array([d[0] for d in g.domain], dtype=xp.float64)
    highs = xp.array([d[1] for d in g.domain], dtype=xp.float64)
    rng   = np.random.default_rng(g.rng_seed)

    pts = rng.random((N, 3)) * (highs - lows) + lows   # [N,3]
    # Build Hessian-logV and get G = ∇² log V + εI
    V, gradV, hessV = _vec_V_grad_hess(xp, pts.astype(np.float64), g.centers, float(g.alpha_metric), float(g.beta_floor), device=None, jitter=getattr(g, "jitter", 1e-9))
    Vcol = V[:, None, None]
    grad_outer = gradV[:, :, None] @ gradV[:, None, :]
    HlogV = (hessV / Vcol) - (grad_outer / (Vcol * Vcol))       # [N,3,3]
    I3 = xp.eye(3, dtype=HlogV.dtype)
    G = HlogV + float(getattr(g, "jitter", 1e-9)) * I3
    sign, logabsdet = xp.linalg.slogdet(G)
    # mean sqrt(det G)
    J0 = float(xp.exp(0.5 * logabsdet).mean())
    if not np.isfinite(J0) or J0 <= 0:
        raise ValueError(f"Bad Fisher Jacobian estimate: J0={J0}")
    return 1.0 / J0