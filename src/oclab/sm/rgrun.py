from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict
from .thresholds import nf_piecewise
from .beta import beta_qcd, beta_SM_alphas
from ..math.integration import rk4_step

@dataclass
class RGHistory:
    mu: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    alpha3: np.ndarray
    nf: np.ndarray

def run_couplings(alphas_mu0: Dict[str,float], mu0: float, target, scheme: str='Q3') -> RGHistory:
    """
    Run (α1, α2, α3) from μ0 to target.mu_target.
    - α1, α2: full SM 1–2 loop (valid with full SM content)
    - α3: QCD with piecewise nf thresholds (scheme ladder)
    """
    n = getattr(target, "n_steps", 400)
    # Ensure monotone spacing regardless of direction
    if target.mu_target == mu0:
        mu = np.full(n, mu0)
    elif target.mu_target < mu0:
        mu = np.geomspace(mu0, target.mu_target, n)
    else:
        mu = np.geomspace(mu0, target.mu_target, n)

    alpha1 = np.empty(n); alpha2 = np.empty(n); alpha3 = np.empty(n)
    nf = np.empty(n, dtype=int)

    alpha1[0] = alphas_mu0["U1"]
    alpha2[0] = alphas_mu0["SU2"]
    alpha3[0] = alphas_mu0["SU3"]
    nf[0] = nf_piecewise(mu[0], scheme)

    # integrate in t = ln μ
    t = np.log(mu)
    y = np.array([alpha1[0], alpha2[0], alpha3[0]], dtype=float)
    nf[0] = nf_piecewise(mu[0], scheme)

    def f(ti, yi):
        # yi = [α1, α2, α3] at current nf
        a1, a2, a3 = yi
        d1, d2, _ = beta_SM_alphas((a1, a2, a3), order=target.loop_order)
        d3 = beta_qcd(a3, int(cur_nf), order=target.loop_order)
        return np.array([d1, d2, d3], dtype=float)

    cur_nf = nf[0]
    for i in range(1, n):
        # piecewise nf: use nf at the previous μ
        cur_nf = nf_piecewise(mu[i-1], scheme)
        dt = t[i] - t[i-1]
        y = rk4_step(y, t[i-1], dt, f)
        alpha1[i], alpha2[i], alpha3[i] = np.maximum(y, 1e-16)
        nf[i] = cur_nf

    return RGHistory(mu=mu, alpha1=alpha1, alpha2=alpha2, alpha3=alpha3, nf=nf)