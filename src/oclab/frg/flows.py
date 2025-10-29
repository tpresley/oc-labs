from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Callable
from .bf_models import eta_minimal, eta_projected
from dataclasses import replace

def solve_growth_c_for_kstar(alpha0: float, alpha_target: float, kmax: float, k_star_desired: float) -> float:
    dln = float(np.log(kmax / k_star_desired))
    if dln <= 0:
        raise ValueError("k_star_desired must be < kmax.")
    return float((1.0/alpha0 - 1.0/alpha_target) / dln)

def kstar_min_reachable(alpha0: float, growth_c: float, kmax: float) -> float:
    return float(kmax * np.exp(-1.0/(growth_c*alpha0)))

def solve_alpha_target_for_kstar(alpha0: float, growth_c: float, kmax: float, k_star_desired: float,
                                 *, snap=False, eps_ln: float = 1e-6) -> tuple[float, float]:
    kmin_feasible = kstar_min_reachable(alpha0, growth_c, kmax)
    used_k = float(k_star_desired)
    if k_star_desired < kmin_feasible:
        if not snap:
            raise ValueError("Requested k* is unreachable for this growth_c.")
        dln_min = 1.0/(growth_c*alpha0)
        dln = max(dln_min - float(eps_ln), 0.0)
        used_k = float(kmax * np.exp(-dln))
    dln = float(np.log(kmax / used_k))
    denom = 1.0 - growth_c * alpha0 * dln
    if denom <= 0:
        raise ValueError("Internal: denominator <= 0; increase eps_ln.")
    return float(alpha0 / denom), used_k



def tune_growth_for_C(alpha0, mu0, scan, ocp, x, target_C, tol=0.05, max_iter=20):
    """
    Binary-search growth_c to achieve target C within ±tol relative error.
    Returns (frg_result, m_tau, C, growth_c_used).
    """
    from ..oc.units import m_tau
    mτ = m_tau(ocp.kappa, x)

    # search bracket: conservative defaults
    lo, hi = 1.0, 12.0
    def run(g):
        s = replace(scan, growth_c=float(g))
        res = scan_freeze(alpha0, mu0, s, model=s.model)
        C = float("nan") if not res.k_star else (res.k_star / mτ)
        return res, C

    res_lo, C_lo = run(lo)
    res_hi, C_hi = run(hi)
    if not (res_lo.k_star and res_hi.k_star):
        return (res_hi if res_hi.k_star else res_lo), mτ, (C_hi if res_hi.k_star else C_lo), (hi if res_hi.k_star else lo)

    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        res_mid, C_mid = run(mid)
        if not res_mid.k_star or not (C_mid > 0):
            lo = mid  # push up if no freeze
            continue
        err = abs(C_mid - target_C)/target_C
        if err <= tol:
            return res_mid, mτ, C_mid, mid
        # monotonic: C grows with growth_c
        if C_mid < target_C: lo = mid
        else: hi = mid

    # fall back
    res_mid, C_mid = run(0.5*(lo+hi))
    return res_mid, mτ, C_mid, 0.5*(lo+hi)



@dataclass
class FRGResult:
    k: np.ndarray
    alpha: np.ndarray
    etaA: np.ndarray
    k_star: float | None
    reason: str | None = None            # NEW: why we stopped
    reached_alpha: float | None = None   # NEW: last α
    reached_eta: float | None = None     # NEW: last |η|
    analytic_note: str | None = None     # NEW: analytic reachability summary

def _eta(model: str) -> Callable[[float], float]:
    if model == "minimal": return eta_minimal
    return eta_projected

# ---------- Analytic helpers for the toy ODE dα/dlnk = c α^2 ----------
def alpha_for_eta(eta_abs: float, model: str = "projected") -> float:
    """
    Invert |η(α)| = eta_abs for our built-in η models.
    projected: |η| = 0.9 * α / (1 + 0.5 α)  =>  α = |η| / (0.9 - 0.5 |η|)
    minimal  : |η| = 0.6 * α / (1 + α)      =>  α = |η| / (0.6 - |η|)
    """
    e = float(eta_abs)
    if not (0.0 < e < 1.0):
        raise ValueError("eta_abs must be in (0,1).")
    if model == "minimal":
        denom = 0.6 - e
        if denom <= 0: raise ValueError("eta_abs too large for minimal model.")
        return e / denom
    # projected
    denom = 0.9 - 0.5 * e
    if denom <= 0: raise ValueError("eta_abs too large for projected model.")
    return e / denom

def expected_kstar_toy(alpha0: float, growth_c: float, kmax: float,
                       eta_freeze: float, model: str = "projected") -> float:
    """
    Analytic k* for the toy flow with η-based stop:
      dα/dlnk = c α^2  and  stop when |η(α)| = eta_freeze.
    """
    a_eta = alpha_for_eta(eta_freeze, model=model)
    dln   = (1.0/alpha0 - 1.0/a_eta) / float(growth_c)
    return float(kmax * np.exp(-dln))

def scan_freeze(alpha0: float, mu0: float, scan, model: str="projected") -> FRGResult:
    # UV → IR grid
    k = np.geomspace(scan.kmax, scan.kmin, scan.n_k)
    alpha = np.empty_like(k)
    etaA  = np.empty_like(k)

    alpha[0] = float(alpha0)
    eta_fn = _eta(model)
    etaA[0] = eta_fn(alpha[0])

    # --- Analytic reachability estimate for α-target ---
    dln_avail = float(np.log(scan.kmax/scan.kmin))
    analytic_note = None
    if scan.alpha_target > alpha[0]:
        # Δln_req = (1/α0 - 1/α_tgt)/c
        dln_req = (1.0/alpha[0] - 1.0/float(scan.alpha_target)) / float(scan.growth_c)
        analytic_note = (f"alpha0={alpha[0]:.6g}, c={scan.growth_c:.3g}, "
                         f"alpha_tgt={scan.alpha_target:.6g}, "
                         f"Δln_req={dln_req:.4f}, Δln_avail={dln_avail:.4f}")

    k_star = None
    last_i = 0

    for i in range(1, len(k)):
        # positive step in ln k toward IR
        dln = np.log(k[i-1] / k[i])  # > 0 since k[i] < k[i-1]
        a = alpha[i-1]
        # toy IR growth with tunable strength; cap to avoid overflow
        a_new = max(a + scan.growth_c * a * a * dln, 1e-16)
        a_new = min(a_new, float(scan.alpha_cap))
        alpha[i] = a_new
        etaA[i] = eta_fn(a_new)

        # early-stop on freeze criterion
        if abs(etaA[i]) >= scan.eta_freeze or a_new >= scan.alpha_target:
            k_star = float(k[i])
            last_i = i
            reason = ("eta_freeze" if abs(etaA[i]) >= scan.eta_freeze else "alpha_target")
            # truncate to computed range
            k = k[: last_i + 1]; alpha = alpha[: last_i + 1]; etaA = etaA[: last_i + 1]
            return FRGResult(k=k, alpha=alpha, etaA=etaA, k_star=k_star,
                             reason=reason, reached_alpha=float(alpha[-1]),
                             reached_eta=float(abs(etaA[-1])), analytic_note=analytic_note)
        last_i = i

    # truncate to computed range (removes tail with uninitialized values)
    k = k[: last_i + 1]
    alpha = alpha[: last_i + 1]
    etaA = etaA[: last_i + 1]

    # No freeze reached: report what we achieved
    return FRGResult(k=k, alpha=alpha, etaA=etaA, k_star=None,
                     reason="range_exhausted",
                     reached_alpha=float(alpha[-1]),
                     reached_eta=float(abs(etaA[-1])),
                     analytic_note=analytic_note)
