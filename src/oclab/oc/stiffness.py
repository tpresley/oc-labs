from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict
from .geometry import potential, fisher_metric, curvature_proxy
from ..math.sampling import uniform_box, latin_hypercube
from ..math.linalg import logdet_pd

@dataclass
class StiffnessResult:
    I: Dict[str,float]
    err: Dict[str,float]
    samples_used: int

def estimate_stiffness(cfg) -> StiffnessResult:
    rng = np.random.default_rng(cfg.rng_seed)
    lows = [d[0] for d in cfg.domain]
    highs = [d[1] for d in cfg.domain]
    pts = latin_hypercube(cfg.n_samples, lows, highs, rng)

    wU1 = np.empty(cfg.n_samples, float)
    wSU2 = np.empty(cfg.n_samples, float)
    wSU3 = np.empty(cfg.n_samples, float)
    logw = np.empty(cfg.n_samples, float)
    kraw = np.empty(cfg.n_samples, float)

    # Vectorized pass (loop in Python but inner ops in NumPy)
    for i, y in enumerate(pts):
        V  = potential(y, cfg.centers, cfg.alpha_metric, cfg.beta_floor)
        G  = fisher_metric(y, V, cfg.alpha_metric, cfg.beta_floor,
                           centers=cfg.centers,
                           model=cfg.metric_model,
                           jitter=cfg.jitter)
        # importance weight
        weight = float(np.exp(0.5 * logdet_pd(G)))  # sqrt(det G)
        # raw log-weight (sqrt(det G))
        logw[i] = 0.5 * logdet_pd(G)
        # store raw curvature; we build group factors after robust normalization
        kraw[i] = curvature_proxy(y, cfg.centers, cfg.alpha_metric, cfg.beta_floor)
        wU1[i]  = 1.0

    # stabilize: center by median, clip, exponentiate, then NORMALIZE
    med = float(np.median(logw))
    lw  = np.clip(logw - med, -50.0, 50.0)
    w   = np.exp(lw)                               # unnormalized metric weights
    w_norm = w / np.sum(w)                         # normalized weights (sum = 1)

    # --- Robust curvature normalization ---
    k_med = float(np.median(kraw))
    mad   = float(np.median(np.abs(kraw - k_med))) + 1e-12
    k_hat = (kraw - k_med) / mad               # dimensionless O(1)
    # Optional clamp to avoid extreme leverage from tails
    k_hat = np.clip(k_hat, -5.0, 5.0)
    k_pos = np.maximum(k_hat, 0.0)             # convexity-only for SU(3) suppression

    # --- Bounded group factors ---
    # SU(2): mild enhancement with curvature (allow both signs, but clip)
    f_su2 = 1.0 + cfg.lambda_su2 * k_hat
    f_su2 = np.clip(f_su2, 0.5, 1.5)
    # SU(3): suppression by positive curvature only, strictly <= 1
    f_su3 = 1.0 / (1.0 + cfg.nu_su3 * k_pos)
    f_su3 = np.clip(f_su3, 0.3, 1.0)

    wSU2 = f_su2
    wSU3 = f_su3

    # Weighted expectations  I_a = E_w[f_a]
    I1  = float(np.sum(w_norm * wU1))
    I2  = float(np.sum(w_norm * wSU2))
    I3  = float(np.sum(w_norm * wSU3))

    # Error bars via weighted variance and effective sample size
    def w_se(f):
        f = np.asarray(f, float)
        mu = float(np.sum(w_norm * f))
        var = float(np.sum(w_norm * (f - mu)**2))
        neff = 1.0 / float(np.sum(w_norm**2) + 1e-16)
        se = np.sqrt(var / max(neff, 1.0))
        return se

    e1, e2, e3 = w_se(wU1), w_se(wSU2), w_se(wSU3)

    return StiffnessResult(I={"U1": I1, "SU2": I2, "SU3": I3},
                           err={"U1": e1, "SU2": e2, "SU3": e3},
                           samples_used=len(pts))