from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from ..compute.backend import get_backend, to_device
from ..math.sampling import uniform_box  # keep for initial CPU sampling of the box

@dataclass
class StiffnessResult:
    I: Dict[str,float]
    err: Dict[str,float]
    samples_used: int

def _get_compute(cfg):
    # cfg may carry compute attributes via GeometryConfig.__dict__
    # from YAML (we won’t add them to the dataclass to keep it simple)
    compute = getattr(cfg, "compute", None) or {}
    backend = compute.get("backend", "numpy")
    batch   = int(compute.get("batch_size", 65536))
    dtype   = compute.get("dtype", "float32")
    bk = get_backend(backend, dtype)
    return bk, batch

def _centers_to_backend(xp, centers, device):
    # Make a backend array on the right device/dtype
    if xp.__name__ == "torch":
        import torch as th
        return th.tensor(centers, dtype=th.float32, device=device)
    else:
        import numpy as np
        return np.asarray(centers, dtype=np.float32)

def _vec_potential(xp, Y, centers, alpha, beta, device):
    """
    Y: [B,3]; centers: list[[x,y,z,A,s]]  (on CPU list)
    Returns: V [B], d [B,C,3], base [B,C], A [C,1], s [C,1]
    """
    C = _centers_to_backend(xp, centers, device)        # [C,5]
    xyz = C[:, :3]                                      # [C,3]
    A   = C[:, 3:4]                                     # [C,1]
    s   = C[:, 4:5]                                     # [C,1]

    d = Y[:, None, :] - xyz[None, :, :]                 # [B,C,3]
    r2 = (d * d).sum(-1)                                # [B,C]
    base = r2 / (s * s).squeeze(-1) + beta              # [B,C]

    # backend-agnostic power
    base_pow = base ** (-alpha)                         # [B,C]
    V = (A.squeeze(-1) * base_pow).sum(-1)              # [B]
    return V, d, base, A, s

def _vec_hess_logV(xp, Y, centers, alpha, beta, device, jitter=1e-9):
    """
    Grad-outer Fisher proxy:
      G = (∇log V)(∇log V)^T + jitter * I
    Returns: logdet(G) [B], kappa surrogate [B]
    """
    V, d, base, A, s = _vec_potential(xp, Y, centers, alpha, beta, device)

    coef = A.squeeze(-1) * (-alpha) * (base ** (-alpha - 1.0)) * (2.0 / (s * s).squeeze(-1))  # [B,C]
    gradV = (coef[..., None] * d).sum(1)                         # [B,3]
    grad_logV = gradV / V[:, None]                               # [B,3]

    # Identity on correct backend/device
    if xp.__name__ == "torch":
        I3 = xp.eye(3, dtype=grad_logV.dtype, device=grad_logV.device)
    else:
        I3 = xp.eye(3, dtype=grad_logV.dtype)

    # G = g g^T + jitter I
    G = grad_logV[:, :, None] @ grad_logV[:, None, :] + jitter * I3  # [B,3,3]

    # slogdet on both backends
    sign, logabsdet = xp.linalg.slogdet(G)
    # curvature surrogate κ ≈ ||∇logV||^2
    kappa = (grad_logV * grad_logV).sum(-1)
    return logabsdet, kappa


def estimate_stiffness(cfg) -> StiffnessResult:
    bk, batch_size = _get_compute(getattr(cfg, "__dict__", cfg))
    xp, dev = bk.xp, bk.device
    N = int(cfg.n_samples)
    jitter = float(getattr(cfg, "jitter", 1e-9))
    alpha = float(cfg.alpha_metric); beta = float(cfg.beta_floor)
    lam2  = float(cfg.lambda_su2);   nu3  = float(cfg.nu_su3)

    # Prepare domain bounds on backend
    lows  = np.array([d[0] for d in cfg.domain], dtype=np.float32)
    highs = np.array([d[1] for d in cfg.domain], dtype=np.float32)
    if xp.__name__ == "torch":
        import torch as th
        lows_t  = th.tensor(lows,  dtype=th.float32, device=dev)
        highs_t = th.tensor(highs, dtype=th.float32, device=dev)
        # Torch RNG for reproducibility (respect cfg.rng_seed if present)
        g = th.Generator(device=dev)
        if cfg.rng_seed is not None:
            g.manual_seed(int(cfg.rng_seed))
        # Pre-create centers tensor once on device
        from .stiffness import _centers_to_backend  # your helper
        centers_dev = _centers_to_backend(xp, cfg.centers, dev)

        # On-device accumulators (float64 for stable sums)
        w_sum   = th.zeros((), dtype=th.float64, device=dev)
        w_sqsum = th.zeros((), dtype=th.float64, device=dev)
        w1_sum  = th.zeros((), dtype=th.float64, device=dev)
        w2_sum  = th.zeros((), dtype=th.float64, device=dev)
        w3_sum  = th.zeros((), dtype=th.float64, device=dev)
        w1_sq   = th.zeros((), dtype=th.float64, device=dev)
        w2_sq   = th.zeros((), dtype=th.float64, device=dev)
        w3_sq   = th.zeros((), dtype=th.float64, device=dev)

        total = 0
        for start in range(0, N, batch_size):
            B = min(batch_size, N - start)
            # Uniform box on device
            U = th.rand((B, 3), generator=g, device=dev, dtype=th.float32)
            Y = U * (highs_t - lows_t) + lows_t  # [B,3]

            # logdetG, kappa on device (pass centers_dev via a light wrapper)
            logdetG, kappa = _vec_hess_logV(xp, Y, cfg.centers, alpha, beta, device=dev, jitter=jitter)
            w = th.exp(0.5 * logdetG)  # [B]

            # Bounded curvature factors on device
            k_norm = th.tanh(kappa)
            f2 = th.clamp(1.0 + lam2 * k_norm, 0.5, 1.5)
            f3 = th.clamp(1.0 / (1.0 + nu3 * th.clamp(k_norm, 0.0)), 0.3, 1.0)

            # Accumulate normalized moments
            w_sum   += th.sum(w)
            w_sqsum += th.sum(w*w)

            w1_sum  += th.sum(w * 1.0)
            w2_sum  += th.sum(w * f2)
            w3_sum  += th.sum(w * f3)

            w1_sq   += th.sum(w * (1.0**2))
            w2_sq   += th.sum(w * (f2**2))
            w3_sq   += th.sum(w * (f3**2))

            total   += B

        # Reduce to CPU scalars
        w_sum_c, w_sqsum_c = float(w_sum), float(w_sqsum)
        I1 = float(w1_sum / w_sum); I2 = float(w2_sum / w_sum); I3 = float(w3_sum / w_sum)
        neff = (w_sum_c**2) / max(w_sqsum_c, 1e-12)
        e1 = float(((w1_sq / w_sum) - I1*I1).clamp_min(0.0).sqrt() / max(neff**0.5, 1.0))
        e2 = float(((w2_sq / w_sum) - I2*I2).clamp_min(0.0).sqrt() / max(neff**0.5, 1.0))
        e3 = float(((w3_sq / w_sum) - I3*I3).clamp_min(0.0).sqrt() / max(neff**0.5, 1.0))

        return StiffnessResult(I={"U1": I1, "SU2": I2, "SU3": I3},
                               err={"U1": e1, "SU2": e2, "SU3": e3},
                               samples_used=total)

    # ---- NumPy CPU path (unchanged logic, vectorized per batch) ----
    rng = np.random.default_rng(cfg.rng_seed)
    pts_cpu = (rng.random((N, 3), dtype=np.float32) * (highs - lows) + lows).astype(np.float32)

    wU1_sum = 0.0; wSU2_sum = 0.0; wSU3_sum = 0.0
    wU1_sq  = 0.0; wSU2_sq  = 0.0; wSU3_sq  = 0.0
    w_sum   = 0.0; w_sq_sum = 0.0
    total   = 0

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        Y = pts_cpu[start:end]  # [B,3]
        logdetG, kappa = _vec_hess_logV(np, Y, cfg.centers, alpha, beta, device=None, jitter=jitter)
        w = np.exp(0.5 * logdetG)

        k_norm = np.tanh(kappa)
        f2 = np.clip(1.0 + lam2 * k_norm, 0.5, 1.5)
        f3 = np.clip(1.0 / (1.0 + nu3 * np.clip(k_norm, 0.0, None)), 0.3, 1.0)

        W = w
        w_sum   += W.sum()
        w_sq_sum+= (W*W).sum()
        wU1_sum += (W * 1.0).sum()
        wSU2_sum+= (W * f2).sum()
        wSU3_sum+= (W * f3).sum()
        wU1_sq  += (W * 1.0**2).sum()
        wSU2_sq += (W * (f2**2)).sum()
        wSU3_sq += (W * (f3**2)).sum()
        total   += (end - start)

    I1 = wU1_sum / w_sum; I2 = wSU2_sum / w_sum; I3 = wSU3_sum / w_sum
    neff = (w_sum**2) / max(w_sq_sum, 1e-12)
    e1 = float(np.sqrt(max((wU1_sq/w_sum - I1*I1), 0.0) / max(neff, 1.0)))
    e2 = float(np.sqrt(max((wSU2_sq/w_sum - I2*I2), 0.0) / max(neff, 1.0)))
    e3 = float(np.sqrt(max((wSU3_sq/w_sum - I3*I3), 0.0) / max(neff, 1.0)))

    return StiffnessResult(I={"U1": float(I1), "SU2": float(I2), "SU3": float(I3)},
                           err={"U1": float(e1), "SU2": float(e2), "SU3": float(e3)},
                           samples_used=total)

def _scale_centers_widths(centers, lam: float):
    """Tighten/loosen features by scaling the shared width s -> s/lam; xyz, A fixed."""
    out = []
    for x, y, z, A, s in centers:
        s_new = float(s) / float(lam)
        out.append([float(x), float(y), float(z), float(A), s_new])
    return out

def scaling_exponent_I3_widths(gcfg, lams=(0.8,0.85,0.9,0.95,1.05,1.1,1.15,1.2), seeds=(0,1,2)) -> float:
    """
    t ≡ d ln I3 / d ln (1/s) at λ=1, via multi-point, multi-seed fit.
    Implemented by replacing center widths s -> s/λ (tighten for λ>1), keeping xyz and domain fixed.
    """
    import numpy as np
    from .stiffness import estimate_stiffness

    xs, ys = [], []
    for lam in lams:
        vals = []
        for bump in seeds:
            g = gcfg.__class__(**{**gcfg.__dict__})
            g.centers = _scale_centers_widths(gcfg.centers, float(lam))
            g.rng_seed = None if gcfg.rng_seed is None else int(gcfg.rng_seed) + int(bump)
            vals.append(estimate_stiffness(g).I["SU3"])
        xs.append(np.log(float(lam)))               # ln(λ) = ln(1/s) change
        ys.append(np.log(float(np.mean(vals))))     # ln I3
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]  # t = d ln I3 / d ln(1/s)
    return float(slope)
