from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Tuple, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from ..compute.backend import get_backend, to_device
from ..math.sampling import uniform_box  # keep for initial CPU sampling of the box

@dataclass
class StiffnessResult:
    I: Dict[str,float]
    err: Dict[str,float]
    samples_used: int

# ---------- helpers ----------

def _get_compute(cfg):
    # Backwards-compatible: adds optional num_workers and num_streams knobs.
    compute = getattr(cfg, "compute", None) or {}
    backend = compute.get("backend", "numpy")
    batch   = int(compute.get("batch_size", 65536))
    dtype   = compute.get("dtype", "float32")
    # new (optional) — defaults are conservative if absent
    num_workers = int(compute.get("num_workers", 0))      # CPU threads for numpy path (0 => auto)
    num_streams = int(compute.get("num_streams", 0))      # CUDA streams for torch path (0 => auto)
    bk = get_backend(backend, dtype)
    return bk, batch, num_workers, num_streams

def _centers_to_backend(xp, centers, device):
    if xp.__name__ == "torch":
        import torch as th
        return th.tensor(centers, dtype=th.float32, device=device)
    else:
        import numpy as np
        return np.asarray(centers, dtype=np.float32)

def _vec_potential(xp, Y, centers, alpha, beta, device):
    C = _centers_to_backend(xp, centers, device)        # [C,5]
    xyz = C[:, :3]
    A   = C[:, 3:4]
    s   = C[:, 4:5]
    d = Y[:, None, :] - xyz[None, :, :]                 # [B,C,3]
    r2 = (d * d).sum(-1)                                # [B,C]
    base = r2 / (s * s).squeeze(-1) + beta              # [B,C]
    base_pow = base ** (-alpha)
    V = (A.squeeze(-1) * base_pow).sum(-1)              # [B]
    return V, d, base, A, s

def _vec_hess_logV(xp, Y, centers, alpha, beta, device, jitter=1e-9):
    V, d, base, A, s = _vec_potential(xp, Y, centers, alpha, beta, device)
    coef = A.squeeze(-1) * (-alpha) * (base ** (-alpha - 1.0)) * (2.0 / (s * s).squeeze(-1))  # [B,C]
    gradV = (coef[..., None] * d).sum(1)                         # [B,3]
    grad_logV = gradV / V[:, None]                               # [B,3]
    if xp.__name__ == "torch":
        I3 = xp.eye(3, dtype=grad_logV.dtype, device=grad_logV.device)
    else:
        I3 = xp.eye(3, dtype=grad_logV.dtype)
    G = grad_logV[:, :, None] @ grad_logV[:, None, :] + jitter * I3  # [B,3,3]
    sign, logabsdet = xp.linalg.slogdet(G)
    kappa = (grad_logV * grad_logV).sum(-1)
    return logabsdet, kappa


def _I3_backend(xp, dtype, device):
    if xp.__name__ == "torch":
        return xp.eye(3, dtype=dtype, device=device)
    return xp.eye(3, dtype=dtype)

def _vec_V_grad_hess(xp, Y, centers, alpha, beta, device, jitter=1e-9):
    """
    Batched V, ∇V, and Hess V for V(y) = Σ_c A_c * (r_c^2/s_c^2 + beta)^(-alpha).
    Inputs:
      Y       : [B,3]
      centers : [[x,y,z,A,s], ...] length C
    Returns:
      V       : [B]
      gradV   : [B,3]
      hessV   : [B,3,3]   (full Hessian of V)
    """
    # centers on backend
    Cmat = _centers_to_backend(xp, centers, device)   # [C,5] float32
    xyz = Cmat[:, :3]                                 # [C,3]
    A   = Cmat[:, 3:4]                                # [C,1]
    s   = Cmat[:, 4:5]                                # [C,1]

    # d = y-c, r2 = ||d||^2
    d = Y[:, None, :] - xyz[None, :, :]              # [B,C,3]
    r2 = (d * d).sum(-1)                             # [B,C]
    s2 = (s * s).squeeze(-1)                         # [C]
    base = r2 / s2 + beta                            # [B,C]

    # common powers
    base_m_a   = base ** (-alpha)                    # [B,C]
    base_m_a_1 = base ** (-alpha - 1.0)              # [B,C]
    base_m_a_2 = base ** (-alpha - 2.0)              # [B,C]

    # V
    V = (A.squeeze(-1) * base_m_a).sum(-1)           # [B]

    # ∇V = Σ A * (-α) * base^(-α-1) * 2*(y-c)/s^2
    coef_g = A.squeeze(-1) * (-alpha) * base_m_a_1 * (2.0 / s2)  # [B,C]
    gradV = (coef_g[..., None] * d).sum(1)           # [B,3]

    # Hess V = Σ A [ 2*(-α)*base^(-α-1)/s^2 * I + 4*(-α)(-α-1)*base^(-α-2)/s^4 * (d d^T) ]
    I3 = _I3_backend(xp, dtype=gradV.dtype, device=(gradV.device if hasattr(gradV, "device") else None))
    term_I = A.squeeze(-1) * (2.0 * (-alpha) * base_m_a_1 / s2)      # [B,C]
    term_O = A.squeeze(-1) * (4.0 * (-alpha) * (-alpha - 1.0) * base_m_a_2 / (s2 * s2))  # [B,C]

    # Build HessV by summing over centers
    # outer(d,d): [B,C,3,3]
    outer_dd = xp.einsum("bcx,bcy->bcxy", d, d)
    hessV = (term_I[..., None, None] * I3).sum(1) + (term_O[..., None, None] * outer_dd).sum(1)  # [B,3,3]

    return V, gradV, hessV

def _vec_hessian_logV_weight_and_kappa(xp, Y, centers, alpha, beta, device, jitter=1e-9):
    """
    Fisher (raw) weight from Hessian of log V:
      G = ∇² log V + jitter * I
    Returns:
      logdetG           : [B]  (log |det G|)
      kappa_hess_trace  : [B]  (tr(∇² log V))
    """
    V, gradV, hessV = _vec_V_grad_hess(xp, Y, centers, alpha, beta, device, jitter=jitter)

    # ∇² log V = (∇² V)/V - (∇V ∇V^T)/V^2
    Vcol = V[:, None, None]
    grad_outer = gradV[:, :, None] @ gradV[:, None, :]
    HlogV = (hessV / Vcol) - (grad_outer / (Vcol * Vcol))            # [B,3,3]

    # Add jitter*I and compute slogdet
    I3 = _I3_backend(xp, dtype=HlogV.dtype, device=(HlogV.device if hasattr(HlogV, "device") else None))
    G = HlogV + jitter * I3
    sign, logabsdet = xp.linalg.slogdet(G)

    # κ = tr(∇² log V)
    kappa_hess_trace = xp.trace(HlogV, axis1=1, axis2=2)             # [B]
    return logabsdet, kappa_hess_trace


# ---------- main compute ----------

def estimate_stiffness(cfg, mode: str = "robust") -> StiffnessResult:
    bk, batch_size, num_workers, num_streams = _get_compute(getattr(cfg, "__dict__", cfg))
    xp, dev = bk.xp, bk.device
    N = int(cfg.n_samples)
    jitter = float(getattr(cfg, "jitter", 1e-9))
    alpha = float(cfg.alpha_metric); beta = float(cfg.beta_floor)
    lam2  = float(cfg.lambda_su2);   nu3  = float(cfg.nu_su3)

    # Domain bounds (CPU numpy is fine for constants)
    lows  = np.array([d[0] for d in cfg.domain], dtype=np.float32)
    highs = np.array([d[1] for d in cfg.domain], dtype=np.float32)

    # ---------------- Torch (GPU) with CUDA streams ----------------
    if xp.__name__ == "torch":
        import torch as th

        lows_t  = th.tensor(lows,  dtype=th.float32, device=dev)
        highs_t = th.tensor(highs, dtype=th.float32, device=dev)

        g = th.Generator(device=dev)
        if getattr(cfg, "rng_seed", None) is not None:
            g.manual_seed(int(cfg.rng_seed))

        # choose streams
        if num_streams <= 0:
            # Heuristic: a few streams often enough to overlap kernels and rng.
            num_streams = 4
        streams = [th.cuda.Stream(device=dev) for _ in range(num_streams)]

        # accumulators (fp64 on device)
        w_sum   = th.zeros((), dtype=th.float64, device=dev)
        w_sqsum = th.zeros((), dtype=th.float64, device=dev)
        w1_sum  = th.zeros((), dtype=th.float64, device=dev)
        w2_sum  = th.zeros((), dtype=th.float64, device=dev)
        w3_sum  = th.zeros((), dtype=th.float64, device=dev)
        w1_sq   = th.zeros((), dtype=th.float64, device=dev)
        w2_sq   = th.zeros((), dtype=th.float64, device=dev)
        w3_sq   = th.zeros((), dtype=th.float64, device=dev)

        total = 0
        # Outer loop over big batches
        for start in range(0, N, batch_size):
            B = min(batch_size, N - start)

            # Split this batch across streams
            # keep chunks balanced; last chunk may be smaller
            sizes = [B // num_streams] * num_streams
            for i in range(B % num_streams):
                sizes[i] += 1
            # Pre-allocate per-stream partials
            partials = []

            for sz, st in zip(sizes, streams):
                if sz == 0:
                    continue
                with th.cuda.stream(st), th.inference_mode():
                    # RNG & sampling on-device in the stream
                    U = th.rand((sz, 3), generator=g, device=dev, dtype=th.float32)
                    Y = U * (highs_t - lows_t) + lows_t

                    logdetG, kappa = _vec_hess_logV(
                        xp, Y, cfg.centers, alpha, beta, device=dev, jitter=jitter
                    )
                    w = th.exp(0.5 * logdetG)

                    k_norm = th.tanh(kappa)
                    f2 = th.clamp(1.0 + lam2 * k_norm, 0.5, 1.5)
                    f3 = th.clamp(1.0 / (1.0 + nu3 * th.clamp(k_norm, 0.0)), 0.3, 1.0)

                    # Compute partial sums in-stream; keep fp64
                    partials.append((
                        th.sum(w, dtype=th.float64),
                        th.sum(w*w, dtype=th.float64),
                        th.sum(w * 1.0, dtype=th.float64),
                        th.sum(w * f2,  dtype=th.float64),
                        th.sum(w * f3,  dtype=th.float64),
                        th.sum(w * (1.0**2), dtype=th.float64),
                        th.sum(w * (f2**2),  dtype=th.float64),
                        th.sum(w * (f3**2),  dtype=th.float64),
                        sz
                    ))

            # Wait for all streams to finish this outer batch, then reduce
            th.cuda.synchronize(device=dev)
            for ps in partials:
                _w_sum, _w_sq, _w1, _w2, _w3, _w1sq, _w2sq, _w3sq, _sz = ps
                w_sum   += _w_sum
                w_sqsum += _w_sq
                w1_sum  += _w1
                w2_sum  += _w2
                w3_sum  += _w3
                w1_sq   += _w1sq
                w2_sq   += _w2sq
                w3_sq   += _w3sq
                total   += _sz

        # Final stats on device -> host
        w_sum_c, w_sqsum_c = float(w_sum), float(w_sqsum)
        I1 = float(w1_sum / w_sum); I2 = float(w2_sum / w_sum); I3 = float(w3_sum / w_sum)
        neff = (w_sum_c**2) / max(w_sqsum_c, 1e-12)
        denom = max(neff**0.5, 1.0)

        e1 = float(((w1_sq / w_sum) - I1*I1).clamp_min(0.0).sqrt() / denom)
        e2 = float(((w2_sq / w_sum) - I2*I2).clamp_min(0.0).sqrt() / denom)
        e3 = float(((w3_sq / w_sum) - I3*I3).clamp_min(0.0).sqrt() / denom)

        return StiffnessResult(
            I={"U1": I1, "SU2": I2, "SU3": I3},
            err={"U1": e1, "SU2": e2, "SU3": e3},
            samples_used=int(total),
        )

    # ---------------- NumPy (CPU) with thread pool ----------------
    rng = np.random.default_rng(getattr(cfg, "rng_seed", None))
    pts_cpu = (rng.random((N, 3), dtype=np.float32) * (highs - lows) + lows).astype(np.float32)

    # choose workers
    if num_workers <= 0:
        try:
            import os
            num_workers = max(1, min(os.cpu_count() or 1, 8))
        except Exception:
            num_workers = 4

    def _chunk_reduce(Y: np.ndarray):
        logdetG, kappa = _vec_hess_logV(np, Y, cfg.centers, alpha, beta, device=None, jitter=jitter)
        w = np.exp(0.5 * logdetG)
        if mode == "raw":
            # curvature κ = tr(∇² log V); build Hessian of log V and take trace
            # (reuse your _hess_logV in a batched way if you have it; if not,
            # compute a per-point Hess trace with current centers in a small loop per batch size ~1e5 is OK on GPU)
            # For a quick improvement: get κ from Hessian trace scalar you already return in your scalar code.
            # Hessian-logV path (raw)
            logdetG, kappa_hess_trace = _vec_hessian_logV_weight_and_kappa(
                xp, Y, cfg.centers, alpha, beta, device=dev, jitter=jitter
            )
            w = xp.exp(0.5 * logdetG)

            # Use concavity magnitude for SU(3) so it is NOT constant:
            # kappa_mag = max(0, - trace(∇² log V))
            if xp.__name__ == "torch":
                kappa_mag = (-kappa_hess_trace).clamp_min(0.0)
            else:
                kappa_mag = np.clip(-kappa_hess_trace, 0.0, None)

            # Unclipped raw factors (slope-sensitive)
            f2 = 1.0 + lam2 * kappa_hess_trace     # SU(2): signed response
            f3 = 1.0 / (1.0 + nu3 * kappa_mag)     # SU(3): suppression from concavity magnitude
            # Fisher weight: prefer Hessian-logV metric (not grad-outer) in raw mode
            # i.e., compute G = +∇² log V + jitter I and use slogdet(G)
        else:
            # current robust path (tanh+clips, grad-outer metric)
            k_norm = xp.tanh(kappa)  # your surrogate
            f2 = xp.clip(1.0 + lam2 * k_norm, 0.5, 1.5)
            f3 = xp.clip(1.0 / (1.0 + nu3 * xp.clip(k_norm, 0.0, None)), 0.3, 1.0)
        return (
            float(w.sum()),
            float((w*w).sum()),
            float((w * 1.0).sum()),
            float((w * f2).sum()),
            float((w * f3).sum()),
            float((w * (1.0**2)).sum()),
            float((w * (f2**2)).sum()),
            float((w * (f3**2)).sum()),
            int(len(Y)),
        )

    wU1_sum = 0.0; wSU2_sum = 0.0; wSU3_sum = 0.0
    wU1_sq  = 0.0; wSU2_sq  = 0.0; wSU3_sq  = 0.0
    w_sum   = 0.0; w_sq_sum = 0.0
    total   = 0

    # Submit each outer batch as parallel tasks; inside each, we still benefit from vectorized numpy.
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            futures.append(ex.submit(_chunk_reduce, pts_cpu[start:end]))
        for f in as_completed(futures):
            _w_sum, _w_sq, _w1, _w2, _w3, _w1sq, _w2sq, _w3sq, _sz = f.result()
            w_sum   += _w_sum
            w_sq_sum+= _w_sq
            wU1_sum += _w1
            wSU2_sum+= _w2
            wSU3_sum+= _w3
            wU1_sq  += _w1sq
            wSU2_sq += _w2sq
            wSU3_sq += _w3sq
            total   += _sz

    I1 = wU1_sum / max(w_sum, 1e-30)
    I2 = wSU2_sum / max(w_sum, 1e-30)
    I3 = wSU3_sum / max(w_sum, 1e-30)
    neff = (w_sum**2) / max(w_sq_sum, 1e-12)

    e1 = float(np.sqrt(max((wU1_sq/w_sum - I1*I1), 0.0) / max(neff, 1.0)))
    e2 = float(np.sqrt(max((wSU2_sq/w_sum - I2*I2), 0.0) / max(neff, 1.0)))
    e3 = float(np.sqrt(max((wSU3_sq/w_sum - I3*I3), 0.0) / max(neff, 1.0)))

    return StiffnessResult(
        I={"U1": float(I1), "SU2": float(I2), "SU3": float(I3)},
        err={"U1": float(e1), "SU2": float(e2), "SU3": float(e3)},
        samples_used=int(total),
    )

# ---------- utility & multi-run ----------

def _scale_centers_widths(centers, lam: float):
    out = []
    for x, y, z, A, s in centers:
        s_new = float(s) / float(lam)
        out.append([float(x), float(y), float(z), float(A), s_new])
    return out

def _eval_su3_once(gcfg, lam: float, bump_seed: int) -> float:
    # isolate one run for process-pool execution
    g = gcfg.__class__(**{**gcfg.__dict__})
    g.centers = _scale_centers_widths(gcfg.centers, float(lam))
    g.rng_seed = None if gcfg.rng_seed is None else int(gcfg.rng_seed) + int(bump_seed)
    from .stiffness import estimate_stiffness  # late import to avoid cycles in workers
    return float(estimate_stiffness(g).I["SU3"])

def scaling_exponent_I3_widths(gcfg, lams=(0.8,0.85,0.9,0.95,1.05,1.1,1.15,1.2), seeds=(0,1,2)) -> float:
    """
    t ≡ d ln I3 / d ln (1/s) at λ=1, via multi-point, multi-seed fit.
    Parallelizes (λ, seed) runs for throughput while preserving API.
    """
    import numpy as np

    # choose workers (CPU-bound orchestration; each run may use CPU or GPU internally)
    try:
        import os
        max_workers = max(1, min(os.cpu_count() or 1, 8))
    except Exception:
        max_workers = 4

    # map (lam, seed) in parallel
    lam_to_vals: Dict[float, List[float]] = {float(l): [] for l in lams}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2meta = {}
        for lam in lams:
            for bump in seeds:
                fut = ex.submit(_eval_su3_once, gcfg, float(lam), int(bump))
                fut2meta[fut] = (float(lam), int(bump))
        for fut in as_completed(fut2meta):
            lam, bump = fut2meta[fut]
            lam_to_vals[lam].append(float(fut.result()))

    xs, ys = [], []
    for lam, vals in lam_to_vals.items():
        xs.append(np.log(float(lam)))                 # ln(λ) = ln(1/s) change
        ys.append(np.log(float(np.mean(vals))))       # ln I3

    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope)
