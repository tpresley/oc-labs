import numpy as np

def _clip_eta(x: float) -> float:
    # keep strictly within (-1, 0]; avoids pathological values and exact -1
    return float(np.clip(x, -0.999, 0.0))

def eta_minimal(alpha: float) -> float:
    a = float(alpha)
    denom = 1.0 + a
    if not np.isfinite(a) or denom <= 0.0 or not np.isfinite(denom):
        return -0.999
    val = -0.6 * (a / denom)
    return _clip_eta(val)

def eta_projected(alpha: float) -> float:
    a = float(alpha)
    denom = 1.0 + 0.5 * a
    if not np.isfinite(a) or denom <= 0.0 or not np.isfinite(denom):
        return -0.999
    val = -0.9 * (a / denom)
    return _clip_eta(val)