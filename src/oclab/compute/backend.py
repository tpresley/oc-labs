from __future__ import annotations
import os

class Backend:
    name: str
    xp: any          # array lib (numpy or torch)
    device: any      # torch device or None
    dtype: any       # xp.dtype

def get_backend(kind: str, dtype: str = "float32") -> Backend:
    import numpy as _np
    b = Backend()
    b.dtype = _np.float32 if dtype == "float32" else _np.float64
    if kind == "torch_mps" or kind == "torch_cpu":
        import torch as _th
        dev = _th.device("mps") if (kind == "torch_mps" and _th.backends.mps.is_available()) else _th.device("cpu")
        b.name = f"torch[{dev.type}]"
        b.xp = _th
        b.device = dev
        return b
    # default: numpy CPU
    b.name = "numpy[cpu]"
    b.xp = _np
    b.device = None
    return b

def to_device(xp, x, device):
    if hasattr(xp, "asarray"):
        arr = xp.asarray(x)
        if device is not None and hasattr(arr, "to"):
            return arr.to(device)
        return arr
    return x  # numpy
