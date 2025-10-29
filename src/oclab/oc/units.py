from __future__ import annotations
from typing import Dict

def m_tau(kappa: Dict[str,int], x: float, *,  # keep old signature, add knobs via kwargs in future calls
          prefactor: float = 1.0,
          kappa_agg: str = "sum",
          kappa_power: float = 0.5,
          x_exponent: float = 0.5) -> float:
    """
    Configurable OC mass scale m_tau = prefactor * (Agg_kappa(|kappa_a|))^kappa_power * x^{x_exponent}
    Agg choices: 'sum' (default), 'max', 'l2'.
    """
    vals = [abs(v) for v in kappa.values()] or [1.0]
    if kappa_agg == "max":
        agg = max(vals)
    elif kappa_agg == "l2":
        agg = sum(v*v for v in vals) ** 0.5
    else:
        agg = sum(vals)
    return float(prefactor * (agg ** kappa_power) * (x ** x_exponent))