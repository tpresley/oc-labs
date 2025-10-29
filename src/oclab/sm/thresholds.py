def nf_piecewise(mu: float, scheme: str='Q3') -> int:
    # Simple flavor thresholds in GeV
    # (placeholders; tune to your preferred scheme)
    if mu >= 173: return 6
    if mu >= 4.18: return 5
    if mu >= 1.27: return 4
    if mu >= 0.095: return 3
    return 3
