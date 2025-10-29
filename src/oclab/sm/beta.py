import numpy as np
from typing import Tuple

# --- SU(3) QCD with nf thresholds (MS-bar) ---
def beta_qcd(alpha: float, nf: int, order: int = 2) -> float:
    """
    dα/dlnμ = -(β0/(2π)) α^2  - (β1/(4π^2)) α^3   with
    β0 = 11 - 2 nf / 3,   β1 = 102 - 38 nf / 3.
    """
    beta0 = 11.0 - 2.0*nf/3.0
    if order == 1:
        return - (beta0/(2.0*np.pi)) * alpha*alpha
    beta1 = 102.0 - 38.0*nf/3.0
    return - (beta0/(2.0*np.pi)) * alpha*alpha - (beta1/(4.0*np.pi*np.pi)) * alpha*alpha*alpha

# --- Full SM betas for (α1, α2, α3) with GUT-normalized U(1) ---
def beta_SM_alphas(alphas: Tuple[float, float, float], order: int = 2) -> Tuple[float, float, float]:
    """
    Standard Model gauge-coupling beta functions in terms of α_i = g_i^2 / (4π).
    GUT-normalized U(1): g1^2 = (5/3) gY^2.

    Returns (dα1/dlnμ, dα2/dlnμ, dα3/dlnμ) for a *single* scale where all SM fields are active.
    (SU(3) nf thresholds are handled in rgrun; here use full SM coefficients.)
    """
    α1, α2, α3 = alphas
    # one-loop vector b_i
    b1, b2, b3 = 41.0/10.0, -19.0/6.0, -7.0
    # two-loop matrix b_ij
    B = np.array([
        [199.0/50.0, 27.0/10.0, 44.0/5.0],
        [9.0/10.0,   35.0/6.0,  12.0     ],
        [11.0/10.0,  9.0/2.0,   -26.0    ],
    ], dtype=float)

    α = np.array([α1, α2, α3], dtype=float)

    # one-loop piece: (b_i / 2π) α_i^2
    dα = (np.array([b1, b2, b3]) / (2.0 * np.pi)) * (α * α)

    if order >= 2:
        # two-loop piece: (1 / 8π^2) * Σ_j b_ij α_i^2 α_j
        dα += (1.0 / (8.0 * np.pi**2)) * ((α * α) * (B @ α))

    return (float(dα[0]), float(dα[1]), float(dα[2]))