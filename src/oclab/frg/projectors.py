# Minimal background-field YM projector (Landau gauge), Litim-type thresholds
from dataclasses import dataclass
import math

@dataclass
class YMProjParams:
    Nc: int = 3
    # regulator-dependent “denominator” slopes; adjust only if you change regulator
    a_gh: float = 1.0/5.0     # in L_gh = 1/(1 - a_gh * eta_c)
    a_gl: float = 1.0/6.0     # in L_gl = 1/(1 - a_gl * eta_A)
    # UV matching coefficients (set by 1-loop)
    C_gh: float = 2.0         # ghost loop weight
    C_gl: float = 1.0         # gluon loop weight

def L_gh(eta_c: float, a_gh: float) -> float:
    return 1.0 / (1.0 - a_gh * eta_c)

def L_gl(eta_A: float, a_gl: float) -> float:
    return 1.0 / (1.0 - a_gl * eta_A)

def eta_A_projector(alpha: float, Nf: int, etaA_guess: float = 0.0, eta_c: float = 0.0,
                    p: YMProjParams = YMProjParams()) -> float:
    """
    Solve implicit eta_A = (alpha Nc / 4π)[ C_gh L_gh(eta_c) - C_gl L_gl(eta_A) ]
    with UV normalization matching β0 = 11 - 2Nf/3 in alpha→0.
    """
    Nc = p.Nc
    # Fix UV normalization by rescaling (C_gh - C_gl) so linear term equals -β0
    beta0 = 11.0 - 2.0*Nf/3.0
    # Effective prefactor for small eta's: A * alpha, with A chosen to match -β0
    # At eta≈0 : L_gh≈1, L_gl≈1  ⇒ bracket ≈ (C_gh - C_gl)
    A = (-beta0) / (Cgh_minus_Cgl := (p.C_gh - p.C_gl))
    pref = (alpha * Nc / (4.0*math.pi)) * A

    eta = etaA_guess
    for _ in range(6):  # few fixed-point iterations suffice
        rhs = pref * ( p.C_gh * L_gh(eta_c, p.a_gh) - p.C_gl * L_gl(eta, p.a_gl) )
        # Clamp to a safe band to avoid runaway; you can relax this if needed
        eta = max(-3.0, min(0.0, rhs))
    return eta