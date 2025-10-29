from __future__ import annotations
import numpy as np
from typing import Sequence
from ..math.linalg import sym_pd_enforce, logdet_pd

def _split_center(c):
    """
    Accept center either as [x,y,z] or [x,y,z,A,s].
    Returns (xyz[3], A, s).
    """
    c = np.asarray(c, float)
    if c.shape[0] == 3:
        return c, 1.0, 1.0
    if c.shape[0] == 5:
        xyz, A, s = c[:3], c[3], c[4]
        s = float(s) if s != 0 else 1.0  # guard
        return xyz, float(A), float(s)
    raise ValueError("Each center must be length 3 or 5: [x,y,z] or [x,y,z,A,s].")

def potential(y: np.ndarray,
              centers: Sequence[Sequence[float]],
              alpha_metric: float,
              beta_floor: float) -> float:
    """
    Softened multi-center potential with optional per-center amplitude & width:
        V(y) = Σ_c A_c * ( (||y-c||^2)/s_c^2 + beta_floor )^(-alpha)
    """
    y = np.asarray(y, dtype=float)
    val = 0.0
    for c in centers:
        xyz, A, s = _split_center(c)
        r2 = np.sum((y - xyz)**2)
        base = (r2 / (s*s)) + beta_floor
        val += A * (base**(-alpha_metric))
    return float(val)

def _grad_logV(y: np.ndarray, centers, alpha, beta) -> np.ndarray:
    """
    ∇ log V = (1/V) ∇V with V = Σ (r2+β)^(-α).
    With width s and amplitude A:
      V = Σ A * (r2/s^2 + β)^(-α)
      ∇V = Σ A * [ -α (r2/s^2 + β)^(-α-1) * 2 (y-c)/s^2 ].
    """
    y = np.asarray(y, float)
    d = len(y)
    V = 0.0
    g = np.zeros(d, float)
    for c in centers:
        xyz, A, s = _split_center(c)
        dvec = y - xyz
        r2 = np.dot(dvec, dvec)
        base = (r2/(s*s) + beta)
        term = A * base**(-alpha)
        V   += term
        g   += A * (-alpha) * base**(-alpha - 1.0) * (2.0/(s*s)) * dvec
    if V <= 0.0:
        return np.zeros_like(g)
    return g / V

def _hess_logV(y: np.ndarray, centers, alpha, beta) -> np.ndarray:
    """
    Hessian of log V via: ∇^2 log V = (∇^2 V)/V - (∇V ∇V^T)/V^2.
    """
    y = np.asarray(y, float)
    d = len(y)
    V = 0.0
    gradV = np.zeros(d, float)
    hessV = np.zeros((d, d), float)
    for c in centers:
        xyz, A, s = _split_center(c)
        dvec = y - xyz
        r2 = np.dot(dvec, dvec)
        base = (r2/(s*s) + beta)
        # scalar pieces with amplitude and width
        t0 = A * base**(-alpha)                               # contributes to V
        t1 = A * (-alpha) * base**(-alpha - 1.0) * (2.0/(s*s)) # ∂/∂y of base^{-α} × dvec
        t2 = A * (-alpha) * (-alpha - 1.0) * base**(-alpha - 2.0) * (1.0/(s*s))**2
        V     += t0
        gradV += t1 * dvec
        # Hessian term: ∂/∂y [ t1 dvec ] = t1 I + (∂t1/∂y) dvec^T
        # Using product form gives:  t1 I  +  2 * t2 * (dvec dvec^T)
        hessV += t1 * np.eye(d) + 2.0 * t2 * np.outer(dvec, dvec)
    if V <= 0.0:
        return np.eye(d)
    return (hessV / V) - np.outer(gradV, gradV) / (V * V)

def fisher_metric(y: np.ndarray,
                  V: float,
                  alpha_metric: float,
                  beta_floor: float,
                  *,
                  centers=None,
                  model: str = "hessian_logV",
                  jitter: float = 1e-9) -> np.ndarray:
    """
    Fisher-like metric options:
    - 'hessian_logV':   G = +∇^2 log V (enforced PD with jitter)
    - 'grad_outer'  :   G = (∇ log V)(∇ log V)^T + jitter I
    """
    if centers is None:
        raise ValueError("centers must be provided for fisher_metric")
    d = len(y)
    if model == "hessian_logV":
        G = _hess_logV(y, centers, alpha_metric, beta_floor)
        # numerical PD enforcement (shift minimum eigenvalue)
        G = 0.5*(G+G.T) + jitter*np.eye(d)
        return sym_pd_enforce(G, eps=jitter)
    # grad-outer model
    g = _grad_logV(y, centers, alpha_metric, beta_floor)
    G = np.outer(g, g) + jitter*np.eye(d)
    return sym_pd_enforce(G, eps=jitter)

def curvature_proxy(y: np.ndarray, centers, alpha, beta) -> float:
    """
    Curvature proxy κ(y) := Δ log V = tr[∇^2 log V].
    """
    H = _hess_logV(np.asarray(y, float), centers, alpha, beta)
    return float(np.trace(H))
