from __future__ import annotations
import numpy as np
from typing import List, Tuple

def _radius_width_from_domain(domain) -> Tuple[float, float]:
    extents = np.array([hi - lo for lo, hi in domain], float)
    mean_half = float(np.mean(0.5 * extents))
    mean_full = float(np.mean(extents))
    R = 0.382 * mean_half      # φ^-2 × mean half-extent
    s = 0.12  * mean_full      # width ≈ 0.30 for [-2.5,2.5]^3
    return R, s

def _normalize_shell(verts: np.ndarray, R: float) -> np.ndarray:
    verts = np.asarray(verts, float)
    return (verts / np.linalg.norm(verts, axis=1, keepdims=True)) * R

def gen_icosa(domain, A: float = 1.0):
    φ = (1 + 5**0.5) / 2
    base = [
        (0,  1,  φ), (0, -1,  φ), (0,  1, -φ), (0, -1, -φ),
        ( 1,  φ, 0), (-1,  φ, 0), ( 1, -φ, 0), (-1, -φ, 0),
        ( φ, 0,  1), ( φ, 0, -1), (-φ, 0,  1), (-φ, 0, -1),
    ]
    R, s = _radius_width_from_domain(domain)
    verts = _normalize_shell(base, R)
    C = [[float(x), float(y), float(z), float(A), float(s)] for x, y, z in verts]
    return C, R, s

def gen_dodeca(domain, A: float = 1.0):
    φ = (1 + 5**0.5) / 2.0
    invφ = 1.0 / φ
    base = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                base.append((sx, sy, sz))
    for sy in (-1, 1):
        for sz in (-1, 1):
            base.append((0.0, sy*invφ, sz*φ))
    for sx in (-1, 1):
        for sy in (-1, 1):
            base.append((sx*invφ, sy*φ, 0.0))
    for sx in (-1, 1):
        for sz in (-1, 1):
            base.append((sx*φ, 0.0, sz*invφ))
    base = np.array(sorted({(float(x), float(y), float(z)) for (x, y, z) in base}))
    assert base.shape[0] == 20
    R, s = _radius_width_from_domain(domain)
    verts = _normalize_shell(base, R)
    C = [[float(x), float(y), float(z), float(A), float(s)] for x, y, z in verts]
    return C, R, s

def gen_octa(domain, A: float = 1.0):
    base = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    R, s = _radius_width_from_domain(domain)
    verts = _normalize_shell(base, R)
    C = [[float(x), float(y), float(z), float(A), float(s)] for x, y, z in verts]
    return C, R, s

def gen_tetra(domain, A: float = 1.0):
    base = [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]
    R, s = _radius_width_from_domain(domain)
    verts = _normalize_shell(base, R)
    C = [[float(x), float(y), float(z), float(A), float(s)] for x, y, z in verts]
    return C, R, s

def gen_random_shell(domain, N: int = 12, A: float = 1.0, seed: int = 123):
    R, s = _radius_width_from_domain(domain)
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(N, 3))
    u = (u / np.linalg.norm(u, axis=1, keepdims=True)) * R
    C = [[float(x), float(y), float(z), float(A), float(s)] for x, y, z in u]
    return C, R, s

# Utilities for experiments
def random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0: Q[:, 0] *= -1
    return Q

def apply_rotation(centers: List[List[float]], R: np.ndarray) -> List[List[float]]:
    out = []
    for x, y, z, A, s in centers:
        v = R @ np.array([x, y, z], float)
        out.append([float(v[0]), float(v[1]), float(v[2]), float(A), float(s)])
    return out

def jitter_centers(centers: List[List[float]], rel_sigma: float, seed: int = 0) -> List[List[float]]:
    rng = np.random.default_rng(seed)
    out = []
    for x, y, z, A, s in centers:
        r = np.linalg.norm([x, y, z])
        sig = rel_sigma * r
        dx, dy, dz = rng.normal(scale=sig, size=3)
        out.append([x+dx, y+dy, z+dz, A, s])
    return out

def symmetry_score(centers: List[List[float]]) -> float:
    P = np.array([[c[0], c[1], c[2]] for c in centers], float)
    D = np.linalg.norm(P[None, :, :] - P[:, None, :], axis=2)
    d = D[np.triu_indices_from(D, k=1)]
    return float(np.var(d) / (np.mean(d)**2 + 1e-12))
