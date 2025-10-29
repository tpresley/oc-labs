import numpy as np

def sym_pd_enforce(M: np.ndarray, eps: float=1e-9) -> np.ndarray:
    M = 0.5*(M+M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    return (V*w)@V.T

def logdet_pd(M: np.ndarray) -> float:
    L = np.linalg.cholesky(M)
    return 2.0*np.sum(np.log(np.diag(L)))
