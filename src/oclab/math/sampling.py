import numpy as np

def uniform_box(n: int, lows, highs, rng: np.random.Generator):
    lows = np.array(lows); highs = np.array(highs)
    return rng.random((n, len(lows)))*(highs-lows)+lows

def latin_hypercube(n: int, lows, highs, rng: np.random.Generator):
    lows = np.array(lows); highs = np.array(highs)
    d = len(lows)
    cut = np.linspace(0, 1, n+1)
    u = rng.random((n, d))
    # stratified in each dimension
    a = cut[:n]; b = cut[1:n+1]
    pts01 = u*(b - a)[:, None] + a[:, None]
    # random permutation per dimension
    for j in range(d):
        rng.shuffle(pts01[:, j])
    return pts01*(highs-lows) + lows