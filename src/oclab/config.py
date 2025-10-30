from dataclasses import dataclass
from typing import Literal, Dict, List, Optional, Any

# --- numeric coercion helpers ---
def _to_float(x) -> float:
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)
def _to_int(x) -> int:
    return int(float(x))

@dataclass
class GeometryConfig:
    centers: List[List[float]]
    alpha_metric: float
    beta_floor: float
    lambda_su2: float
    nu_su3: float
    domain: List[List[float]]
    n_samples: int
    rng_seed: Optional[int] = None
    metric_model: Literal["hessian_logV","grad_outer"] = "grad_outer"
    weight_form: Literal["sqrt_detG"] = "sqrt_detG"
    jitter: float = 1e-9
    generator: Optional[Dict[str, Any]] = None
    compute: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        self.alpha_metric = _to_float(self.alpha_metric)
        self.beta_floor   = _to_float(self.beta_floor)
        self.lambda_su2   = _to_float(self.lambda_su2)
        self.nu_su3       = _to_float(self.nu_su3)
        self.n_samples    = _to_int(self.n_samples)
        self.jitter       = _to_float(self.jitter)
        # centers/domain left as-is (lists of lists)

@dataclass
class EWAnchor:
    mu0: float                 # GeV
    alpha_em_mu0: float
    sin2_thetaW_mu0: float
    scheme: Literal["Q0","Q3"] = "Q3"
    def __post_init__(self):
        self.mu0               = _to_float(self.mu0)
        self.alpha_em_mu0      = _to_float(self.alpha_em_mu0)
        self.sin2_thetaW_mu0   = _to_float(self.sin2_thetaW_mu0)

@dataclass
class OCParams:
    kappa: Dict[str,int]       # {"U1":..., "SU2":..., "SU3":...}
    weights: Dict[str,float]   # W_a
    n_color_discrete: int      # e.g., 2
    lock_modes: Literal["EW","none"] = "EW"
    Ka_mode: Literal["fixed","slope"] = "fixed" 
    # --- NEW: Ka normalization knobs (for 1/α_a = Ka(a) * x * I_a) ---
    Ka_prefactor: float = 1.0              # global scale multiplier
    Ka_power_kappa: float = 2.0            # use |kappa_a|^power (default square)
    Ka_divide_by_n: bool = True            # divide by discrete color multiplicity
    Ka_group_scale: Dict[str, float] | None = None  # per-group extra factor, e.g. {"SU3": 1.0}
    # --- NEW: m_tau normalization knobs ---
    m_tau_prefactor: float = 1.0           # overall units/normalization
    m_tau_kappa_agg: Literal["sum","max","l2"] = "sum"  # how to aggregate kappa over groups
    m_tau_kappa_power: float = 0.5         # exponent applied to kappa aggregate
    m_tau_x_exponent: float = 0.5          # exponent of x in m_tau
    def __post_init__(self):
        self.n_color_discrete = _to_int(self.n_color_discrete)
        # coerce numeric weights just in case
        self.weights = {k: _to_float(v) for k, v in self.weights.items()}
        self.Ka_prefactor   = _to_float(self.Ka_prefactor)
        self.Ka_power_kappa = _to_float(self.Ka_power_kappa)
        self.m_tau_prefactor    = _to_float(self.m_tau_prefactor)
        self.m_tau_kappa_power  = _to_float(self.m_tau_kappa_power)
        self.m_tau_x_exponent   = _to_float(self.m_tau_x_exponent)

@dataclass
class RGRun:
    mu_target: float
    loop_order: Literal[1,2] = 2
    n_steps: int = 2000
    def __post_init__(self):
        self.mu_target = _to_float(self.mu_target)
        self.loop_order = _to_int(self.loop_order)
        self.n_steps = _to_int(self.n_steps)

@dataclass
class FRGScan:
    kmin: float
    kmax: float
    n_k: int
    alpha_target: float = 1.0
    model: Literal["minimal","projected"] = "projected"
    growth_c: float = 5.0
    eta_freeze: float = 0.99
    alpha_cap: float = 5.0   # NEW

    def __post_init__(self):
        self.kmin         = _to_float(self.kmin)
        self.kmax         = _to_float(self.kmax)
        self.n_k          = _to_int(self.n_k)
        self.alpha_target = _to_float(self.alpha_target)
        self.growth_c     = _to_float(self.growth_c)
        self.eta_freeze   = _to_float(self.eta_freeze)
        self.alpha_cap    = _to_float(self.alpha_cap)

def realize_geometry_centers(gcfg: GeometryConfig, raw_geometry: Dict[str, Any] | None = None) -> GeometryConfig:
    """If a generator is present, synthesize centers procedurally."""
    gen = (raw_geometry or {}).get("generator") or getattr(gcfg, "generator", None) or {}
    if not gen: return gcfg
    from .oc.centers import gen_icosa, gen_dodeca, gen_octa, gen_tetra, gen_random_shell
    kind = (gen.get("type") or "").lower()
    g2 = GeometryConfig(**{**gcfg.__dict__})
    if kind == "icosa":
        C, _, _ = gen_icosa(gcfg.domain)
    elif kind == "dodeca":
        C, _, _ = gen_dodeca(gcfg.domain)
    elif kind == "octa":
        C, _, _ = gen_octa(gcfg.domain)
    elif kind == "tetra":
        C, _, _ = gen_tetra(gcfg.domain)
    elif kind == "random":
        C, _, _ = gen_random_shell(gcfg.domain, N=int(gen.get("N", 12)), seed=int(gen.get("seed", 0)))
    else:
        raise ValueError(f"Unknown geometry.generator.type: {kind}")
    g2.centers = C
    return g2

