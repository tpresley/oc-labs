from __future__ import annotations
from typing import Tuple
import numpy as np
from .config import GeometryConfig, EWAnchor, OCParams, RGRun, FRGScan
from .frg.projectors import YMProjParams
from .oc.calibration import calibrate_x_and_couplings
from .sm.rgrun import run_couplings, RGHistory
from .oc.units import m_tau
from .frg.flows import scan_freeze, scan_bf_projector, FRGResult

def geometry_to_couplings(
    gcfg: GeometryConfig, ew: EWAnchor, ocp: OCParams, rg: RGRun
) -> tuple:
    calib = calibrate_x_and_couplings(gcfg, ew, ocp)
    hist  = run_couplings(calib.alphas_mu0, ew.mu0, rg, ew.scheme)
    return calib, hist

def oc_gap_from_frg(
    frg: FRGScan, alpha0: float, mu0: float, ocp: OCParams, x: float
) -> tuple:
    frg_res = scan_freeze(alpha0, mu0, frg)
    # frg_res = scan_bf_projector(alpha0, mu0, frg, YMProjParams())
    m_tau_val = m_tau(
        ocp.kappa, x,
        prefactor=ocp.m_tau_prefactor,
        kappa_agg=ocp.m_tau_kappa_agg,
        kappa_power=ocp.m_tau_kappa_power,
        x_exponent=ocp.m_tau_x_exponent,
    )
    C = (frg_res.k_star / m_tau_val) if frg_res.k_star else float('nan')
    return frg_res, m_tau_val, C
