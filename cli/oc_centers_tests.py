#!/usr/bin/env python3
import argparse, yaml, numpy as np, sys
from copy import deepcopy
from oclab.config import GeometryConfig, EWAnchor, OCParams, RGRun, FRGScan, realize_geometry_centers
from oclab.pipeline import geometry_to_couplings, oc_gap_from_frg
from oclab.oc.centers import (
    gen_icosa, gen_dodeca, gen_octa, gen_tetra, gen_random_shell,
    random_rotation, apply_rotation, jitter_centers, symmetry_score,
)
from oclab.frg.flows import expected_kstar_toy
from oclab.io.datasets import write_csv

PDG_ALPHA3_MZ = 0.1181

def fit_su3_scale_on_centers(g, ew, oc, rg):
    def set_run(scale):
        oc_local = deepcopy(oc)
        oc_local.Ka_group_scale = dict(oc_local.Ka_group_scale or {}, SU3=float(scale))
        calib2, hist2 = geometry_to_couplings(g, ew, oc_local, rg)
        return float(hist2.alpha3[-1])
    s0 = (oc.Ka_group_scale or {}).get("SU3", 1.0)
    a0 = set_run(s0)
    s1 = s0 / (PDG_ALPHA3_MZ / a0)
    a1 = set_run(s1)
    a, b = (s0, a0), (s1, a1)
    for _ in range(8):
        (x0,y0),(x1,y1) = a,b
        if abs(y1-y0) < 1e-12: break
        x2 = x1 - (y1-PDG_ALPHA3_MZ)*(x1-x0)/(y1-y0)
        x2 = float(np.clip(x2, 0.2*s0, 5.0*s0))
        y2 = set_run(x2)
        a, b = b, (x2, y2)
        if abs(y2-PDG_ALPHA3_MZ) < 2e-5: break
    return float(b[0])

def case_report(name, g, ew, oc, rg, frg):
    calib, hist = geometry_to_couplings(g, ew, oc, rg)
    alpha0 = float(calib.alphas_mu0["SU3"])
    alpha3_MZ = float(hist.alpha3[-1])
    d_pct = 100.0*(alpha3_MZ - PDG_ALPHA3_MZ)/PDG_ALPHA3_MZ
    frg2 = deepcopy(frg); frg2.alpha_target = max(frg2.alpha_target, 1e9)
    frg_res, m_tau, C = oc_gap_from_frg(frg2, alpha0, ew.mu0, oc, calib.x)
    k_exp = expected_kstar_toy(alpha0, frg2.growth_c, frg2.kmax, frg2.eta_freeze, model=frg2.model)
    ln_err = abs(np.log(k_exp / frg_res.k_star)) if frg_res.k_star else float("inf")
    return {
        "name": name,
        "N": len(g.centers),
        "symmetry_score": symmetry_score(g.centers),
        "I1": calib.Ia["U1"],
        "I2": calib.Ia["SU2"],
        "I3": calib.Ia["SU3"],
        "x": calib.x,
        "alpha3_mu0": alpha0,
        "alpha3_MZ": alpha3_MZ,
        "d_pct_MZ": d_pct,
        "k_star": frg_res.k_star,
        "eta_at_kstar": getattr(frg_res, "reached_eta", None),
        "k_star_expected": k_exp,
        "ln_err": ln_err,
        "m_tau": m_tau,
        "C": C,
        "reason": getattr(frg_res, "reason", None)
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--jitter", type=float, default=None, help="Relative jitter σ/R for Icosa (e.g., 0.02 = 2%%).")
    p.add_argument("--rotations", type=int, default=0, help="Rotation samples for marginalization (0=skip).")
    p.add_argument("--random-seeds", type=str, default="42,777", help="Comma seeds for random shells.")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    g   = realize_geometry_centers(GeometryConfig(**cfg["geometry"]))
    ew  = EWAnchor(**cfg["ew_anchor"])
    oc  = OCParams(**cfg["oc_params"])
    rg  = RGRun(**cfg["rg"])
    frg = FRGScan(**cfg["frg_scan"])

    # 1) Calibrate SU(3) scale once on Icosa
    g_ico = deepcopy(g); g_ico.centers, _, _ = gen_icosa(g.domain)
    oc = deepcopy(oc)
    oc.Ka_group_scale = dict(oc.Ka_group_scale or {}, SU3=fit_su3_scale_on_centers(g_ico, ew, oc, rg))

    # 2) Core test set (duality + lower symmetry + random)
    rows = []
    def add(name, centers):
        g2 = deepcopy(g); g2.centers = centers
        rows.append(case_report(name, g2, ew, deepcopy(oc), rg, frg))

    add("Icosa (calib)", gen_icosa(g.domain)[0])
    add("Dodeca",       gen_dodeca(g.domain)[0])
    add("Octa",         gen_octa(g.domain)[0])
    add("Tetra",        gen_tetra(g.domain)[0])
    for s in [int(x) for x in args.random_seeds.split(",") if x.strip()]:
        add(f"Random12_{s}", gen_random_shell(g.domain, N=12, seed=s)[0])

    # 3) Jitter sensitivity (optional)
    if args.jitter:
        base = gen_icosa(g.domain)[0]
        for rel in [args.jitter/2, args.jitter, 2*args.jitter]:
            Cj = jitter_centers(base, rel_sigma=float(rel), seed=123)
            add(f"Icosa+jitter{rel:.3%}", Cj)

    # 4) Rotation marginalization (optional)
    if args.rotations > 0:
        base = gen_icosa(g.domain)[0]
        vals = []
        for k in range(args.rotations):
            R = random_rotation(seed=1000+k)
            Cr = apply_rotation(base, R)
            rep = case_report(f"Icosa@R{k}", deepcopy(g).__class__(**{**g.__dict__, "centers": Cr}), ew, deepcopy(oc), rg, frg)
            vals.append(rep["alpha3_MZ"])
        mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append({
            "name": "Icosa rotation marginalization",
            "N": len(base), "symmetry_score": symmetry_score(base),
            "alpha3_MZ_mean": mean, "alpha3_MZ_std": std,
            "d_pct_MZ_mean": 100.0*(mean - PDG_ALPHA3_MZ)/PDG_ALPHA3_MZ
        })

    # Save & pretty print
    headers = list(rows[0].keys())
    path = write_csv([[r.get(h, "") for h in headers] for r in rows], headers, "centers_uniqueness_tests.csv")
    print(f"Wrote {path}")
    # Console table (short)
    cols = ["name","N","alpha3_MZ","d_pct_MZ","k_star","eta_at_kstar","ln_err","C","x","I3","symmetry_score"]
    print("\n" + " | ".join(cols))
    for r in rows:
        if "alpha3_MZ" in r:
            print(" | ".join(str(r.get(k,"")) for k in cols))
        else:
            print(f"{r['name']} | N={r['N']} | alpha3_MZ_mean={r['alpha3_MZ_mean']:.6f} ± {r['alpha3_MZ_std']:.6f} "
                  f"| Δ%={r['d_pct_MZ_mean']:.3f}")

if __name__ == "__main__":
    main()
