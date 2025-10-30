#!/usr/bin/env python3
import argparse, yaml, json
from oclab.config import GeometryConfig, EWAnchor, OCParams, RGRun, realize_geometry_centers
from oclab.oc.normalization import compute_K0_by_calibration, ActionMatchSpec, compute_K0_from_components

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/default.yaml")
p.add_argument("--alpha3-MZ", type=float, default=0.1181, help="Target αs(MZ) for one-time K0 calibration.")
p.add_argument("--components", action="store_true", help="Compose K0 from action/trace/Jacobian conventions (no data).")
p.add_argument("--trace", default="tr(TT)=1/2", choices=["tr(TT)=1/2","tr(TT)=1"])
p.add_argument("--jacobian", default="detG_sqrt", choices=["unity","detG_sqrt"])
p.add_argument("--extra", type=float, default=1.0)
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g = realize_geometry_centers(GeometryConfig(**cfg["geometry"]), raw_geometry=cfg["geometry"])
ew = EWAnchor(**cfg["ew_anchor"])
oc = OCParams(**cfg["oc_params"])
rg = RGRun(**cfg["rg"])

if args.components:
    spec = ActionMatchSpec(trace_convention=args.trace, jacobian=args.jacobian, extra_factor=args.extra)
    K0 = compute_K0_from_components(spec, g=g)
    mode = "components"
else:
    K0 = compute_K0_by_calibration(g, ew, oc, rg, alpha3_target_MZ=args.alpha3_MZ)
    mode = "calibrate"

print(json.dumps({"mode": mode, "Ka_slope_K0": K0}, indent=2))
print(f"\nAdd to oc_params:\n  Ka_slope_K0: {K0:.6f}")