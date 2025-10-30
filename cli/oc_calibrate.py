#!/usr/bin/env python3
from oclab.config import GeometryConfig, EWAnchor, OCParams, realize_geometry_centers
from oclab.oc.calibration import calibrate_x_and_couplings
import yaml, argparse, json

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.yaml')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g = GeometryConfig(**cfg['geometry'])
g = realize_geometry_centers(g)
ew = EWAnchor(**cfg['ew_anchor'])
oc = OCParams(**cfg['oc_params'])
res = calibrate_x_and_couplings(g, ew, oc)
print(json.dumps({'x': res.x, 'alphas_mu0': res.alphas_mu0, 'Ia': res.Ia, 'Ia_err': res.Ia_err}, indent=2))
