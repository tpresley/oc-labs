#!/usr/bin/env python3
from oclab.config import GeometryConfig
from oclab.oc.stiffness import estimate_stiffness
import yaml, argparse, json

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.yaml')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g = GeometryConfig(**cfg['geometry'])
res = estimate_stiffness(g)
print(json.dumps({'I': res.I, 'err': res.err, 'samples': res.samples_used}, indent=2))
