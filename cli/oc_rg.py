#!/usr/bin/env python3
from oclab.config import GeometryConfig, EWAnchor, OCParams, RGRun, realize_geometry_centers
from oclab.pipeline import geometry_to_couplings
import yaml, argparse
from oclab.viz.plotting import plot_alpha_running
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.yaml')
p.add_argument('--no-plot', action='store_true')
args = p.parse_args()

cfg = yaml.safe_load(open(args.config))
g   = GeometryConfig(**cfg['geometry'])
g = realize_geometry_centers(g)
ew  = EWAnchor(**cfg['ew_anchor'])
oc  = OCParams(**cfg['oc_params'])
rg  = RGRun(**cfg['rg'])

calib, hist = geometry_to_couplings(g, ew, oc, rg)
print(f"x = {calib.x:.6g}")  # print minimal summary
if not args.no_plot:
    fig = plot_alpha_running(hist.mu, hist.alpha1, hist.alpha2, hist.alpha3)
    plt.show()
