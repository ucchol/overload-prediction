"""Separate recomputation of the H3 observed rows of scripts/individual_level.py (point values).
Own growth per participant = OLS slope of the participant's miss rate on log2 set size, using numpy.polyfit."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from overload.experiments.overload_premonition import groups
G = groups({"reutskaja": "data/reutskaja", "snack": "data/snack"})
ref = json.load(open("results/individual_level/summary.json"))
worst = 0.0
for name in ("snack", "reutskaja"):
    (panel,), P = G[name]
    v = panel.values; top = np.nanmax(v, axis=1); ch = v[np.arange(len(v)), panel.choice]
    miss = (ch < top).astype(float); sizes = np.unique(panel.n)
    g = []
    for p in np.unique(panel.participant):
        r = panel.participant == p
        rates = [miss[r & (panel.n == s)].mean() for s in sizes if np.any(r & (panel.n == s))]
        if len(rates) == len(sizes):
            g.append(np.polyfit(np.log2(sizes), rates, 1)[0])
    g = np.array(g)
    for k, val in ((f"H3|{name}|share_own_growth_positive", 100 * np.mean(g > 0)), (f"H3|{name}|median_own_growth", np.median(g))):
        d = abs(val - ref[k]["point"]); worst = max(worst, d); print(k, round(val, 6), round(ref[k]["point"], 6), d)
print("max abs difference", worst)
