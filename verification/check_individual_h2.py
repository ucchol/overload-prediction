"""Separate recomputation of the H2 rows of scripts/individual_level.py (point values).
Uses the package loaders only. Fits by Nelder-Mead on a likelihood written here (no gradient, no package fit)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "src")
from overload.experiments import h2_return as H2

def ll_rows(lam, beta, b, cue):
    Z = b.cue[cue] if cue else 0.0 * b.V
    U = lam * b.V + beta * Z
    m = U.max(axis=1, keepdims=True)
    lp = U - m - np.log(np.exp(U - m).sum(axis=1, keepdims=True))
    return lp[np.arange(b.choice.size), b.choice]

def fit(blocks, cue):
    if cue is None:
        f = lambda x: -sum(ll_rows(abs(x[0]), 0.0, b, None).sum() for b in blocks)
        best = min((minimize(f, [s], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000}) for s in (0.5, 2.0)), key=lambda r: r.fun)
        return abs(best.x[0]), 0.0
    f = lambda x: -sum(ll_rows(abs(x[0]), x[1], b, cue).sum() for b in blocks)
    best = min((minimize(f, s, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 8000}) for s in ((0.5, 0.0), (0.5, 2.0), (0.1, -1.0))), key=lambda r: r.fun)
    return abs(best.x[0]), best.x[1]

ref = json.load(open("results/individual_level/summary.json"))
data = {"snack": H2.load_snack_gaze(Path("data/snack"), Path("data/snack_gaze")),
        "reutskaja": H2.load_reutskaja_gaze(Path("data/reutskaja"))}
worst = 0.0
for name, (blocks, P) in data.items():
    tot = {m: np.zeros(P) for m in ("CL", "return", "dwell")}; cnt = np.zeros(P)
    for k, b in enumerate(blocks):
        others = [x for j, x in enumerate(blocks) if j != k]
        for m in tot:
            cue = None if m == "CL" else m
            lam, beta = fit(others, cue)
            np.add.at(tot[m], b.person, ll_rows(lam, beta, b, cue))
        np.add.at(cnt, b.person, 1.0)
    seen = cnt > 0
    sc = {m: tot[m][seen] / cnt[seen] for m in tot}
    for k, val in ((f"H2|{name}|share_dwell_beats_return", 100 * np.mean(sc["dwell"] > sc["return"])),
                   (f"H2|{name}|share_return_beats_CL", 100 * np.mean(sc["return"] > sc["CL"]))):
        d = abs(val - ref[k]["point"]); worst = max(worst, d); print(k, round(val, 4), round(ref[k]["point"], 4))
print("max abs difference", worst)
