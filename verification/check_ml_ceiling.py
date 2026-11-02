"""Separate check of scripts/ml_ceiling.py (point values).

1. CL held out log2 probabilities refitted here with a likelihood and Nelder Mead fit written here, on folds rebuilt here.
2. Bits per choice and shares of the ceiling recomputed from the saved held out trial scores.
Usage, python3 -I verification/check_ml_ceiling.py results/ml_ceiling
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "src")
from overload.experiments.overload_premonition import groups  # noqa: E402

res = Path(sys.argv[1])
S = json.load(open(res / "summary.json"))
Z = np.load(res / "trial_scores.npz")
G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
panels = G["snack"][0] + G["reutskaja"][0] + G["cdm_standard"][0] + G["cdm_process"][0]


def cl_logp(v, ch, lam):
    u = np.where(np.isfinite(v), lam * v, -np.inf)
    m = u.max(axis=1, keepdims=True)
    lz = m[:, 0] + np.log(np.exp(u - m).sum(axis=1))
    return (u[np.arange(len(ch)), ch] - lz) / np.log(2)


worst_cl = 0.0
for p in panels:
    P = int(p.participant.max()) + 1
    perm = np.random.default_rng(20261014).permutation(P)
    fold = np.empty(P, int)
    for k, q in enumerate(perm):
        fold[q] = k % 10
    f = fold[p.participant]
    mine = np.full(len(p.choice), np.nan)
    for k in range(10):
        tr, te = f != k, f == k
        if not te.any():
            continue
        r = minimize(lambda x: -cl_logp(p.values[tr], p.choice[tr], np.exp(x[0])).sum(), [0.0], method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-12})
        mine[te] = cl_logp(p.values[te], p.choice[te], np.exp(r.x[0]))
    d = np.max(np.abs(mine - Z[f"{p.name}|CL"]))
    worst_cl = max(worst_cl, d)
    print(p.name, "CL trial scores, max difference", d)

groups_out = {"snack": ["snack"], "reutskaja": ["reutskaja"], "cdm": [p.name for p in panels if p.name.startswith("cdm")]}
worst = 0.0
for g, names in groups_out.items():
    bits = {m: np.mean([np.mean(Z[f"{n}|{m}"] - Z[f"{n}|uniform"]) for n in names]) for m in ("CL", "SR", "NN", "GBM")}
    c = S[f"{g}|ceiling_model"]
    vals = {f"{g}|bits|{m}": bits[m] for m in bits} | {f"{g}|share_of_ceiling|{m}": 100 * bits[m] / bits[c] for m in ("CL", "SR")}
    for k, v in vals.items():
        worst = max(worst, abs(v - S[k]["point"]))
print("CL refit, max difference in trial scores", worst_cl)
print("summary points from trial scores, max difference", worst)
