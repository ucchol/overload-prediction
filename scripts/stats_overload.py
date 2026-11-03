"""Statistics layer for H1 and H3 (docs/statistics_plan.md). Logistic GLMMs of missing the best option.

For each dataset (snacks at a glance, snacks in three seconds, the four sums panels pooled with panel intercepts):
1. Growth. miss ~ log2 n, random intercept and slope of log2 n per participant.
2. Excess growth beyond a model. The same with the model's held out log odds of a miss as an offset (CL, CL with
   compensation, sample until return), each set size forecast from fits to the others as in Amendment 8.
3. Premonition. miss ~ set size intercepts + z(model log odds) + z(near best count), both standardized within
   set size, random intercept per participant. Reported for sample until return.
Usage, python3 -I scripts/stats_overload.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logit

sys.path.insert(0, "src")
from overload.experiments.overload_premonition import groups  # noqa: E402
from overload.overload_forecast import features, miss_forecasts  # noqa: E402
from overload.glmm import fit_glmm  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
sets = {"snack": G["snack"][0], "reutskaja": G["reutskaja"][0], "cdm": G["cdm_standard"][0] + G["cdm_process"][0]}
models = ("CL", "CL_comp", "SR")
rows = []


def zwithin(x, size):
    z = np.zeros_like(x, float)
    for s in np.unique(size):
        m = size == s
        sd = x[m].std()
        z[m] = (x[m] - x[m].mean()) / sd if sd > 0 else 0.0
    return z


for ds, panels in sets.items():
    Y, L, PID, PAN, OFF, NB, SZ = [], [], [], [], {m: [] for m in models}, [], []
    for k, p in enumerate(panels):
        fc, _ = miss_forecasts(p, np.ones(p.n.size), models)
        Y.append(p.miss()); L.append(np.log2(p.n)); SZ.append(p.n + 1000 * k)
        # experiment prefix, so a person keeps one identity across the two complexities of an experiment
        exp = p.name.split("_")[1] if p.name.startswith("cdm") else p.name
        PID.append(np.array([f"{exp}_{q}" for q in p.participant])); PAN.append(np.full(p.n.size, k))
        for m in models:
            OFF[m].append(logit(np.clip(fc[m], 1e-6, 1 - 1e-6)))
        NB.append(features(p)["near_best"])
    y = np.concatenate(Y).astype(float); x = np.concatenate(L); pid = np.concatenate(PID); pan = np.concatenate(PAN)
    sz = np.concatenate(SZ); nb = np.concatenate(NB)
    xc = x - x.mean()
    dummies = [(pan == k).astype(float) for k in range(1, len(panels))]
    X = np.column_stack([np.ones(y.size), xc, *dummies])
    names = ["intercept", "log2 n"] + [f"panel {panels[k].name}" for k in range(1, len(panels))]
    f = fit_glmm(y, X, pid, names=names)
    base = f.table()[1]
    p0 = 1 / (1 + np.exp(-f.beta[0]))
    rows.append({"dataset": ds, "analysis": "growth", **base, "sd_random_intercept": float(np.sqrt(f.sigma[0, 0])),
                 "sd_random_slope": float(np.sqrt(f.sigma[1, 1])), "miss_rate_at_mean_size": float(p0),
                 "pp_change_per_doubling_at_mean": float(100 * f.beta[1] * p0 * (1 - p0)), "n_trials": int(y.size)})
    print(ds, "growth", {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    for m in models:
        off = np.concatenate(OFF[m])
        fm = fit_glmm(y, X, pid, offset=off, names=names)
        rows.append({"dataset": ds, "analysis": f"excess beyond {m}", **fm.table()[1],
                     "sd_random_slope": float(np.sqrt(fm.sigma[1, 1])), "n_trials": int(y.size)})
        print(ds, m, {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    # premonition, sample until return against the near best count, within set size
    sizes = np.unique(sz)
    Xs = np.column_stack([(sz == s).astype(float) for s in sizes] + [zwithin(np.concatenate(OFF["SR"]), sz), zwithin(nb, sz)])
    nm = [f"size {s}" for s in sizes] + ["SR log odds, per SD", "near best count, per SD"]
    # random intercept only, placed through Z
    fp = fit_glmm(y, Xs, pid, Z=np.ones((y.size, 1)), names=nm)
    for r in fp.table()[-2:]:
        rows.append({"dataset": ds, "analysis": "premonition", **r, "n_trials": int(y.size)})
        print(ds, "premonition", {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)

json.dump(rows, open(out / "overload_glmm.json", "w"), indent=1)
