"""Sensitivity of A1b to the median convention for even set sizes (not part of the specified results).

Main run: numpy.median (midpoint of the two middle values), aspect when x >= median.
Alternative: lower middle value. Same seeds and bootstrap as the main run; CL (beta = 0) held out log
scores are taken from the main run's bootstrap file. Writes sensitivity_median.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
boot = np.load(os.path.join(HERE, "boot_B1000.npz"))
B = boot["file__L"].shape[0]
CL = core.ALL_MODELS.index(core.beta_name("CL", 0.0))

tasks = {n: core.build_task(n) for n in core.TASKS}
A_low = {}
for n, task in tasks.items():
    A = np.zeros((1,) + task["A"].shape[1:])
    for c, t in enumerate(task["trials"]):
        ks = np.where(task["r"] > 0)[0]
        Xk = t["X"][:, ks]
        lower = np.sort(Xk, axis=0)[(t["n"] - 1) // 2]
        A[0, c, : t["n"]] = core.eba_from_aspects(Xk >= lower, task["r"][ks])
    A_low[n] = A


def eba_ho_bits(task, A, cnt):
    T = core.fold_masks(6, include_full=False)
    eps = core.fit_lapse(A, task["valid"], cnt, task["nvec"], T)
    Q = core.lapse_predict(eps, A, task["valid"], task["nvec"])
    return core.log_scores(Q, cnt, task["nvec"])[0], eps


out = {}
pts = {}
for n, task in tasks.items():
    cnt = core.counts_from_weights(task, np.ones(task["P"]))
    Ls = core.lstar(cnt, task["nvec"])
    L_e, _ = eba_ho_bits(task, A_low[n], cnt)
    ev = core.evaluate(task, cnt, include_full=False)
    pts[n] = dict(L_e=L_e, L_cl=ev["L_ho"][CL], Ls=Ls)
reps = {n: dict(L_e=np.zeros((B, 6))) for n in core.TASKS}
for b in range(B):
    rng = np.random.default_rng(10000 + b)
    draws = {n: rng.integers(0, tasks[n]["P"], size=tasks[n]["P"]) for n in core.TASKS}
    for n, task in tasks.items():
        cnt = core.counts_from_weights(task, np.bincount(draws[n], minlength=task["P"]).astype(float))
        reps[n]["L_e"][b], _ = eba_ho_bits(task, A_low[n], cnt)
for n, task in tasks.items():
    s8, s32 = task["sizes"] == 8, task["sizes"] == 32
    d = pts[n]["L_e"] - pts[n]["L_cl"]
    db = reps[n]["L_e"] - boot[f"{n}__L"][:, CL]
    Ke = core.completeness(pts[n]["L_e"], pts[n]["Ls"])
    Kb = core.completeness(reps[n]["L_e"], pts[n]["Ls"])
    Kcl = core.completeness(pts[n]["L_cl"], pts[n]["Ls"])
    Kclb = core.completeness(boot[f"{n}__L"][:, CL], pts[n]["Ls"])
    out[n] = {
        "EBA_lower_median": {
            "completeness": core.summary_level(Ke, Kb),
            "minus_CL": core.summary_diff(Ke - Kcl, Kb - Kclb),
            "delta_bits_by_size": {str(s): float(d[task["sizes"] == s].mean()) for s in (8, 16, 32)},
            "large_minus_small_bits": core.summary_diff(d[s32].mean() - d[s8].mean(), db[:, s32].mean(axis=1) - db[:, s8].mean(axis=1)),
        }
    }
    print(n, json.dumps(out[n]["EBA_lower_median"]["large_minus_small_bits"]), "K", round(Ke, 2))
with open(os.path.join(HERE, "sensitivity_median.json"), "w") as fh:
    json.dump(out, fh, indent=1)
