"""Post-run checks on the original sample of the real snack data (run once, after the main run).

1. Amendment 5 item 2 locates SR's maximum on a grid of zero and 21 values spaced evenly in log between
   0.01 and 100, refined by bounded Brent search between the neighbours of the best grid value. The main
   run uses a finer grid and refines every local maximum. This recomputes the four held-out SR
   sensitivities with the amendment's procedure (at a tight tolerance and at scipy's default) and
   compares them with results.json.
2. Dense-grid global maximum check of the four SR fold fits, and stationarity of the static fits.
Prints sensitivities and differences only (no counts).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_search as rs  # noqa: E402
import search_core as srch  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, rs, sc)
R = json.load(open(os.path.join(HERE, "results.json")))
data = srch.load_search_data(srch.REAL_DATA, allow_real=True)
pre = rs.Pre(data)
w = np.ones(len(data["choice"]))
amend_grid = np.concatenate([[0.0], np.logspace(-2, 2, 21)])
dense = np.concatenate([[0.0], np.logspace(-5, 3, 641)])
tab = np.array([srch.sr_chosen(x, pre.C, pre.lc, need_grad=False)[0] for x in dense])
out = {"amendment_procedure": {}, "dense_check": {}}
for k in range(4):
    wk = np.where(pre.train_mask[k], w, 0.0)
    mine = R["lambda"]["SR"][k]
    ll = np.array([pre.sr.loglik(x, wk) for x in amend_grid])
    g = int(np.argmax(ll))
    lo, hi = amend_grid[max(g - 1, 0)], amend_grid[min(g + 1, len(amend_grid) - 1)]
    tight = minimize_scalar(lambda x: -pre.sr.loglik(x, wk), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}).x
    default = minimize_scalar(lambda x: -pre.sr.loglik(x, wk), bounds=(lo, hi), method="bounded").x
    out["amendment_procedure"][srch.SET_SIZES[k]] = {"main_run": mine, "amendment_tight": float(tight), "amendment_default_tolerance": float(default),
                                                     "relative_difference_tight": float(abs(tight - mine) / mine),
                                                     "relative_difference_default": float(abs(default - mine) / mine)}
    ll_fit = pre.sr.loglik(mine, wk)
    out["dense_check"][srch.SET_SIZES[k]] = {"dense_excess_over_fit": float((tab @ wk).max() - ll_fit),
                                             "relative_score_at_fit": float(abs(pre.sr.score(mine, wk)) / abs(ll_fit))}
worst_static = 0.0
for fam in srch.STATIC:
    for k in range(4):
        Pt = pre.train[fam][k]
        lam = R["lambda"][fam][k]
        m1, var, _ = sc.moments(Pt, np.full((1, len(Pt.trials)), lam))
        worst_static = max(worst_static, abs(float((Pt.dc - m1[0]).sum()) / float(var[0].sum())) / lam)
out["static_largest_relative_newton_step"] = worst_static
with open(os.path.join(HERE, "postrun_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
