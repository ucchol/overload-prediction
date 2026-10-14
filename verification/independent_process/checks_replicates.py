"""Global maximum check of the search model fits in bootstrap replicates, against a dense lambda grid.

Recomputes replicates 0, 1, 499 and 999 from their seeds (file participants drawn before laptop
participants), checks that no lambda on a dense grid beats any fold's fit, and that the fitted lambdas
equal those stored by the main run. Prints derived values only.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process_core as pc  # noqa: E402
from process_core import core  # noqa: E402

pc.assert_local(pc, core)
HERE = os.path.dirname(os.path.abspath(__file__))
boot = np.load(os.path.join(HERE, "boot_process_B1000.npz"))
tasks = {n: core.build_task(n) for n in core.TASKS}
models = {n: {k: pc.SearchModel(tasks[n], k) for k in pc.SEARCH_MODELS} for n in core.TASKS}
N = tasks["file"]["P"]
dense = np.concatenate([[0.0], np.logspace(-5, 5, 2001)])
ok = True
for b in (0, 1, 499, 999):
    rng = np.random.default_rng(10000 + b)
    draws = {n: rng.integers(0, N, size=N) for n in core.TASKS}
    worst_gap, worst_stored = -np.inf, 0.0
    for n in core.TASKS:
        cnt = core.counts_from_weights(tasks[n], np.bincount(draws[n], minlength=N).astype(float))
        for i, k in enumerate(pc.SEARCH_MODELS):
            sm = models[n][k]
            for h in range(6):
                tr = np.array([c for c in range(6) if c != h])
                lam, info = sm.fit(cnt, tr)
                gap = max(sm.loglik(x, cnt, tr) for x in dense) - info["loglik"]
                worst_gap = max(worst_gap, gap)
                worst_stored = max(worst_stored, abs(lam - boot[f"{n}__lam_search"][b, i, h]) / lam)
    good = worst_gap <= 1e-9 and worst_stored < 1e-12
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} replicate {b}: largest dense excess over the fitted log likelihood {worst_gap:.1e} "
          f"(48 fold fits), largest relative difference from the stored lambda {worst_stored:.1e}", flush=True)
print("ALL REPLICATE CHECKS PASSED" if ok else "SOME REPLICATE CHECKS FAILED")
