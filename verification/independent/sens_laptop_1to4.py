"""Sensitivity analysis. Laptop rubric shares from the first four laptop trials only.

Everything else follows the plan and the brief. All six laptop trials are predicted and evaluated,
with the same models, folds, benchmarks, bootstrap and B1 seeds. Replicate b draws the file panel
first and then the laptop panel from rng = default_rng(10000 + b), so the file draw is made and
discarded to keep the laptop draws identical to the main run.

Usage: python sens_laptop_1to4.py [--B 1000] [--rubric-trials 0 1 2 3] [--out sens_laptop_1to4.json]
Writes the laptop results in the layout of results.json["laptop"]. No counts are printed or stored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402
import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(rubric_trials, B, log_every=100):
    t0 = time.time()
    P_file = core.build_task("file")["P"]  # only the panel length is needed, to advance the generator
    task = core.build_task("laptop", rubric_trials=rubric_trials)
    pt = analysis.point_sample(task)
    C = 6
    M, E = len(core.ALL_MODELS), len(core.EBA_MODELS)
    store = dict(
        L=np.zeros((B, M, C)),
        O=np.zeros((B, M, C)),
        lam=np.zeros((B, len(core.SOFTMAX_MODELS), C)),
        eps=np.zeros((B, E, C)),
        e_obs=np.zeros((B, C)),
        e_mod=np.zeros((B, C)),
    )
    init = pt["ev"]["lam"][:, :C]
    cl0 = core.ALL_MODELS.index(core.beta_name("CL", 0.0))
    for b in range(B):
        rng = np.random.default_rng(10000 + b)
        rng.integers(0, P_file, size=P_file)  # file panel draw, discarded
        draw = rng.integers(0, task["P"], size=task["P"])
        cnt = core.counts_from_weights(task, np.bincount(draw, minlength=task["P"]).astype(float))
        ev = core.evaluate(task, cnt, include_full=False, init=init)
        store["L"][b], store["O"][b] = ev["L_ho"], ev["O_ho"]
        store["lam"][b], store["eps"][b] = ev["lam"], ev["eps"]
        store["e_obs"][b], store["e_mod"][b] = core.b1_components(task, cnt, ev["Q_ho"][cl0], b=b)
        if (b + 1) % log_every == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B} done, {time.time() - t0:.1f}s elapsed", flush=True)
    return task, pt, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--rubric-trials", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=os.path.join(HERE, "sens_laptop_1to4.json"))
    args = ap.parse_args()
    t0 = time.time()
    task, pt, store = run(args.rubric_trials, args.B)
    for k in ("lam", "eps", "L", "O"):
        if not np.all(np.isfinite(store[k])):
            raise RuntimeError(f"non-finite values in replicate {k}")
    print(f"replicate fits with lambda == 0: {int((store['lam'] == 0).sum())}, "
          f"lapse at 0 or 1: {int(((store['eps'] == 0) | (store['eps'] == 1)).sum())}", flush=True)
    res = analysis.summarize_task(task, pt, store)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
