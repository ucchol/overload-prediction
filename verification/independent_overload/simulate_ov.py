"""Synthetic inputs for Amendment 8 (brief test 3): a click file in the format of ProcessedCP_All_Final.csv and a
Stata file with the columns of fixations_data.dta, choices simulated from CL with a fixed sensitivity (--model cl)
or with compensation, sensitivity lam ((n - 1) / 7) ** 0.18 (--model comp).

Usage: python -I simulate_ov.py --model {cl,comp} --outdir DIR [--seed 5]
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

LAM_CDM, LAM_REUT = 0.3, 0.8


def choose(rng, v, lam, n, comp):
    s = lam * (((n - 1) / 7.0) ** 0.18 if comp else 1.0) * v
    q = np.exp(s - s.max())
    return int(rng.choice(len(v), p=q / q.sum()))


def clicks(rng, comp, path):
    rows, used = [], set()
    for tracking, P in ((0, 40), (1, 45)):
        for _ in range(P):
            while True:
                uid = int(rng.integers(10000, 99999))
                if uid not in used:
                    used.add(uid)
                    break
            cells = [(n, c) for n in (10, 20, 40) for c in (3, 7)] * 5
            rng.shuffle(cells)
            codes = {10: iter([1, 2, 3, 4, 5, 6] * 5), 20: iter(range(7, 60)), 40: iter(range(60, 120))}
            for rnd, (n, c) in enumerate(cells, start=1):
                V = np.clip(np.round(rng.normal(12.0, 7.0, size=n)), 1, 35)
                j = choose(rng, V, LAM_CDM, n, comp)
                sel = []
                if rng.random() < 0.5:  # an earlier, worse selection now and then
                    k = int(rng.integers(0, n))
                    if k != j:
                        sel.append(k)
                sel.append(j)
                vals = {f"val{i}": np.nan for i in range(1, 42)}
                vals["val1"] = 0.0
                for i in range(n):
                    vals[f"val{i + 2}"] = V[i]
                t = 0.0
                for k_, o in enumerate(sel, start=1):
                    t += 0.5 * c * math.exp(0.3 * rng.normal())
                    rows.append(dict(treatid=1 + tracking, tracking=tracking, userid=uid, rndnum=rnd, rndcode=next(codes[n]) if k_ == 1 else None,
                                     practice=0, submit=1, numopts=n, rndmaxnumoper=c, slctnum=k_, numslct=len(sel), timeslct=float(math.ceil(t)),
                                     valnum=V[o], optord=o + 2, **vals))
                code = rows[-len(sel)]["rndcode"]
                for r in rows[-len(sel):]:
                    r["rndcode"] = code
    pd.DataFrame(rows).sample(frac=1.0, random_state=1).to_csv(path, index=False)


def stata(rng, comp, path):
    rows = []
    for subj in rng.choice(np.arange(100, 999), size=40, replace=False):
        sizes = [4] * 25 + [9] * 25 + [16] * 25
        rng.shuffle(sizes)
        for tid, n in enumerate(sizes, start=1):
            v = rng.integers(-5, 6, size=n).astype(float)
            j = choose(rng, v, LAM_REUT, n, comp)
            seq = list(rng.integers(0, n, size=int(rng.integers(1, 2 * n)))) + [j]
            ratings = {f"rating{i}": (v[i - 1] if i <= n else 0.0) for i in range(1, 17)}
            for o, roi in enumerate(seq, start=1):
                rows.append(dict(subject=float(subj), trialid=float(tid), numalt=float(n), chosendestring=float(j + 1),
                                 Roinum=float(roi + 1), fixOrder=float(o), **ratings))
    pd.DataFrame(rows).to_stata(path, write_index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("cl", "comp"), required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    clicks(rng, a.model == "comp", os.path.join(a.outdir, f"clicks_{a.model}.csv"))
    stata(rng, a.model == "comp", os.path.join(a.outdir, f"trials_{a.model}.dta"))


if __name__ == "__main__":
    main()
