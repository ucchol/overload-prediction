"""Synthetic click files in the input format of ProcessedCP_All_Final.csv (brief test 4).

Usage: python -I simulate_cdm.py --scenario {list,random} --out FILE [--seed 11] [--exp1-final-only]

Option values are whole dollars. Searches select an option whenever its (noisily) evaluated worth beats the
current selection, so the first option worked out is the first selection, and the last selection is the choice.
  list    satisficing in list order: options are worked out from the top, and search stops once the current
          selection's evaluated worth reaches a reservation amount of the cell (set size and complexity) shifted
          by participant.
  random  options are read at random with replacement, the best option found is kept as the selection, and
          search stops at the first return to an option already read.
Times are whole seconds since the round began (ceil of the clock), working out an option takes about 0.6 s per
operation times a participant speed factor. Extra rounds and rows test the inclusion rules: practice rounds,
unsubmitted rounds, sets of 10 with round code 8 or none, a set of 20 without a round code, Experiment 3
(complexity 9), Experiment 4 (treatments 21, 24, 25), rounds without any selection, and selections without a
valid time. A truth file lists the rounds that should be analysed.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

COLS = (['TREATID', 'TRACKING', 'USERID', 'RNDCODE', 'SUBMIT', 'RNDNUM', 'PRACTICE', 'NUMOPTS', 'NUMSLCT', 'SWITCHES',
         'RNDMAXNUMOPER', 'RNDMAXVALNUM', 'TIMESLCT', 'VALNUM', 'NUMOPER', 'OPTORD', 'SLCTNUM', 'VALDIFF',
         'SmallLargePosition ', 'LargeSmallPosition ', 'SmallLargeOperations ', 'LargeSmallOperations ',
         'SmallLargeValue ', 'LargeSmallValue'] + [f'VAL{i}' for i in range(1, 42)] + ['TOPBOTTOM'])
SIZES, CPLX = (10, 20, 40), (3, 7)
RES = {(10, 3): 19.0, (20, 3): 21.0, (40, 3): 23.0, (10, 7): 17.0, (20, 7): 19.0, (40, 7): 21.0}  # reservation amounts
NOISE = {3: 1.5, 7: 3.5}


def values(n, rng):
    """Whole dollars from 1 to 35, as the real sets hold."""
    return np.clip(np.round(rng.normal(12.0, 7.0, size=n)), 1, 35)


def search(V, cplx, scenario, rng, speed, shift):
    """Selections (option index, time in seconds) of one round."""
    n = len(V)
    sel, cur, t = [], -np.inf, 0.0
    noise = NOISE.get(cplx, 5.0)  # complexity 9 (Experiment 3) is left out of the analysis
    if scenario == "list":
        for j in range(n):
            t += 0.6 * cplx * speed * math.exp(0.25 * rng.normal())
            u = V[j] + noise * rng.normal()
            if u > cur:
                cur = u
                sel.append((j, t))
            if cur >= RES.get((n, cplx), 20.0) + shift:
                break
    else:
        read = np.zeros(n, bool)
        while True:
            j = int(rng.integers(0, n))
            if read[j]:
                break  # first return
            read[j] = True
            t += 0.6 * cplx * speed * math.exp(0.25 * rng.normal())
            u = V[j] + noise * rng.normal()
            if u > cur:
                cur = u
                sel.append((j, t))
    return sel


def round_rows(base, V, sel, cplx, rng, time_defect=None):
    n = len(V)
    vals = {f"VAL{i}": np.nan for i in range(1, 42)}
    vals["VAL1"] = 0.0
    for j in range(n):
        vals[f"VAL{j + 2}"] = V[j]
    head = dict(base, NUMOPTS=n, RNDMAXNUMOPER=cplx, RNDMAXVALNUM=V.max(), NUMSLCT=len(sel), SWITCHES=max(len(sel) - 1, 0), **vals)
    if not sel:
        return [dict(head, TIMESLCT=np.nan, VALNUM=np.nan, NUMOPER=np.nan, OPTORD=np.nan, SLCTNUM=np.nan, VALDIFF=np.nan, TOPBOTTOM=np.nan)]
    rows = []
    for k, (j, t) in enumerate(sel, start=1):
        ts = float(math.ceil(t))
        if time_defect is not None and k == time_defect[0]:
            ts = time_defect[1]
        rows.append(dict(head, TIMESLCT=ts, VALNUM=V[j], NUMOPER=int(rng.integers(1, cplx + 1)), OPTORD=j + 2, SLCTNUM=k,
                         VALDIFF=V[j] - V.max(), TOPBOTTOM=int((V[:j] > V[j]).any())))
    return rows


def make(scenario: str, out: str, seed: int, exp1_final_only: bool = False):
    rng = np.random.default_rng(seed)
    rows, truth = [], {"standard": [], "process": []}
    used = set()

    def new_id():
        while True:
            u = int(rng.integers(10000, 99999))
            if u not in used:
                used.add(u)
                return u

    design = [("standard", 0, (1, 2), 20, CPLX), ("process", 1, (3, 4), 26, CPLX),
              (None, 1, (7,), 5, (9,)), (None, 1, (21, 24, 25), 6, (3,))]  # Experiments 3 and 4 are left out
    for name, tracking, treats, P, cplxs in design:
        for p in range(P):
            uid, treat = new_id(), int(treats[p % len(treats)])
            speed, shift = math.exp(0.3 * rng.normal()), rng.normal(0.0, 1.0)
            cells = [(n, c) for n in SIZES for c in cplxs] * (4 if len(cplxs) == 2 else 3)
            rng.shuffle(cells)
            codes = {10: iter([1, 2, 3, 4, 5, 6] * 4), 20: iter(range(7, 40)), 40: iter(range(40, 80))}
            rnd = 0
            for kind in ["practice", "practice"] + ["regular"] * len(cells) + ["unsubmitted", "submit_missing", "code8", "nocode10", "nocode20", "noselection"]:
                n, c = (cells.pop() if kind == "regular" else (10 if kind != "nocode20" else 20, cplxs[0]))
                V = values(n, rng)
                sel = search(V, c, scenario, rng, speed, shift)
                if exp1_final_only and tracking == 0 and sel:
                    sel = sel[-1:]  # Experiment 1 participants click only their final choice
                base = dict(TREATID=treat, TRACKING=tracking, USERID=uid, PRACTICE=0, SUBMIT=1)
                if kind == "practice":
                    base.update(PRACTICE=1, RNDNUM=np.nan, RNDCODE=np.nan)
                else:
                    rnd += 1
                    code = next(codes[n]) if kind == "regular" else {"code8": 8, "nocode10": np.nan, "nocode20": np.nan}.get(kind, 1)
                    base.update(RNDNUM=rnd, RNDCODE=code)
                    if kind == "unsubmitted":
                        base["SUBMIT"] = 0
                    if kind == "submit_missing":
                        base["SUBMIT"] = np.nan
                if kind == "noselection":
                    sel = []
                defect = None
                if kind == "regular" and rng.random() < 0.04:
                    defect = (int(rng.integers(1, len(sel) + 1)), [np.nan, 0.0][int(rng.integers(0, 2))])
                rows += round_rows(base, V, sel, c, rng, defect)
                if name is not None and kind == "regular":
                    truth[name].append([tracking, uid, rnd])
    df = pd.DataFrame(rows)
    for col in COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLS].sample(frac=1.0, random_state=seed).reset_index(drop=True)  # row order carries no information
    df.columns = [c.lower() for c in df.columns]  # the real file's column names are lower case (brief, Inputs)
    df.to_csv(out, index=False)
    with open(os.path.splitext(out)[0] + "_truth.json", "w") as fh:
        json.dump({"scenario": scenario, "seed": seed, "exp1_final_only": exp1_final_only, "rounds": truth,
                   "reservation": {f"{k[0]}_{k[1]}": v for k, v in RES.items()}, "noise": NOISE}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=("list", "random"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exp1-final-only", action="store_true", help="Experiment 1 rounds keep only their final selection")
    args = ap.parse_args()
    make(args.scenario, args.out, args.seed, args.exp1_final_only)


if __name__ == "__main__":
    main()
