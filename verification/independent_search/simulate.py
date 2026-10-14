"""Synthetic snack-format data in which choices, gaze and response times come from the SR process.

Usage: python -I simulate.py --out DIR [--lam 1.0] [--seed 7] [--subjects S] [--trials K] [--noise 0.3]

Each display draws its ratings as clip(round(mu + sd z), -3, 3) with mu and sd varying across displays,
so that the expected search differs between displays of the same size. Looks fall on items with
probability softmax(lam v), with replacement, until the first repeat, which is the choice. Cumulative
gaze is positive exactly for the items looked at. rt = 1000 (1 + 0.4 looks) exp(noise z) milliseconds.
A few edge cases test the inclusion rules (a missing rating, an out of range choice, missing or
non-positive rt, missing gaze values).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

SET_SIZES = (9, 16, 25, 36)


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def simulate_sr(V: np.ndarray, lam: float, rng: np.random.Generator):
    """SR process on displays V (trials x n): choice, number of looks, looks per item."""
    T, n = V.shape
    q = softmax_rows(lam * V)
    cum = np.cumsum(q, axis=1)
    U = rng.random((T, n + 1))
    looks = np.minimum((U[:, :, None] > cum[:, None, :]).sum(axis=2), n - 1)  # inverse cdf
    rows = np.arange(T)
    seen = np.zeros((T, n), bool)
    counts = np.zeros((T, n))
    choice = np.full(T, -1)
    nlooks = np.zeros(T, int)
    for t in range(n + 1):  # a repeat is certain by look n + 1
        live = choice < 0
        if not live.any():
            break
        col = looks[:, t]
        rep = live & seen[rows, col]
        choice[rep] = col[rep]
        nlooks[rep] = t + 1
        counts[rows[live], col[live]] += 1
        seen[rows[live], col[live]] = True
    return choice, nlooks, counts


def displays(T: int, n: int, rng: np.random.Generator) -> np.ndarray:
    mu = rng.uniform(-1.5, 1.5, size=(T, 1))
    sd = rng.uniform(0.3, 2.5, size=(T, 1))
    return np.clip(np.round(mu + sd * rng.normal(size=(T, n))), -3, 3)


def make_dataset(out: str, lam: float, seed: int, subjects: int, trials: int, noise: float, edge: bool = True,
                 choice_model: str = "sr", rt_law: str = "looks"):
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(100, 1000), size=subjects, replace=False)  # unsorted, non-consecutive
    os.makedirs(out, exist_ok=True)
    for n in SET_SIZES:
        subj = np.repeat(ids, trials)
        T = len(subj)
        V = displays(T, n, rng)
        choice, nlooks, counts = simulate_sr(V, lam, rng)
        if choice_model == "cl":  # null control: choices from the conditional logit, independent of the looks
            q = softmax_rows(lam * V)
            choice = np.minimum((rng.random((T, 1)) > np.cumsum(q, axis=1)).sum(axis=1), n - 1)
        gaze = 0.9 * counts / counts.sum(axis=1, keepdims=True)
        drive = nlooks if rt_law == "looks" else 0.25 * n * np.ones(T)  # null control: time depends on set size only
        rt = 1000.0 * (1 + 0.4 * drive) * np.exp(noise * rng.normal(size=T))
        df = pd.DataFrame({"setsize": n, "subject": subj, "trial": np.tile(np.arange(trials), subjects), "rt": rt,
                           "rt_choice_indication": rt + 300.0, "choice": choice.astype(float),
                           "best_chosen": 0, "best_seen_chosen": 0, "longest_chosen": 0,
                           "returning_gaze_count": 0, "seen_items_count": (counts > 0).sum(axis=1)})
        vals = pd.DataFrame(V, columns=[f"item_value_{i}" for i in range(n)])
        gz = pd.DataFrame(gaze, columns=[f"cumulative_gaze_{i}" for i in range(n)])
        df = pd.concat([df, vals, gz], axis=1)
        if edge:
            df.loc[3, "item_value_2"] = np.nan  # excluded: a missing rating
            df.loc[5, "choice"] = n  # excluded: choice outside the display
            df.loc[7, "choice"] = 1.5  # excluded: non-integer choice
            df.loc[11, "rt"] = np.nan  # kept, not timed
            df.loc[13, "rt"] = 0.0  # kept, not timed
            df.loc[17, "rt"] = -40.0  # kept, not timed
            unl = np.flatnonzero(gaze[19] == 0)
            if len(unl):
                df.loc[19, f"cumulative_gaze_{unl[0]}"] = np.nan  # missing gaze counts as zero
        df = df.sample(frac=1.0, random_state=seed + n).reset_index(drop=True)  # row order is irrelevant
        df.to_csv(os.path.join(out, f"setsize-{n}_desc-data.csv"), index=False)
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump({"lambda": lam, "choice_model": choice_model, "rt_law": rt_law, "noise": noise, "seed": seed}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--choice-model", choices=("sr", "cl"), default="sr")
    ap.add_argument("--rt-law", choices=("looks", "setsize"), default="looks")
    args = ap.parse_args()
    make_dataset(args.out, args.lam, args.seed, args.subjects, args.trials, args.noise,
                 choice_model=args.choice_model, rt_law=args.rt_law)


if __name__ == "__main__":
    main()
