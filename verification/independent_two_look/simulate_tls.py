"""Synthetic snack-format data in which choices, gaze and response times come from two look satisficing.

Usage: python -I simulate_tls.py --out DIR [--lam 1.2] [--seed 5] [--subjects S] [--trials K] [--noise 0.3]
Looks fall on items with probability softmax(lam v / 2) and register with probability exp(lam (v - 3) / 2);
the first item with two registered looks is chosen. Cumulative gaze is positive exactly for the items
looked at. rt = 1000 (1 + 0.3 looks) exp(noise z) milliseconds. Edge cases as in the search simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "independent_search"))
import simulate as sim_sr  # noqa: E402  (display generator of the search work)
import two_look_core as tl  # noqa: E402

tl.srch.assert_local(sim_sr, tl)
SET_SIZES = (9, 16, 25, 36)


def make_dataset(out, lam, seed, subjects, trials, noise, per_look=0.3):
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(100, 1000), size=subjects, replace=False)
    os.makedirs(out, exist_ok=True)
    for n in SET_SIZES:
        subj = np.repeat(ids, trials)
        T = len(subj)
        V = sim_sr.displays(T, n, rng)
        q = sim_sr.softmax_rows(lam * V / 2.0)
        h = np.exp(lam * (V - 3.0) / 2.0)
        choice, nlooks, distinct, lpi = tl.simulate_two_look(q, h, rng)
        gaze = 0.9 * lpi / lpi.sum(axis=1, keepdims=True)
        rt = 1000.0 * (1 + per_look * nlooks) * np.exp(noise * rng.normal(size=T))
        df = pd.DataFrame({"setsize": n, "subject": subj, "trial": np.tile(np.arange(trials), subjects), "rt": rt,
                           "rt_choice_indication": rt + 300.0, "choice": choice.astype(float),
                           "best_chosen": 0, "best_seen_chosen": 0, "longest_chosen": 0,
                           "returning_gaze_count": 0, "seen_items_count": distinct})
        df = pd.concat([df, pd.DataFrame(V, columns=[f"item_value_{i}" for i in range(n)]),
                        pd.DataFrame(gaze, columns=[f"cumulative_gaze_{i}" for i in range(n)])], axis=1)
        df.loc[3, "item_value_2"] = np.nan
        df.loc[5, "choice"] = n
        df.loc[7, "choice"] = 1.5
        df.loc[11, "rt"] = np.nan
        df.loc[13, "rt"] = 0.0
        df.loc[17, "rt"] = -40.0
        df = df.sample(frac=1.0, random_state=seed + n).reset_index(drop=True)
        df.to_csv(os.path.join(out, f"setsize-{n}_desc-data.csv"), index=False)
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump({"lambda": lam, "rt_law": f"1000 (1 + {per_look} looks) exp(noise z) ms", "noise": noise, "seed": seed}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.3)
    args = ap.parse_args()
    make_dataset(args.out, args.lam, args.seed, args.subjects, args.trials, args.noise)


if __name__ == "__main__":
    main()
