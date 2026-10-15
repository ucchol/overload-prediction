"""Synthetic gaze-format files (all four) in which choices come from two look satisficing.

Usage: python -I simulate_gaze.py --out DIR [--lam 0.5] [--seed 9]

Values are integers 0..10. Looks fall on items with probability softmax(lam v / 2) and register with
probability exp(lam (v - 10) / 2); the first item with two registered looks is chosen.
rt = speed_p * 1000 (1 + 0.25 looks) exp(0.25 z) milliseconds, with a participant speed factor speed_p.
Folke's original file carries Choice_RT (equal to rt on every row) and a confidence that falls with looks.
Edge cases test the inclusion rules (missing value, invalid choice, missing or non-positive rt, missing Conf).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "independent_two_look"))
import two_look_core as tl  # noqa: E402  (process simulation)

tl.srch.assert_local(tl)
DESIGN = {"krajbich2010": (2, 30, 80), "krajbich2011": (3, 25, 80), "folke2016": (3, 28, 90)}  # items, participants, trials


def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def make_dataset(out: str, lam: float, seed: int):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for name, (n, P, K) in DESIGN.items():
        ids = rng.choice(np.arange(1000, 9999), size=P, replace=False)
        speed = np.exp(0.3 * rng.normal(size=P))
        subj = np.repeat(ids, K)
        sp = np.repeat(speed, K)
        T = len(subj)
        V = rng.integers(0, 11, size=(T, n)).astype(float)
        q = softmax_rows(lam * V / 2.0)
        h = np.exp(lam * (V - 10.0) / 2.0)
        choice, nlooks, distinct, lpi = tl.simulate_two_look(q, h, rng)
        rt = sp * 1000.0 * (1 + 0.25 * nlooks) * np.exp(0.25 * rng.normal(size=T))
        df = pd.DataFrame({"subject": subj, "trial": np.tile(np.arange(K), P), "rt": rt, "choice": choice.astype(float)})
        for i in range(n):
            df[f"item_value_{i}"] = V[:, i]
        for i in range(n):
            df[f"gaze_{i}"] = lpi[:, i] / lpi.sum(axis=1)
        df.loc[2, "item_value_1"] = np.nan  # excluded
        df.loc[4, "choice"] = n  # excluded
        df.loc[6, "choice"] = 0.5  # excluded
        df.loc[8, "rt"] = np.nan  # excluded
        df.loc[10, "rt"] = 0.0  # excluded
        df.loc[12, "rt"] = -15.0  # excluded
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        if name == "folke2016":
            conf = np.clip(np.round(80 - 4.0 * nlooks + 8 * rng.normal(size=T)), 0, 100)
            conf[[20, 21, 22]] = np.nan  # trials without a confidence rating
            orig = pd.DataFrame({"Subject": subj, "Choice_RT": df["rt"].to_numpy(), "Conf": conf, "Trial": np.tile(np.arange(K), P)})
            orig.to_csv(os.path.join(out, "folke2016_original.csv"), index=False)
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump({"lambda": lam, "rt_law": "speed_p 1000 (1 + 0.25 looks) exp(0.25 z) ms", "conf": "80 - 4 looks + noise", "seed": seed}, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()
    make_dataset(args.out, args.lam, args.seed)


if __name__ == "__main__":
    main()
