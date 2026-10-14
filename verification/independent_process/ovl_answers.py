"""Overload index analysis (section B of VERIFY_BRIEF_PROCESS.md).

Arrays are participants by twelve trials (file trials 1..6, then laptop trials 1..6). Nothing here
prints or stores identifiers or counts.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
PREFIX = (("file", "ci_co_fl_", "file_trials.json"), ("laptop", "em_co_lp_", "laptop_trials.json"))


def trial_names() -> list:
    names = []
    for _, _, fn in PREFIX:
        with open(os.path.join(VERIFY, "stimuli", fn)) as fh:
            names += list(json.load(fh).keys())
    return names


def load_answers():
    """Raw answers S (N, 12, 6), regret (N, 12) and seconds (N, 12), rows in file order."""
    cols = [f"{p}{k}{s}" for _, p, _ in PREFIX for k in range(1, 7)
            for s in ["_regret", "_timer"] + [f"_survey_{j}" for j in range(1, 7)]]
    df = pd.read_csv(os.path.join(VERIFY, "Updated_Choice_Overload.csv"), usecols=cols)
    S = np.stack([np.stack([df[f"{p}{k}_survey_{j}"].to_numpy(float) for j in range(1, 7)], axis=1)
                  for _, p, _ in PREFIX for k in range(1, 7)], axis=1)
    regret = np.stack([df[f"{p}{k}_regret"].to_numpy(float) for _, p, _ in PREFIX for k in range(1, 7)], axis=1)
    seconds = np.stack([df[f"{p}{k}_timer"].to_numpy(float) for _, p, _ in PREFIX for k in range(1, 7)], axis=1)
    if not (np.isfinite(S).all() and np.isfinite(regret).all() and (seconds > 0).all()):
        raise ValueError("missing answers")
    return S, regret, seconds


def aligned(S: np.ndarray) -> np.ndarray:
    A = S.copy()
    A[..., 0] = 6 - S[..., 0]  # satisfaction
    A[..., 1] = 6 - S[..., 1]  # certainty of having chosen best
    return A


def overload_index(S: np.ndarray) -> np.ndarray:
    return aligned(S).mean(axis=2)


def cronbach(rows: np.ndarray) -> float:
    k = rows.shape[1]
    return float(k / (k - 1) * (1 - rows.var(axis=0, ddof=1).sum() / rows.sum(axis=1).var(ddof=1)))


def double_center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()


def analyse(S: np.ndarray, regret: np.ndarray, seconds: np.ndarray) -> dict:
    """Every quantity of B.1 to B.4 for one sample of participant rows (duplicates are separate rows)."""
    A = aligned(S)
    Ac = (A - A.mean(axis=1, keepdims=True)).reshape(-1, 6)  # centered within participant, stacked
    ev = np.linalg.eigvalsh(np.corrcoef(Ac, rowvar=False))
    Sc = (S - S.mean(axis=1, keepdims=True)).reshape(-1, 6)
    out = {
        "alpha_within_person": cronbach(Ac),
        "alpha_raw": cronbach(A.reshape(-1, 6)),
        "first_component_share": float(ev.max() / ev.sum()),
        "q6": np.array([np.corrcoef(Sc[:, 5], Sc[:, j])[0, 1] for j in range(5)]),
    }
    idx = A.mean(axis=2)
    logt = np.log(seconds)
    out["overload"] = idx.mean(axis=0)
    out["overload_vs_average"] = idx.mean(axis=0) - idx.mean()
    out["regret"] = regret.mean(axis=0)
    out["best_chosen"] = (regret == 0).mean(axis=0)
    out["geo_seconds"] = np.exp(logt.mean(axis=0))
    contrasts = {}
    for t, (task, _, _) in enumerate(PREFIX):
        cols = slice(6 * t, 6 * t + 6)
        for qname, X in (("overload", idx), ("regret", regret), ("log_seconds", logt)):
            M = X[:, cols].mean(axis=0)
            d32 = ((M[4] + M[5]) - (M[0] + M[1])) / 2
            dfb = ((M[1] + M[3] + M[5]) - (M[0] + M[2] + M[4])) / 3
            if qname == "log_seconds":
                d32, dfb = 100 * (np.exp(d32) - 1), 100 * (np.exp(dfb) - 1)
            contrasts[(task, qname, "32_minus_8")] = d32
            contrasts[(task, qname, "full_minus_brief")] = dfb
    out["contrasts"] = contrasts
    assoc = {}
    for sname, cols in (("file", slice(0, 6)), ("laptop", slice(6, 12)), ("both", slice(0, 12))):
        r = {k: double_center(X[:, cols]).ravel() for k, X in (("overload", idx), ("regret", regret), ("logtime", logt))}
        for a, b in (("overload", "regret"), ("overload", "logtime"), ("regret", "logtime")):
            assoc[(sname, f"{a}~{b}")] = float(np.corrcoef(r[a], r[b])[0, 1])
    out["associations"] = assoc
    return out
