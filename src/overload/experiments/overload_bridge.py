"""Can worse choices stand in for felt overload? Behavior against feelings (development data, exploratory).

Question. The open datasets record what people chose and how long they took, never how overloaded they felt. Only the
development data have both. There, does choosing worse or slower go with feeling overloaded, trial by trial and
display by display?

Why it matters. Choice overload is defined by its outcomes, worse or more regretted decisions, and the confirmatory
tests measure it as missed best options. Whether that also means people felt overloaded is an empirical question.
These data answer it with a number instead of an assumption.

How, separately for files and laptops and for both together.
1. Trial by trial. Felt overload (the overload index), a missed best option (regret above zero), regret and log time,
   each with every participant's average and every display's average removed. The correlation of felt overload with
   each behavioral sign says whether a person who did worse or slower than expected on a display also felt more
   overloaded than expected there. The area under the curve says how well each sign picks out the trials a person
   felt more overloaded than expected (one half is chance).
2. Display by display. Over the displays, the rank correlation of mean felt overload with each sign's mean, and the
   change from 8 to 32 options in each.

Uncertainty. 1000 participant resamples, each participant drawn with all twelve trials (seed 90000 + b).

Usage
    python -m overload.experiments.overload_bridge --data PATH_TO_CSV --out results/overload_bridge
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..overload_forecast import weighted_auc
from ..responses import load_responses
from .overload_index import _two_way, to_arrays

SEED_OFFSET = 90000
SIGNS = ("missed", "regret", "logtime")
SCOPES = {"file": slice(0, 6), "laptop": slice(6, 12), "both": slice(0, 12)}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def statistics(arr: dict) -> dict:
    out = {}
    signs = {"missed": (arr["regret"] > 0).astype(float), "regret": arr["regret"], "logtime": arr["logtime"]}
    for scope, cols in SCOPES.items():
        O = _two_way(arr["overload"][:, cols])
        felt_more = (O > 0).ravel()
        for s in SIGNS:
            X = _two_way(signs[s][:, cols])
            out[f"{scope}|trial|{s}|correlation"] = float(np.corrcoef(O.ravel(), X.ravel())[0, 1])
            out[f"{scope}|trial|{s}|auc"] = weighted_auc(X.ravel(), felt_more, np.ones(felt_more.size))
            out[f"{scope}|display|{s}|rank"] = _spearman(arr["overload"][:, cols].mean(0), signs[s][:, cols].mean(0))
        if scope != "both":
            for name, M in (("overload", arr["overload"]), *((s, signs[s]) for s in SIGNS)):
                m = M[:, cols].mean(axis=0)  # displays 8L, 8H, 16L, 16H, 32L, 32H
                out[f"{scope}|growth|{name}"] = float((m[4] + m[5] - m[0] - m[1]) / 2.0)
    return out


def run(csv: Path, out: Path, replicates: int = 1000) -> dict:
    t0 = time.time()
    arr = to_arrays(load_responses(csv))
    P = arr["overload"].shape[0]
    point = statistics(arr)
    reps = []
    for b in range(replicates):
        idx = np.random.default_rng(SEED_OFFSET + b).integers(0, P, size=P)
        reps.append(statistics({k: v[idx] for k, v in arr.items() if isinstance(v, np.ndarray)}))
    summary = {"measures": {}}
    for k, v in point.items():
        e = summarize(v, [r[k] for r in reps])
        if k.endswith("|auc"):  # chance is one half
            d = summarize(v - 0.5, [r[k] - 0.5 for r in reps])
            e["clear"], e["p"] = d["clear"], d["p"]
        summary["measures"][k] = e
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    (out / "table_overload_bridge.md").write_text(table(summary))
    return summary


def _f(e: dict, d: int = 2) -> str:
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{' *' if e['clear'] else ''}"


def table(s: dict) -> str:
    M = s["measures"]
    names = {"missed": "Missed the best", "regret": "Regret", "logtime": "Log time"}
    L = ["# Behavior and felt overload (development data, exploratory)", ""]
    L += ["## Trial by trial, person and display averages removed", ""]
    L += [
        "| Sign | Scope | Correlation with felt overload | Picks out trials felt more overloaded (AUC) |",
        "|---|---|---|---|",
    ]
    for sgn in SIGNS:
        for scope in SCOPES:
            L.append(
                f"| {names[sgn]} | {scope} | {_f(M[f'{scope}|trial|{sgn}|correlation'])} | "
                f"{_f(M[f'{scope}|trial|{sgn}|auc'])} |"
            )
    L += [
        "",
        "## Display by display",
        "",
        "| Sign | Scope | Rank correlation of display means with felt overload |",
        "|---|---|---|",
    ]
    for sgn in SIGNS:
        for scope in SCOPES:
            L.append(f"| {names[sgn]} | {scope} | {_f(M[f'{scope}|display|{sgn}|rank'])} |")
    L += ["", "## From 8 to 32 options", "", "| Measure | Files | Laptops |", "|---|---|---|"]
    for name, label in (
        ("overload", "Felt overload, scale points"),
        ("missed", "Missed the best, share"),
        ("regret", "Regret, points"),
        ("logtime", "Log time"),
    ):
        L.append(f"| {label} | {_f(M[f'file|growth|{name}'])} | {_f(M[f'laptop|growth|{name}'])} |")
    L += [
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose "
        "percentile and basic intervals both exclude zero (one half for the area under the curve).",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/overload_bridge"))
    ap.add_argument("--replicates", type=int, default=1000)
    a = ap.parse_args()
    run(a.data, a.out, a.replicates)
    print((a.out / "table_overload_bridge.md").read_text())


if __name__ == "__main__":
    main()
