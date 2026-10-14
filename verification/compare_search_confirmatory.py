"""Compare the confirmatory search model results on the snack data with their independent reimplementation and a rerun.

Written before the run of Amendment 5. The independent implementation follows `VERIFY_BRIEF_SEARCH.md` and writes the
layout given there.

Usage
    python verification/compare_search_confirmatory.py --independent PATH/results.json \
        --results results/search_confirmatory [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402

MODELS = ("CL", "DbS_avg", "MLBA_avg", "SR")
STATIC = ("CL", "DbS_avg", "MLBA_avg")
# Both sides locate the search model's sensitivity to about 1e-10 and the time law at its global minimum, so values
# agree to about 1e-8. Rank correlations can move when two forecasts are equal to within rounding, which the looser
# tolerance for S3 and its p values allows for. Every decision must agree exactly.
TOL = 1e-6
TOL_RANK = 1e-3
TOL_P_RANK = 0.006


def _summary(ledger: Ledger, kind: str, label: str, a: dict, b: dict, tol: float = TOL, tol_p: float = 1e-12) -> None:
    ledger.num(kind, f"{label} point", a["point"], b["point"], tol)
    for which in ("percentile", "basic"):
        if which in a and which in b:
            for k in range(2):
                ledger.num("interval", f"{label} {which}[{k}]", a[which][k], b[which][k], tol)
    if "p" in a and "p" in b:
        ledger.num("p value", f"{label} p", a["p"], b["p"], tol_p)
        ledger.flag("clear", f"{label} clear", a["clear"], b["clear"])


def _list(ledger: Ledger, kind: str, label: str, a: list, b: list, tol: float = TOL, relative: bool = False) -> None:
    for k, (x, y) in enumerate(zip(a, b, strict=True)):
        ledger.num(kind, f"{label} [{k}]", x, y, tol * max(1.0, abs(x)) if relative else tol)


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    for m in MODELS:
        _list(ledger, "parameter", f"lambda {m}", mine["lambda"][m], theirs["lambda"][m], TOL, relative=True)

    s1, t1 = mine["S1"], theirs["S1"]
    for key in ("observed_by_size", "predicted_by_size", "observed_over_predicted_by_size"):
        _list(ledger, "search", f"S1 {key}", s1[key], t1[key])
    for key in ("observed_elasticity", "predicted_elasticity", "one_minus_observed", "predicted_minus_observed"):
        _summary(ledger, "search", f"S1 {key}", s1[key], t1[key])
    ledger.flag("decision", "S1 within margin", s1["within_margin"], t1["within_margin"])
    ledger.flag("decision", "S1", s1["supported"], t1["supported"])

    s2, t2 = mine["S2"], theirs["S2"]
    for m in MODELS:
        _summary(ledger, "time", f"S2 error {m}", s2["error"][m], t2["error"][m])
        _list(ledger, "time", f"S2 error {m} by size", s2["error"][m]["by_size"], t2["error"][m]["by_size"])
    for m in STATIC:
        _summary(ledger, "time", f"S2 SR minus {m}", s2["SR_minus"][m], t2["SR_minus"][m])
    ledger.flag("decision", "S2", s2["supported"], t2["supported"])

    s3, t3 = mine["S3"], theirs["S3"]
    for name, e in s3["correlation"].items():
        _summary(ledger, "rank correlation", f"S3 {name}", e, t3["correlation"][name], TOL_RANK, TOL_P_RANK)
    _summary(ledger, "rank correlation", "S3 SR minus CL", s3["SR_minus_CL"], t3["SR_minus_CL"], TOL_RANK, TOL_P_RANK)
    ledger.flag("decision", "S3", s3["supported"], t3["supported"])

    s4, t4 = mine["S4"], theirs["S4"]
    for m in (*MODELS, "constant"):
        _summary(ledger, "premonition", f"S4 score {m}", s4["score"][m], t4["score"][m])
    for m in MODELS:
        _summary(ledger, "premonition", f"S4 {m} minus constant", s4["minus_constant"][m], t4["minus_constant"][m])
    _summary(ledger, "premonition", "S4 SR minus CL", s4["SR_minus_CL"], t4["SR_minus_CL"])

    s5, t5 = mine["S5"], theirs["S5"]
    _summary(ledger, "choices", "S5 SR minus CL percent", s5["SR_minus_CL_percent"], t5["SR_minus_CL_percent"])
    for key in ("noninferior", "better", "worse", "supported"):
        ledger.flag("decision", f"S5 {key}", s5[key], t5[key])

    c, tc = mine["choices"], theirs["choices"]
    for m in MODELS:
        a, b = c["times_chance"][m], tc["times_chance"][m]
        ledger.num("choices", f"{m} times chance", a["point"], b["point"], TOL)
        _list(ledger, "interval", f"{m} times chance percentile", a["percentile"], b["percentile"])
        _list(ledger, "choices", f"{m} times chance by size", a["by_size"], b["by_size"])
    for m in STATIC:
        _summary(ledger, "choices", f"SR minus {m} percent", c["SR_minus_percent"][m], tc["SR_minus_percent"][m])

    _summary(ledger, "secondary", "implied elasticity", mine["implied_elasticity"], theirs["implied_elasticity"])
    _list(
        ledger, "secondary", "implied CL sensitivity", mine["implied_cl_sensitivity"], theirs["implied_cl_sensitivity"]
    )
    _list(ledger, "secondary", "median seconds", mine["median_seconds"], theirs["median_seconds"])
    for m, v in mine["per_step_seconds"].items():
        _list(ledger, "secondary", f"per step seconds {m}", v, theirs["per_step_seconds"][m])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/search_confirmatory"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()
    mine = json.loads((args.results / "summary.json").read_text())
    theirs = json.loads(args.independent.read_text())
    print("Independent reimplementation against the package, search model on the snack data (Amendment 5)\n")
    ledger = Ledger()
    compare(ledger, mine, theirs)
    failures = ledger.report()
    if args.rerun:
        print("\nSecond run of the package")
        for fname in ("summary.json", "point.json"):
            same, _ = identical(args.results / fname, args.rerun / fname)
            print(f"  {fname}: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
        r1, r2 = args.results / "replicates.jsonl", args.rerun / "replicates.jsonl"
        same = [json.loads(a) for a in r1.open()] == [json.loads(b) for b in r2.open()]
        print(f"  replicates.jsonl: {'identical' if same else 'DIFFERENT'}")
        failures += 0 if same else 1
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
