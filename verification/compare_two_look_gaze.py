"""Compare the confirmatory two look satisficing results on three gaze datasets (Amendment 6) with their independent
reimplementation, and the saved results with a second run. Written, with its tolerances, before the run.

Tolerances are those of the earlier comparisons. 1e-6 for values, intervals and p values (lambdas relative), and 1e-3
for rank correlations with 0.006 for their p values, since a fitted sensitivity that differs in the eighth digit can
reorder two nearly equal forecasts. Decisions must agree exactly.

Usage
    python verification/compare_two_look_gaze.py --independent PATH/results.json \
        --results results/two_look_gaze [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402
from compare_search_confirmatory import TOL, TOL_P_RANK, TOL_RANK, _list, _summary  # noqa: E402

DATASETS = ("krajbich2010", "krajbich2011", "folke2016")
TIME = ("static", "SR", "TLS", "TLS_minus_SR", "TLS_minus_static")
SLOW = (
    "looks_TLS",
    "looks_SR",
    "top_value",
    "mean_value",
    "top_gap",
    "TLS_minus_SR",
    "TLS_plus_top_value",
    "TLS_plus_mean_value",
    "TLS_plus_top_gap",
)
CHOICE_MODELS = ("CL", "DbS", "MLBA", "SR")
STATIC = ("CL", "DbS", "MLBA")
SEARCH = ("SR", "TLS")
CONFIDENCE = ("looks_TLS", "looks_SR", "entropy_CL", "TLS_minus_SR")


def block(ledger: Ledger, label: str, a: dict, b: dict) -> None:
    for k in TIME:
        _summary(ledger, "time", f"{label} time {k}", a["time"][k], b["time"][k])
    for k in SLOW:
        _summary(ledger, "rank correlation", f"{label} slow {k}", a["slow"][k], b["slow"][k], TOL_RANK, TOL_P_RANK)
    for m in CHOICE_MODELS:
        x, y = a["choices"]["times_chance"][m], b["choices"]["times_chance"][m]
        ledger.num("choices", f"{label} times chance {m}", x, y, TOL)
    for m in STATIC:
        key = f"TLS_minus_{m}_percent"
        _summary(ledger, "choices", f"{label} {key}", a["choices"][key], b["choices"][key])


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    P, Q = mine["pooled"], theirs["pooled"]
    block(ledger, "pooled", P, Q)
    for g in ("G1", "G2"):
        ledger.flag("decision", g, P[g]["supported"], Q[g]["supported"])
    for key in ("noninferior", "better", "worse", "supported"):
        ledger.flag("decision", f"G3 {key}", P["G3"][key], Q["G3"][key])
    for name in DATASETS:
        A, B = mine[name], theirs[name]
        block(ledger, name, A, B)
        for m in CHOICE_MODELS:
            _list(ledger, "parameter", f"{name} lambda {m}", A["lambda"][m], B["lambda"][m], TOL, relative=True)
        for m in SEARCH:
            for key in ("t0", "tau"):
                _summary(ledger, "time law", f"{name} law {m} {key}", A["law"][m][key], B["law"][m][key])
            _summary(ledger, "looks", f"{name} looks {m}", A["looks"][m], B["looks"][m])
        _summary(ledger, "time", f"{name} typical seconds", A["typical_seconds"], B["typical_seconds"])
    C, D = mine["folke2016"]["confidence"], theirs["folke2016"]["confidence"]
    for k in CONFIDENCE:
        _summary(ledger, "rank correlation", f"confidence {k}", C[k], D[k], TOL_RANK, TOL_P_RANK)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/two_look_gaze"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()
    mine = json.loads((args.results / "summary.json").read_text())
    theirs = json.loads(args.independent.read_text())
    print("Independent reimplementation against the package, two look satisficing on three gaze datasets\n")
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
