"""Compare the exploratory two look satisficing results on the snack data with their independent reimplementation.

Usage
    python verification/compare_two_look.py --independent PATH/results.json \
        --results results/two_look_snack [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402
from compare_search_confirmatory import TOL, TOL_P_RANK, TOL_RANK, _list, _summary  # noqa: E402

MODELS = ("CL", "SR", "TLS")


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    _list(ledger, "parameter", "lambda", mine["lambda"], theirs["lambda"], TOL, relative=True)
    for m in MODELS:
        a, b = mine["E1"]["error"][m], theirs["E1"]["error"][m]
        _summary(ledger, "time", f"E1 error {m}", a, b)
        _list(ledger, "time", f"E1 error {m} by size", a["by_size"], b["by_size"])
        for k, (p, q) in enumerate(zip(a["params"], b["params"], strict=True)):
            _list(ledger, "time", f"E1 params {m} fold {k}", p, q, 1e-5, relative=True)
    for m in ("CL", "SR"):
        _summary(ledger, "time", f"E1 TLS minus {m}", mine["E1"]["TLS_minus"][m], theirs["E1"]["TLS_minus"][m])
    for name, e in mine["E2"]["correlation"].items():
        _summary(ledger, "rank correlation", f"E2 {name}", e, theirs["E2"]["correlation"][name], TOL_RANK, TOL_P_RANK)
    for key in ("TLS_minus_SR", "TLS_plus_top_rating"):
        _summary(ledger, "rank correlation", f"E2 {key}", mine["E2"][key], theirs["E2"][key], TOL_RANK, TOL_P_RANK)
    E3, T3 = mine["E3"], theirs["E3"]
    for key in ("observed", "predicted", "predicted_SR", "observed_over_predicted"):
        _list(ledger, "search", f"E3 {key}", E3[key], T3[key])
    for key in ("observed_interval", "predicted_interval"):
        for j in range(2):
            _list(ledger, "interval", f"E3 {key}[{j}]", E3[key][j], T3[key][j])
    for key in ("observed_elasticity", "predicted_elasticity", "predicted_minus_observed_elasticity"):
        _summary(ledger, "search", f"E3 {key}", E3[key], T3[key])
    U, V = mine["uniform_attention"], theirs["uniform_attention"]
    ledger.num("sensitivity", "uniform error", U["error"], V["error"], TOL)
    ledger.num("sensitivity", "uniform correlation", U["correlation"], V["correlation"], TOL_RANK)
    _list(ledger, "sensitivity", "uniform distinct", U["distinct"], V["distinct"])
    ledger.num("sensitivity", "uniform elasticity", U["elasticity"], V["elasticity"], TOL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/two_look_snack"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()
    mine = json.loads((args.results / "summary.json").read_text())
    theirs = json.loads(args.independent.read_text())
    print("Independent reimplementation against the package, two look satisficing on the snack data\n")
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
