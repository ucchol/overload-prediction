"""Compare the confirmatory stopping tests on the data of Caplin, Dean and Martin (Amendment 7) with their independent
reimplementation, and the saved results with a second run. Written, with its tolerances, before the run.

Tolerances are those of the earlier comparisons, 1e-6 for values, intervals and p values (lambdas relative). Every
quantity here is a smooth function of fitted parameters, with no rank statistic, so no looser tolerance is needed.
Decisions must agree exactly.

Usage
    python verification/compare_stopping_cdm.py --independent PATH/results.json \
        --results results/stopping_cdm [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402
from compare_search_confirmatory import TOL, _list, _summary  # noqa: E402

EXPERIMENTS = ("standard", "process")
STOP = ("value", "share", "value_time", "share_time", "value_minus_share", "value_minus_share_with_time")
GROWTH = ("observed", "predicted", "one_minus_observed", "predicted_minus_observed")
CHOICE_MODELS = ("CL", "DbS", "MLBA", "SR")
STATIC = ("CL", "DbS", "MLBA")
TIME = ("static", "SR", "SR_minus_static")
FIRST = ("observed", "predicted", "minus_half", "minus_predicted")


def block(ledger: Ledger, label: str, a: dict, b: dict) -> None:
    for k in STOP:
        _summary(ledger, "stopping", f"{label} stop {k}", a["stop"][k], b["stop"][k])
    for k in GROWTH:
        _summary(ledger, "growth", f"{label} growth {k}", a["growth"][k], b["growth"][k])
    for m in CHOICE_MODELS:
        ledger.num(
            "choices",
            f"{label} times chance {m}",
            a["choices"]["times_chance"][m],
            b["choices"]["times_chance"][m],
            TOL,
        )
    for m in STATIC:
        key = f"SR_minus_{m}_percent"
        _summary(ledger, "choices", f"{label} {key}", a["choices"][key], b["choices"][key])
    for k in TIME:
        _summary(ledger, "time", f"{label} time {k}", a["time"][k], b["time"][k])


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    P, Q = mine["pooled"], theirs["pooled"]
    block(ledger, "pooled", P, Q)
    for key in ("absolute", "relative"):
        ledger.flag("decision", f"C1 {key}", P["C1"][key], Q["C1"][key])
    ledger.flag("decision", "C2 order matters", P["C2"]["order_matters"], Q["C2"]["order_matters"])
    for key in ("noninferior", "better", "worse", "supported"):
        ledger.flag("decision", f"C3 {key}", P["C3"][key], Q["C3"][key])
    for name in EXPERIMENTS:
        A, B = mine[name], theirs[name]
        block(ledger, name, A, B)
        for k in FIRST:
            _summary(ledger, "first selection", f"{name} first {k}", A["first"][k], B["first"][k])
        for m in CHOICE_MODELS:
            _list(ledger, "parameter", f"{name} lambda {m}", A["lambda"][m], B["lambda"][m], TOL, relative=True)
        if "unsubmitted" in mine and "unsubmitted" in theirs:
            _list(ledger, "descriptive", f"{name} unsubmitted", mine["unsubmitted"][name], theirs["unsubmitted"][name])
        for key in ("observed_by_size", "looks_by_size"):
            _list(ledger, "growth", f"{name} {key}", A["growth"][key], B["growth"][key], TOL, relative=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/stopping_cdm"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()
    mine = json.loads((args.results / "summary.json").read_text())
    theirs = json.loads(args.independent.read_text())
    print("Independent reimplementation against the package, stopping tests on the data of Caplin, Dean and Martin\n")
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
