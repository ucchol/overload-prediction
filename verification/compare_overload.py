"""Compare the confirmatory overload forecasts (Amendment 8) with their independent reimplementation, and the saved
results with a second run. Written, with its tolerances, before the run.

Tolerances are those of the earlier comparisons, 1e-6 for values, intervals and p values (lambdas relative). The area
under the curve is a rank statistic of forecasts that are smooth in the fitted sensitivity, so a sensitivity differing
in the eighth digit can reorder two nearly equal forecasts. It gets 1e-3, and its p values 0.006, as the rank
correlations did. Decisions must agree exactly.

Usage
    python verification/compare_overload.py --independent PATH/results.json \
        --results results/overload_premonition [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402
from compare_search_confirmatory import TOL, TOL_P_RANK, TOL_RANK, _list, _summary  # noqa: E402

MODELS = ("CL", "CL_comp", "SR")
AUC = (*MODELS, "near_best", "minus_gap")


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    for ds, A in mine["datasets"].items():
        B = theirs["datasets"][ds]
        for k in ("observed", *MODELS):
            _summary(ledger, "growth", f"{ds} growth {k}", A["growth"][k], B["growth"][k])
        for m in MODELS:
            _summary(ledger, "growth", f"{ds} error {m}", A["error"][m], B["error"][m])
        _summary(ledger, "growth", f"{ds} comp minus CL error", A["comp_minus_cl_error"], B["comp_minus_cl_error"])
        for k in AUC:
            _summary(ledger, "auc", f"{ds} auc {k}", A["auc"][k], B["auc"][k], TOL_RANK, TOL_P_RANK)
        for k in ("sr_auc_minus_half", "sr_auc_minus_near_best"):
            _summary(ledger, "auc", f"{ds} {k}", A[k], B[k], TOL_RANK, TOL_P_RANK)
        ledger.flag("decision", f"{ds} O1 overload", A["O1"]["verdict"] == "overload", B["O1"]["verdict"] == "overload")
        ledger.flag(
            "decision",
            f"{ds} O1 compensation",
            A["O1"]["verdict"] == "compensation",
            B["O1"]["verdict"] == "compensation",
        )
        ledger.flag("decision", f"{ds} O2", A["O2"]["supported"], B["O2"]["supported"])
    for name, P in mine["panels"].items():
        Q = theirs["panels"][name]
        for k in ("observed", *MODELS):
            _list(ledger, "rates", f"{name} rate {k}", P["rate"][k], Q["rate"][k])
        for m in MODELS:
            _list(ledger, "parameter", f"{name} lambda {m}", P["lambda"][m], Q["lambda"][m], TOL, relative=True)
        if "miss_best_unseen" in P:
            _list(ledger, "rates", f"{name} misses of unseen best", P["miss_best_unseen"], Q["miss_best_unseen"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/overload_premonition"))
    ap.add_argument("--rerun", type=Path)
    a = ap.parse_args()
    mine = json.loads((a.results / "summary.json").read_text())
    theirs = json.loads(a.independent.read_text())
    print("Independent reimplementation against the package, forecasting choice overload\n")
    ledger = Ledger()
    compare(ledger, mine, theirs)
    failures = ledger.report()
    if a.rerun:
        print("\nSecond run of the package")
        for fname in ("summary.json", "point.json"):
            same, _ = identical(a.results / fname, a.rerun / fname)
            print(f"  {fname}: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
        same = [json.loads(x) for x in (a.results / "replicates.jsonl").open()] == [
            json.loads(x) for x in (a.rerun / "replicates.jsonl").open()
        ]
        print(f"  replicates.jsonl: {'identical' if same else 'DIFFERENT'}")
        failures += 0 if same else 1
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
