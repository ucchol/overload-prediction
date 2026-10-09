"""Compare the confirmatory H1 results on the snack data with their independent reimplementation and a second run.

Usage
    python verification/compare_h1_confirmatory.py --independent PATH/results.json \
        --results results/h1_confirmatory [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, identical  # noqa: E402

# Elasticities come from a profile search with a tolerance near 1e-9 in beta, every other value from root finding
# to machine precision.
TOL_VALUE = 1e-9
TOL_BETA = 1e-6


def compare(ledger: Ledger, mine: dict, theirs: dict) -> None:
    c2, t2 = mine["C2"], theirs["C2"]
    for m, e in c2["cells"].items():
        t = t2["cells"][m]
        ledger.summary("held-out L", f"{m} L", e["L"], t["L"], with_test=False)
        for k, (tc, tl) in enumerate(zip(e["times_chance_by_size"].values(), t["by_size"], strict=True)):
            ledger.num("held-out L", f"{m} size {k}", math.log2(tc), tl, TOL_VALUE)
        for k, (a, b) in enumerate(zip(e["lambda_held_out"], t["lambda_held_out"], strict=True)):
            ledger.num("parameter", f"{m} lambda fold {k}", a, b, TOL_VALUE * max(1.0, abs(a)))
        if "minus_beta0" in e:
            ledger.summary("difference", f"{m} minus beta 0", e["minus_beta0"], t["minus_beta0"])
    for m, d in c2["primary"].items():
        ledger.num("p value", f"{m} Holm", d["p_holm"], t2["primary"][m]["p_holm"], 1e-12)
        ledger.flag("decision", f"{m} supports H1", d["supports_H1"], t2["primary"][m]["supports_H1"])
    ledger.flag("decision", "C2", c2["H1_supported_C2"], t2["supported"])

    c1, t1 = mine["C1"], theirs["C1"]
    for row, trow in zip(c1["per_size"], t1["per_size"], strict=True):
        n = row["set_size"]
        ledger.num("concentration", f"C1 {n} observed", row["observed_effective_share"], trow["observed_share"], 1e-12)
        ledger.num("concentration", f"C1 {n} model", row["model_effective_share"], trow["model_share"], TOL_VALUE)
        ledger.num("concentration", f"C1 {n} gap", row["gap"], trow["gap"], TOL_VALUE)
    ledger.summary("concentration", "C1 mean gap", c1["mean_gap"], t1["mean_gap"])
    ledger.summary("concentration", "C1 slope", c1["slope_gap_per_doubling"], t1["slope"])
    ledger.flag("decision", "C1", c1["H1_supported_C1"], t1["supported"])

    c3, t3 = mine["C3"], theirs["C3"]
    for key in ("beta_all", "beta_seen", "beta_all_minus_seen"):
        a, b = c3[key], t3[key]
        ledger.num("elasticity", f"C3 {key} point", a["point"], b["point"], TOL_BETA)
        for which in ("percentile", "basic"):
            for k in range(2):
                ledger.num("elasticity", f"C3 {key} {which}[{k}]", a[which][k], b[which][k], TOL_BETA)
        ledger.num("p value", f"C3 {key} p", a["p"], b["p"], 1e-12)
        ledger.flag("clear", f"C3 {key} clear", a["clear"], b["clear"])
    for key in ("lambda_all", "lambda_seen"):
        ledger.num("elasticity", f"C3 {key}", c3[key], t3[key], TOL_BETA)
    ledger.flag("decision", "C3 narrowing", c3["narrowing_supported"], t3["narrowing_supported"])
    ledger.flag("decision", "C3 noise", c3["noise_supported"], t3["noise_supported"])

    d, td = mine["descriptives"], theirs["descriptives"]
    for key in ("share_looked_at", "share_chosen_looked_at"):
        for k, (a, b) in enumerate(zip(d[key], td[key], strict=True)):
            ledger.num("descriptive", f"{key} {k} point", a["point"], b["point"], 1e-12)
            for j in range(2):
                ledger.num("descriptive", f"{key} {k} percentile[{j}]", a["percentile"][j], b["percentile"][j], 1e-12)
    for name, rows in d["lambda_alone"].items():
        for k, (a, b) in enumerate(zip(rows, td["lambda_alone"][name], strict=True)):
            ledger.num("parameter", f"lambda alone {name} {k}", a["point"], b["point"], TOL_VALUE)
            for j in range(2):
                ledger.num(
                    "interval", f"lambda alone {name} {k} [{j}]", a["percentile"][j], b["percentile"][j], TOL_VALUE
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/h1_confirmatory"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()
    mine = json.loads((args.results / "summary.json").read_text())
    theirs = json.loads(args.independent.read_text())
    print("Independent reimplementation against the package, confirmatory H1 on the snack data\n")
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
