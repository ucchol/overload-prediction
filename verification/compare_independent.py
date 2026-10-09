"""Compare the package's R0 and H1 results with an independent reimplementation, and two runs of the package.

The independent results come from a separate implementation written from ANALYSIS_PLAN.md and a written brief,
without access to this package (see verification/README.md). This script reads both sets of results and reports
the largest absolute difference for every kind of quantity and every value outside its tolerance.

Usage
    python verification/compare_independent.py --data PATH_TO_CSV --independent PATH/results.json \
        --baseline results/baseline --h1 results/h1 [--rerun-baseline DIR --rerun-h1 DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from overload.data import load_study
from overload.engine import subset_completeness

TOL = {"value": 1e-6, "interval": 1e-6, "p": 0.0}


class Ledger:
    def __init__(self) -> None:
        self.max_diff: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)
        self.failures: list[str] = []

    def num(self, kind: str, label: str, mine: float, theirs: float, tol: float) -> None:
        d = abs(float(mine) - float(theirs))
        if math.isnan(d):
            d = math.inf
        self.max_diff[kind] = max(self.max_diff[kind], d)
        self.count[kind] += 1
        if d > tol:
            self.failures.append(f"{kind:<14} {label}: package {mine!r} independent {theirs!r} (diff {d:.3g})")

    def flag(self, kind: str, label: str, mine: bool, theirs: bool) -> None:
        self.count[kind] += 1
        if bool(mine) != bool(theirs):
            self.max_diff[kind] = 1.0
            self.failures.append(f"{kind:<14} {label}: package {mine} independent {theirs}")

    def summary(self, kind: str, label: str, mine: dict, theirs: dict, with_test: bool = True) -> None:
        self.num(kind, f"{label} point", mine["point"], theirs["point"], TOL["value"])
        for which in ("percentile", "basic"):
            if which in mine and which in theirs:
                for k in range(2):
                    self.num("interval", f"{label} {which}[{k}]", mine[which][k], theirs[which][k], TOL["interval"])
        if with_test and "p" in mine and "p" in theirs:
            self.num("p value", f"{label} p", mine["p"], theirs["p"], TOL["p"] + 1e-12)
            self.flag("clear", f"{label} clear", mine["clear"], theirs["clear"])

    def report(self) -> int:
        print("Kind            checked   largest difference")
        for kind in sorted(self.count):
            print(f"{kind:<15} {self.count[kind]:>7}   {self.max_diff[kind]:.3g}")
        if self.failures:
            print(f"\n{len(self.failures)} values outside tolerance:")
            for f in self.failures:
                print("  " + f)
        else:
            print("\nEvery value agrees within tolerance.")
        return len(self.failures)


def compare_r0(ledger: Ledger, tasks, base_sum: dict, base_point: dict, reps: dict, ind: dict) -> None:
    for name, task in tasks.items():
        keys = task.keys
        mine_r = dict(zip(task.attributes, task.shares, strict=True))
        for a, v in ind[name]["rubric_shares"].items():
            ledger.num("rubric share", f"{name} {a}", mine_r[a], v, 1e-10)
        counts = task.counts()
        for j, t in enumerate(task.trials):
            p = counts[j] / counts[j].sum()
            theirs = ind[name]["shares"][t.key]
            for i, o in enumerate(t.options):
                ledger.num("choice share", f"{name} {t.key} {o}", p[i], theirs[o], 1e-12)
            ledger.num(
                "benchmark", f"{name} {t.key} L*", base_point[name]["Lstar"][j], ind[name]["Lstar"][t.key], 1e-10
            )
        bench = base_point[name]["Lstar"]
        for m, mine in base_sum[name]["all"].items():
            theirs = ind[name]["R0"][m]
            pt = base_point[name][m]
            lab = f"{name} {m}"
            ledger.num("parameter", f"{lab} lambda", pt["theta"], theirs["lambda"], 1e-6 * max(1.0, abs(pt["theta"])))
            for j, k in enumerate(keys):
                ledger.num("trial bits", f"{lab} {k} held out", pt["ho"][j], theirs["ho_bits"][k], TOL["value"])
                ledger.num("trial bits", f"{lab} {k} fitted", pt["fit"][j], theirs["fit_bits"][k], TOL["value"])
                ledger.num("trial overlap", f"{lab} {k}", pt["O_ho"][j], theirs["overlap_ho"][k], TOL["value"])
            ledger.summary(
                "completeness",
                f"{lab} held out",
                mine["completeness_held_out"],
                theirs["completeness_ho"],
                with_test=False,
            )
            ledger.num(
                "completeness",
                f"{lab} fitted",
                mine["completeness_fitted"]["point"],
                theirs["completeness_fit"],
                TOL["value"],
            )
            for level in ("low", "high"):
                ledger.num(
                    "completeness",
                    f"{lab} held out {level}",
                    base_sum[name][level][m]["completeness_held_out"]["point"],
                    theirs[f"completeness_ho_{level}"],
                    TOL["value"],
                )
            ledger.num(
                "overlap",
                f"{lab} held out mean",
                mine["overlap_held_out"]["point"],
                theirs["overlap_ho_mean"],
                TOL["value"],
            )
            if "completeness_minus_CL" in mine:
                ledger.summary("difference", f"{lab} minus CL", mine["completeness_minus_CL"], theirs["minus_CL"])
                ledger.summary(
                    "difference", f"{lab} overlap minus CL", mine["overlap_minus_CL"], theirs["overlap_minus_CL"]
                )
            for b, rec in reps.items():
                if str(b) in theirs.get("replicates", {}):
                    ledger.num(
                        "replicate",
                        f"{lab} b={b}",
                        subset_completeness(rec[name], m, "ho", range(len(keys)), bench),
                        theirs["replicates"][str(b)],
                        TOL["value"],
                    )


def compare_h1(ledger: Ledger, tasks, h1_sum: dict, ind: dict) -> None:
    for name in tasks:
        mine, theirs = h1_sum[name], ind[name]["H1"]
        for m, cell in mine["A1a"]["cells"].items():
            t = theirs["A1a"][m]
            lab = f"{name} {m}"
            ledger.summary("completeness", f"{lab} held out", cell["completeness"], t["completeness"], with_test=False)
            for size, v in cell["completeness_by_size"].items():
                ledger.num("completeness", f"{lab} at {size}", v, t["by_size"][size], TOL["value"])
            t_ov = t["overlap"]["point"] if isinstance(t["overlap"], dict) else t["overlap"]
            ledger.num("overlap", f"{lab} held out", cell["overlap"]["point"], t_ov, TOL["value"])
            if "minus_beta0" in cell:
                ledger.summary("difference", f"{lab} minus beta 0", cell["minus_beta0"], t["minus_beta0"])
                t_od = t["overlap_minus_beta0"]
                t_od = t_od["point"] if isinstance(t_od, dict) else t_od
                ledger.num(
                    "difference",
                    f"{lab} overlap minus beta 0",
                    cell["overlap_minus_beta0"]["point"],
                    t_od,
                    TOL["value"],
                )
        for m, d in mine["A1a"]["primary"].items():
            t = theirs["A1a"]["primary"][m]
            ledger.num("p value", f"{name} {m} Holm", d["p_holm"], t["p_holm"], 1e-12)
            ledger.flag("decision", f"{name} {m} supports H1", d["supports_H1"], t["supports_H1"])
            ledger.flag("decision", f"{name} {m} overlap same way", d["overlap_moves_same_way"], t["overlap_same_way"])
        ledger.flag("decision", f"{name} A1a", mine["A1a"]["H1_supported_A1a"], theirs["A1a"]["supported"])
        for m in ("EBA", "EBA_best"):
            d, t = mine["A1b"][m], theirs["A1b"][m]
            lab = f"{name} {m}"
            ledger.num("parameter", f"{lab} lapse", d["lapse"], t["lapse"], 1e-6)
            ledger.summary("completeness", f"{lab} held out", d["completeness"], t["completeness"], with_test=False)
            ledger.summary("difference", f"{lab} minus CL", d["minus_CL_completeness"], t["minus_CL"])
            for size, v in d["delta_bits_by_size"].items():
                ledger.num("trial bits", f"{lab} delta at {size}", v, t["delta_bits_by_size"][size], TOL["value"])
            ledger.summary("difference", f"{lab} 32 minus 8", d["large_minus_small_bits"], t["large_minus_small_bits"])
        ledger.flag("decision", f"{name} A1b", mine["A1b"]["H1_supported_A1b"], theirs["A1b"]["supported"])
        b1, tb1 = mine["B1"], theirs["B1"]
        for row in b1["per_trial"]:
            tr = tb1["per_trial"][row["trial"]]
            ledger.num(
                "concentration",
                f"{name} {row['trial']} observed",
                row["observed_effective_share"],
                tr["observed_share"],
                1e-12,
            )
            ledger.num(
                "concentration", f"{name} {row['trial']} CL", row["model_effective_share"], tr["model_share"], 1e-9
            )
            ledger.num("concentration", f"{name} {row['trial']} gap", row["gap"], tr["gap"], 1e-9)
        ledger.summary("concentration", f"{name} B1 mean gap", b1["mean_gap"], tb1["mean_gap"])
        ledger.summary("concentration", f"{name} B1 slope", b1["slope_gap_per_doubling"], tb1["slope"])
        ledger.flag("decision", f"{name} B1", b1["H1_supported_B1"], tb1["supported"])


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k != "provenance"}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def identical(a: Path, b: Path) -> tuple[bool, str]:
    x, y = strip(json.loads(a.read_text())), strip(json.loads(b.read_text()))
    return (x == y), f"{a.name}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=Path("results/baseline"))
    ap.add_argument("--h1", type=Path, default=Path("results/h1"))
    ap.add_argument("--rerun-baseline", type=Path)
    ap.add_argument("--rerun-h1", type=Path)
    args = ap.parse_args()

    tasks = load_study(args.data)
    ind = json.loads(args.independent.read_text())
    base_sum = json.loads((args.baseline / "summary.json").read_text())
    base_point = json.loads((args.baseline / "point.json").read_text())
    h1_sum = json.loads((args.h1 / "summary.json").read_text())
    reps = {}
    rep_file = args.baseline / "replicates.jsonl"
    if rep_file.exists():
        wanted = {0, 1, 499, 999}
        for line in rep_file.open():
            rec = json.loads(line)
            if rec["b"] in wanted:
                reps[rec["b"]] = rec

    print("Independent reimplementation against the package\n")
    ledger = Ledger()
    compare_r0(ledger, tasks, base_sum, base_point, reps, ind)
    compare_h1(ledger, tasks, h1_sum, ind)
    failures = ledger.report()

    for label, first, second in (("baseline", args.baseline, args.rerun_baseline), ("H1", args.h1, args.rerun_h1)):
        if second is None:
            continue
        print(f"\nSecond run of the package, {label}")
        for fname in ("summary.json", "point.json"):
            same, _ = identical(first / fname, second / fname)
            print(f"  {fname}: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
        r1, r2 = first / "replicates.jsonl", second / "replicates.jsonl"
        if r1.exists() and r2.exists():
            same = [json.loads(a) for a in r1.open()] == [json.loads(b) for b in r2.open()]
            print(f"  replicates.jsonl: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
