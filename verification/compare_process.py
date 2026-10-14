"""Compare the overload, decision time, search model and premonition results with their independent version.

The independent results come from a separate implementation written from ``VERIFY_BRIEF_PROCESS.md`` without access
to this package. This script reads both and reports the largest absolute difference for every kind of quantity and
every value outside its tolerance. With ``--rerun-*`` it also checks that a second run of the package reproduced the
saved files.

Usage
    python verification/compare_process.py --independent PATH/results.json \
        --overload results/overload --process results/process_dev [--rerun-overload DIR --rerun-process DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import TOL, Ledger, identical  # noqa: E402

TASK_KEYS = {
    "file": ("T1_L8", "T2_H8", "T3_L16", "T4_H16", "T5_L32", "T6_H32"),
    "laptop": ("LT1_L8", "LT2_H8", "LT3_L16", "LT4_H16", "LT5_L32", "LT6_H32"),
}
SCOPE_KEYS = {**TASK_KEYS, "both": TASK_KEYS["file"] + TASK_KEYS["laptop"]}
ITEM_CODES = {"satisfied": "s1", "certain": "s2", "confused": "s3", "frustrated": "s4", "complex": "s5"}
MEASURE_CODES = {"overload": "overload", "regret": "regret", "logtime": "log_seconds_percent"}

# Documented tolerances. Point values must agree to 1e-6. Interval bounds of completeness and its differences may
# differ by up to 1e-5, since the search models' sensitivities are found by different optimizers and agree to about
# 1e-9 in relative terms. Rank correlations agree exactly in the original sample, but in a resample two forecasts or
# outcomes that are equal up to 1e-7 can be ordered differently by the two implementations, which moves a bound by one
# rank step (6 / (k (k^2 - 1)) times 2, 0.057 for six trials) and a p value by a few replicates. A difference of two
# errors can be exactly zero in a resample (when the time law's slope is zero), where rounding decides its sign.
INTERVAL_COMPLETENESS = 1e-5
RANK_STEP = 0.058
P_SLACK = 0.02


def _clear(s: dict) -> bool:
    pct, bas = s["percentile"], s["basic"]
    return bool((pct[0] > 0 and bas[0] > 0) or (pct[1] < 0 and bas[1] < 0))


def compare_summary(
    ledger: Ledger,
    kind: str,
    label: str,
    mine: dict,
    theirs: dict,
    interval_tol: float | None = None,
    p_tol: float | None = None,
) -> None:
    """Point, intervals present in both, p value and clear flag (derived from the intervals when not stored)."""
    ledger.num(kind, f"{label} point", mine["point"], theirs["point"], TOL["value"])
    itol = TOL["interval"] if interval_tol is None else interval_tol
    ikind = "interval" if interval_tol is None else f"{kind} interval"
    for which in ("percentile", "basic"):
        if which in mine and which in theirs:
            for k in range(2):
                ledger.num(ikind, f"{label} {which}[{k}]", mine[which][k], theirs[which][k], itol)
    if "p" in mine and "p" in theirs:
        ptol = TOL["p"] + 1e-12 if p_tol is None else p_tol
        ledger.num("p value" if p_tol is None else f"{kind} p", f"{label} p", mine["p"], theirs["p"], ptol)
    if "basic" in mine and "basic" in theirs:
        mc = mine.get("clear", _clear(mine))
        tc = theirs.get("clear", _clear(theirs))
        ledger.flag("clear", f"{label} clear", mc, tc)


def compare_overload(ledger: Ledger, summary: dict, ind: dict) -> None:
    M, O = summary["measures"], ind["overload"]
    for name in ("alpha_within_person", "alpha_raw", "first_component_share"):
        compare_summary(ledger, "measurement", name, M[f"measurement|{name}"], O["measurement"][name])
    for item, code in ITEM_CODES.items():
        mine = summary["point_only"]["question6_raw_correlations"][item]
        ledger.num(
            "measurement",
            f"question 6 with {item}",
            mine,
            O["measurement"]["question6_raw_correlations"][code]["point"],
            TOL["value"],
        )
    for task, keys in TASK_KEYS.items():
        for key in keys:
            for name in ("overload", "overload_vs_average", "regret", "best_chosen", "geo_seconds"):
                compare_summary(
                    ledger, "trial mean", f"{key} {name}", M[f"trial|{key}|{name}"], O["per_trial"][key][name]
                )
        for measure, code in MEASURE_CODES.items():
            for contrast in ("32_minus_8", "full_minus_brief"):
                compare_summary(
                    ledger,
                    "contrast",
                    f"{task} {measure} {contrast}",
                    M[f"contrast|{task}|{measure}|{contrast}"],
                    O["contrasts"][task][code][contrast],
                )
    for scope in ("file", "laptop", "both"):
        for pair in ("overload~regret", "overload~logtime", "regret~logtime"):
            compare_summary(
                ledger,
                "association",
                f"{scope} {pair}",
                M[f"association|{scope}|{pair}"],
                O["associations"][scope][pair],
            )


def compare_choices(ledger: Ledger, summary: dict, point: dict, ind: dict) -> None:
    for task, keys in TASK_KEYS.items():
        C, I = summary["choices"][task], ind["choices"][task]
        for model, s in C["held_out"].items():
            compare_summary(
                ledger, "completeness", f"{task} {model}", s, I["completeness"][model], INTERVAL_COMPLETENESS
            )
        for block in ("process", "against_best"):
            for name, s in C[block].items():
                compare_summary(
                    ledger, "difference", f"{task} {name}", s, I["differences"][name], INTERVAL_COMPLETENESS
                )
        for model, theirs in I["lambda_held_out"].items():
            mine = point[task]["EBA" if model == "EBA_lapse" else model]["theta_ho"]
            for j, key in enumerate(keys):
                ledger.num(
                    "parameter", f"{task} {model} {key}", mine[j], theirs[key], 1e-6 * max(1.0, abs(theirs[key]))
                )


def compare_time(ledger: Ledger, summary: dict, ind: dict) -> None:
    for task, keys in TASK_KEYS.items():
        T, I = summary["time"][task], ind["time"][task]
        for j, key in enumerate(keys):
            ledger.num("seconds", f"{task} {key}", T["seconds"][j], I["typical_seconds"][key]["point"], TOL["value"])
        for model, e in T["per_step"].items():
            theirs = I["models"][model]
            for j, key in enumerate(keys):
                ledger.num("effort", f"{task} {model} {key}", e["effort"][j], theirs["effort"][key], TOL["value"])
                sp = theirs["seconds_per_step"][key]
                ledger.num("per step", f"{task} {model} {key}", e["seconds_per_step"][j], sp["point"], TOL["value"])
                for k in range(2):
                    ledger.num(
                        "interval",
                        f"{task} {model} {key} per step [{k}]",
                        e["percentile"][k][j],
                        sp["percentile"][k],
                        TOL["interval"],
                    )
        for model, pred in T["held_out"].items():
            theirs = I["models"][model]
            for j, key in enumerate(keys):
                ledger.num(
                    "time forecast", f"{task} {model} {key}", pred[j], theirs["predicted_seconds"][key], TOL["value"]
                )
            compare_summary(ledger, "time error", f"{task} {model}", T["error"][model], theirs["typical_error"])
        for model, s in T["error_minus_constant"].items():
            compare_summary(
                ledger,
                "time error",
                f"{task} {model} minus constant",
                s,
                I["models"][model]["log_error_minus_constant"],
                TOL["interval"],
                P_SLACK,
            )


def compare_premonition(ledger: Ledger, summary: dict, ind: dict) -> None:
    for scope, keys in SCOPE_KEYS.items():
        P, I = summary["premonition"][scope], ind["premonition"][scope]
        for j, key in enumerate(keys):
            ledger.num(
                "outcome", f"{scope} overload {key}", P["overload"][j], I["overload"][key]["point"], TOL["value"]
            )
            ledger.num("outcome", f"{scope} regret {key}", P["regret"][j], I["regret"][key]["point"], TOL["value"])
        for name, values in P["forecasts"].items():
            theirs = I["forecasts"][name]
            for j, key in enumerate(keys):
                ledger.num("forecast", f"{scope} {name} {key}", values[j], theirs["values"][key]["point"], TOL["value"])
            e = P["by_forecast"][name]
            compare_summary(
                ledger, "spearman", f"{scope} {name}", e["spearman"], theirs["spearman"], RANK_STEP, P_SLACK
            )
            compare_summary(ledger, "forecast error", f"{scope} {name}", e["error"], theirs["error"])
            compare_summary(
                ledger,
                "forecast error",
                f"{scope} {name} minus constant",
                e["error_minus_constant"],
                theirs["error_minus_constant"],
                TOL["interval"],
                P_SLACK,
            )
            compare_summary(
                ledger, "forecast error", f"{scope} constant", P["constant_error"], theirs["constant_error"]
            )
        for model, values in P["regret_forecasts"].items():
            for j, key in enumerate(keys):
                ledger.num(
                    "forecast",
                    f"{scope} regret {model} {key}",
                    values[j],
                    I["regret_forecasts"][model][key]["point"],
                    TOL["value"],
                )
            e = P["by_regret_forecast"][model]
            compare_summary(ledger, "regret error", f"{scope} {model}", e["error"], I["regret_error"][model])
            compare_summary(
                ledger,
                "regret error",
                f"{scope} {model} minus constant",
                e["error_minus_constant"],
                I["regret_error_minus_constant"][model],
                TOL["interval"],
                P_SLACK,
            )
            compare_summary(
                ledger,
                "spearman",
                f"{scope} regret {model}",
                e["spearman"],
                I["regret_spearman"][model],
                RANK_STEP,
                P_SLACK,
            )
            compare_summary(
                ledger,
                "regret error",
                f"{scope} constant",
                P["regret_constant_error"],
                I["regret_constant_error"][model],
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--overload", type=Path, default=Path("results/overload"))
    ap.add_argument("--process", type=Path, default=Path("results/process_dev"))
    ap.add_argument("--rerun-overload", type=Path)
    ap.add_argument("--rerun-process", type=Path)
    args = ap.parse_args()

    ind = json.loads(args.independent.read_text())
    over = json.loads((args.overload / "summary.json").read_text())
    proc = json.loads((args.process / "summary.json").read_text())
    point = json.loads((args.process / "point.json").read_text())

    print("Independent reimplementation against the package, overload, time, search models and premonition\n")
    ledger = Ledger()
    compare_overload(ledger, over, ind)
    compare_choices(ledger, proc, point, ind)
    compare_time(ledger, proc, ind)
    compare_premonition(ledger, proc, ind)
    failures = ledger.report()

    for label, saved, rerun, files in (
        ("overload", args.overload, args.rerun_overload, ("summary.json",)),
        ("process", args.process, args.rerun_process, ("summary.json", "point.json")),
    ):
        if rerun is None:
            continue
        print(f"\nSecond run of the package, {label}")
        for fname in files:
            same, _ = identical(saved / fname, rerun / fname)
            print(f"  {fname}: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
        if (saved / "replicates.jsonl").exists():
            r1, r2 = saved / "replicates.jsonl", rerun / "replicates.jsonl"
            same = [json.loads(a) for a in r1.open()] == [json.loads(b) for b in r2.open()]
            print(f"  replicates.jsonl: {'identical' if same else 'DIFFERENT'}")
            failures += 0 if same else 1
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
