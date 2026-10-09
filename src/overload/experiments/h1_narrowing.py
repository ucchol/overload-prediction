"""H1 of ANALYSIS_PLAN.md, narrowing with set size, on the development data.

A1a  held out completeness and overlap of CL, DbS_avg and MLBA_avg across the set size scaling grid,
     with the primary tests CL beta = +0.5 against 0 and CL beta = -0.5 against 0, Holm corrected per task.
A1b  elimination by aspects against CL, the 32 option trials minus the 8 option trials, in bits.
B1   the concentration gap between observed shares and held out CL predictions, its mean and its slope on
     log2 of the number of options, with the centered bootstrap of Amendment 2.

Usage
    python -m overload.experiments.h1_narrowing --data PATH_TO_CSV --out results/h1 --replicates 1000 --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import holm, summarize
from ..data import TASKS, Task, load_study
from ..engine import precompute_bases, provenance, run_replicate, subset_completeness, subset_overlap, write_json
from ..models import BETAS, Model, screening_models, set_size_models

REF = "CL|beta=+0.0"
FAMILIES = ("CL", "DbS_avg", "MLBA_avg")
PRIMARY = {"CL|beta=+0.5": +1, "CL|beta=-0.5": -1}  # predicted sign of the difference from beta = 0
SCREENING = ("EBA", "EBA_best")

_STATE: dict = {}


def models() -> list[Model]:
    return set_size_models() + screening_models()


def _name(family: str, beta: float) -> str:
    return f"{family}|beta={beta:+.1f}"


def _worker(bs: list[int]) -> list[dict]:
    t, m, b_ = _STATE["tasks"], _STATE["models"], _STATE["bases"]
    return [run_replicate(t, m, b_, b, concentration=(REF,)) for b in bs]


def run(csv: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    tasks = load_study(csv)
    ms = models()
    bases = precompute_bases(tasks, ms)
    _STATE.update(tasks=tasks, models=ms, bases=bases)
    t0 = time.time()
    point = run_replicate(tasks, ms, bases, -1, concentration=(REF,))
    order = list(range(replicates))
    if jobs > 1 and replicates:
        size = max(1, replicates // (4 * jobs))
        groups = [order[i : i + size] for i in range(0, len(order), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            reps = [r for chunk in pool.map(_worker, groups) for r in chunk]
    else:
        reps = _worker(order)
    reps.sort(key=lambda r: r["b"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "point.json", point)
    summary = summarize_h1(tasks, point, reps)
    summary["provenance"] = provenance({"replicates": replicates, "seconds": round(time.time() - t0, 1)})
    write_json(out / "summary.json", summary)
    write_tables(summary, out)
    return summary


# ----------------------------------------------------------------------------- summaries
def _by_size(task: Task) -> dict[int, list[int]]:
    sizes: dict[int, list[int]] = {}
    for j, t in enumerate(task.trials):
        sizes.setdefault(t.intended_options, []).append(j)
    return sizes


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc**2).sum())


def summarize_a1a(task: Task, name: str, point: dict, reps: list[dict]) -> dict:
    S = list(range(len(task.trials)))
    bench = point[name]["Lstar"]
    sizes = _by_size(task)

    def K(rec: dict, m: str, trials=S) -> float:
        return subset_completeness(rec, m, "ho", trials, bench)

    def ov(rec: dict, m: str) -> float:
        return subset_overlap(rec, m, "O_ho", S)

    cells: dict = {}
    for family in FAMILIES:
        base = _name(family, 0.0)
        for beta in BETAS:
            m = _name(family, beta)
            cell = {
                "completeness": summarize(K(point[name], m), [K(r[name], m) for r in reps]),
                "overlap": {"point": ov(point[name], m)},
                "completeness_by_size": {str(k): K(point[name], m, v) for k, v in sizes.items()},
                "lambda": point[name][m]["theta"],
            }
            if beta != 0.0:
                cell["minus_beta0"] = summarize(
                    K(point[name], m) - K(point[name], base),
                    [K(r[name], m) - K(r[name], base) for r in reps],
                )
                cell["overlap_minus_beta0"] = summarize(
                    ov(point[name], m) - ov(point[name], base),
                    [ov(r[name], m) - ov(r[name], base) for r in reps],
                )
            cells[m] = cell
    adjusted = holm({m: cells[m]["minus_beta0"]["p"] for m in PRIMARY})
    decisions = {}
    for m, sign in PRIMARY.items():
        d = cells[m]["minus_beta0"]
        in_direction = sign * d["point"] > 0
        clear = d["clear"] and in_direction
        overlap_same = sign * cells[m]["overlap_minus_beta0"]["point"] > 0
        decisions[m] = {
            "predicted_sign": sign,
            "clear_in_predicted_direction": bool(clear),
            "p": d["p"],
            "p_holm": adjusted[m],
            "overlap_moves_same_way": bool(overlap_same),
            "supports_H1": bool(clear and adjusted[m] < 0.05 and overlap_same),
        }
    supported = all(v["supports_H1"] for v in decisions.values())
    return {"cells": cells, "primary": decisions, "H1_supported_A1a": supported}


def summarize_a1b(task: Task, name: str, point: dict, reps: list[dict]) -> dict:
    sizes = _by_size(task)
    small, large = sizes[min(sizes)], sizes[max(sizes)]
    bench = point[name]["Lstar"]
    out: dict = {}
    for m in SCREENING:

        def delta(rec: dict, trials, m=m) -> float:
            return float(np.mean([rec[m]["ho"][t] - rec[REF]["ho"][t] for t in trials]))

        def contrast(rec: dict) -> float:
            return delta(rec, large) - delta(rec, small)

        out[m] = {
            "large_minus_small_bits": summarize(contrast(point[name]), [contrast(r[name]) for r in reps]),
            "delta_bits_by_size": {str(k): delta(point[name], v) for k, v in sizes.items()},
            "completeness": summarize(
                subset_completeness(point[name], m, "ho", range(len(task.trials)), bench),
                [subset_completeness(r[name], m, "ho", range(len(task.trials)), bench) for r in reps],
            ),
            "completeness_by_size": {
                str(k): subset_completeness(point[name], m, "ho", v, bench) for k, v in sizes.items()
            },
            "minus_CL_completeness": summarize(
                subset_completeness(point[name], m, "ho", range(len(task.trials)), bench)
                - subset_completeness(point[name], REF, "ho", range(len(task.trials)), bench),
                [
                    subset_completeness(r[name], m, "ho", range(len(task.trials)), bench)
                    - subset_completeness(r[name], REF, "ho", range(len(task.trials)), bench)
                    for r in reps
                ],
            ),
            "overlap": {"point": subset_overlap(point[name], m, "O_ho", range(len(task.trials)))},
            "lapse": point[name][m]["theta"],
        }
        d = out[m]["large_minus_small_bits"]
        out[m]["supports_H1"] = bool(d["clear"] and d["point"] > 0)
    out["H1_supported_A1b"] = out["EBA"]["supports_H1"]
    return out


def _centered(point_value: float, errors: np.ndarray) -> dict:
    """Intervals of Amendment 2: the estimate plus (percentile) or minus (basic) the error quantiles."""
    lo, hi = np.percentile(errors, [2.5, 97.5])
    pct = [point_value + lo, point_value + hi]
    bas = [point_value - hi, point_value - lo]
    world = point_value + errors
    p = min(1.0, 2.0 * min(float(np.mean(world <= 0)), float(np.mean(world >= 0))))
    clear = (pct[0] > 0 and bas[0] > 0) or (pct[1] < 0 and bas[1] < 0)
    return {"point": point_value, "percentile": pct, "basic": bas, "p": p, "clear": bool(clear)}


def summarize_b1(task: Task, name: str, point: dict, reps: list[dict]) -> dict:
    n = np.asarray(point[name]["n"], float)
    x = np.log2(n)
    G = np.asarray(point[name]["E_obs"]) / n - np.asarray(point[name][REF]["E_samp_ho"]) / n
    G0 = np.asarray(point[name]["E_samp_obs"]) / n - np.asarray(point[name][REF]["E_samp_ho"]) / n
    errs = np.array([np.asarray(r[name]["E_obs"]) / n - np.asarray(r[name][REF]["E_samp_ho"]) / n - G0 for r in reps])
    mean = _centered(float(G.mean()), errs.mean(axis=1))
    slope = _centered(_slope(x, G), np.array([_slope(x, e) for e in errs]))
    per_trial = [
        {
            "trial": t.key,
            "options": int(n[j]),
            "observed_effective_share": float(point[name]["E_obs"][j] / n[j]),
            "model_effective_share": float(point[name][REF]["E_samp_ho"][j] / n[j]),
            "gap": float(G[j]),
        }
        for j, t in enumerate(task.trials)
    ]
    supported = bool(mean["clear"] and mean["point"] < 0 and slope["clear"] and slope["point"] < 0)
    return {"mean_gap": mean, "slope_gap_per_doubling": slope, "per_trial": per_trial, "H1_supported_B1": supported}


def summarize_h1(tasks: dict[str, Task], point: dict, reps: list[dict]) -> dict:
    return {
        name: {
            "A1a": summarize_a1a(tasks[name], name, point, reps),
            "A1b": summarize_a1b(tasks[name], name, point, reps),
            "B1": summarize_b1(tasks[name], name, point, reps),
        }
        for name in TASKS
    }


# ----------------------------------------------------------------------------- tables
def _f(e: dict, scale: float = 1.0, digits: int = 1) -> str:
    s = f"{e['point'] * scale:.{digits}f}"
    if "percentile" in e:
        lo, hi = (v * scale for v in e["percentile"])
        s += f" [{lo:.{digits}f}, {hi:.{digits}f}]" + (" *" if e.get("clear") else "")
    return s


def write_tables(summary: dict, out: Path) -> None:
    lines = ["# H1, narrowing with set size (development data)", ""]
    for name in TASKS:
        s = summary[name]
        lines += [f"## {name}", "", "### A1a, set size scaling, held out completeness (%)", ""]
        lines += ["| Model | beta | Completeness | 8 | 16 | 32 | Overlap % | Minus beta 0 | Overlap minus beta 0 |"]
        lines += ["|---|---|---|---|---|---|---|---|---|"]
        for family in FAMILIES:
            for beta in BETAS:
                c = s["A1a"]["cells"][_name(family, beta)]
                by = c["completeness_by_size"]
                lines.append(
                    f"| {family} | {beta:+.1f} | {_f(c['completeness'])} | {by['8']:.1f} | {by['16']:.1f} | "
                    f"{by['32']:.1f} | {100 * c['overlap']['point']:.1f} | "
                    f"{_f(c['minus_beta0']) if 'minus_beta0' in c else ''} | "
                    f"{_f(c['overlap_minus_beta0'], 100.0) if 'overlap_minus_beta0' in c else ''} |"
                )
        lines += ["", "Primary tests:", ""]
        for m, d in s["A1a"]["primary"].items():
            lines.append(
                f"- {m} minus beta 0, predicted {'positive' if d['predicted_sign'] > 0 else 'negative'}. "
                f"Clear in that direction {d['clear_in_predicted_direction']}, Holm p {d['p_holm']:.3f}, "
                f"overlap the same way {d['overlap_moves_same_way']}, supports H1 {d['supports_H1']}."
            )
        lines += [f"- A1a supports H1 in this task: {s['A1a']['H1_supported_A1a']}", ""]
        lines += ["### A1b, elimination by aspects against CL", ""]
        for m in SCREENING:
            e = s["A1b"][m]
            by = e["delta_bits_by_size"]
            lines.append(
                f"- {m}. Completeness {_f(e['completeness'])}, minus CL {_f(e['minus_CL_completeness'])}. "
                f"Held out bits minus CL at 8, 16, 32 options {by['8']:+.3f}, {by['16']:+.3f}, {by['32']:+.3f}. "
                f"32 minus 8 {_f(e['large_minus_small_bits'], 1.0, 3)}, supports H1 {e['supports_H1']}."
            )
        b1 = s["B1"]
        lines += ["", "### B1, concentration gap (observed minus CL, effective options as a share of the set)", ""]
        lines += ["| Trial | Options | Observed | CL held out | Gap |", "|---|---|---|---|---|"]
        for t in b1["per_trial"]:
            lines.append(
                f"| {t['trial']} | {t['options']} | {100 * t['observed_effective_share']:.1f} | "
                f"{100 * t['model_effective_share']:.1f} | {100 * t['gap']:+.1f} |"
            )
        lines += [
            "",
            f"- Mean gap {_f(b1['mean_gap'], 100.0)} points, slope per doubling of options "
            f"{_f(b1['slope_gap_per_doubling'], 100.0)} points, supports H1 {b1['H1_supported_B1']}.",
            "",
        ]
    lines.append(
        "Brackets give 95% intervals from 1000 participant resamples, a star marks a value whose percentile and "
        "basic intervals both exclude zero. Completeness holds each trial's benchmark at its full sample value. "
        "B1 intervals are centered as in Amendment 2."
    )
    (out / "table_h1.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/h1"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.data, args.out, args.replicates, args.jobs)
    print(f"H1 written to {args.out} in {summary['provenance']['seconds']} s")
    print((args.out / "table_h1.md").read_text())


if __name__ == "__main__":
    main()
