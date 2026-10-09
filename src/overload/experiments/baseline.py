"""Result R0 of ANALYSIS_PLAN.md, the baseline.

Held out and fitted completeness and held out overlap of QR, CL, DbS_avg and MLBA_avg, with the summed
comparison versions for continuity, per task, over all six trials and within each information level,
each with its difference from CL and participant bootstrap intervals.

Usage
    python -m overload.experiments.baseline --data PATH_TO_CSV --out results/baseline --replicates 1000
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import summarize
from ..data import TASKS, Task, load_study
from ..engine import precompute_bases, provenance, run_replicate, subset_completeness, subset_overlap, write_json
from ..models import Model, baseline_models

REFERENCE = "CL"
SUBSETS = ("all", "low", "high")

# Shared with worker processes through fork, so the data are loaded once.
_STATE: dict = {}


def _subset(task: Task, which: str) -> list[int]:
    return list(range(len(task.trials))) if which == "all" else task.trials_at(which)


def _worker(bs: Sequence[int]) -> list[dict]:
    tasks, models, bases = _STATE["tasks"], _STATE["models"], _STATE["bases"]
    return [run_replicate(tasks, models, bases, b) for b in bs]


def run(csv: Path, out: Path, replicates: int = 1000, jobs: int = 1, models: list[Model] | None = None) -> dict:
    tasks = load_study(csv)
    models = baseline_models() if models is None else models
    bases = precompute_bases(tasks, models)
    _STATE.update(tasks=tasks, models=models, bases=bases)

    t0 = time.time()
    point = run_replicate(tasks, models, bases, -1, alone=True)
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

    summary = summarize_results(tasks, [m.name for m in models], point, reps)
    summary["provenance"] = provenance({"replicates": replicates, "seconds": round(time.time() - t0, 1)})
    write_json(out / "summary.json", summary)
    write_tables(summary, out)
    return summary


def summarize_results(tasks: dict[str, Task], names: list[str], point: dict, reps: list[dict]) -> dict:
    """Completeness, overlap and differences from CL, with intervals, for every task and subset."""
    out: dict = {}
    for task_name in TASKS:
        task = tasks[task_name]
        out[task_name] = {}
        for which in SUBSETS:
            S = _subset(task, which)
            block: dict = {}
            for m in names:
                p_ho = subset_completeness(point[task_name], m, "ho", S)
                p_fit = subset_completeness(point[task_name], m, "fit", S)
                p_ov = subset_overlap(point[task_name], m, "O_ho", S)
                b_ho = [subset_completeness(r[task_name], m, "ho", S) for r in reps]
                b_ov = [subset_overlap(r[task_name], m, "O_ho", S) for r in reps]
                entry = {
                    "completeness_held_out": summarize(p_ho, b_ho) if reps else {"point": p_ho},
                    "completeness_fitted": {"point": p_fit},
                    "overlap_held_out": summarize(p_ov, b_ov) if reps else {"point": p_ov},
                }
                if m != REFERENCE:
                    d_ho = p_ho - subset_completeness(point[task_name], REFERENCE, "ho", S)
                    d_ov = p_ov - subset_overlap(point[task_name], REFERENCE, "O_ho", S)
                    b_dho = [
                        subset_completeness(r[task_name], m, "ho", S)
                        - subset_completeness(r[task_name], REFERENCE, "ho", S)
                        for r in reps
                    ]
                    b_dov = [
                        subset_overlap(r[task_name], m, "O_ho", S) - subset_overlap(r[task_name], REFERENCE, "O_ho", S)
                        for r in reps
                    ]
                    entry["completeness_minus_CL"] = summarize(d_ho, b_dho) if reps else {"point": d_ho}
                    entry["overlap_minus_CL"] = summarize(d_ov, b_dov) if reps else {"point": d_ov}
                block[m] = entry
            out[task_name][which] = block
    return out


def _fmt(entry: dict, scale: float = 1.0, digits: int = 1) -> str:
    p = entry["point"] * scale
    if "percentile" not in entry:
        return f"{p:.{digits}f}"
    lo, hi = (x * scale for x in entry["percentile"])
    mark = " *" if entry.get("clear") else ""
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]{mark}"


def write_tables(summary: dict, out: Path) -> None:
    """Markdown and CSV versions of the baseline table."""
    rows = []
    header = (
        "| Task | Model | Held out completeness, % | Low | High | Fitted, % | Held out overlap, % "
        "| Completeness minus CL, points | Overlap minus CL, points |"
    )
    lines = [header, "|" + "---|" * 9]
    for task_name in TASKS:
        for m, e in summary[task_name]["all"].items():
            low = summary[task_name]["low"][m]["completeness_held_out"]
            high = summary[task_name]["high"][m]["completeness_held_out"]
            dc = e.get("completeness_minus_CL")
            do = e.get("overlap_minus_CL")
            cells = [
                task_name,
                m,
                _fmt(e["completeness_held_out"]),
                _fmt(low),
                _fmt(high),
                _fmt(e["completeness_fitted"]),
                _fmt(e["overlap_held_out"], 100.0),
                _fmt(dc) if dc else "",
                _fmt(do, 100.0) if do else "",
            ]
            lines.append("| " + " | ".join(cells) + " |")
            rows.append(cells)
    note = (
        "\nBrackets give 95% percentile intervals from the participant bootstrap. "
        "A star marks a difference whose percentile and basic intervals both exclude zero.\n"
    )
    (out / "table_baseline.md").write_text("\n".join(lines) + "\n" + note)
    csv_lines = [",".join(h.strip() for h in header.strip("|").split("|"))]
    csv_lines += [",".join('"' + c + '"' for c in r) for r in rows]
    (out / "table_baseline.csv").write_text("\n".join(csv_lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="path to the study's CSV file")
    ap.add_argument("--out", type=Path, default=Path("results/baseline"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.data, args.out, args.replicates, args.jobs)
    secs = summary["provenance"]["seconds"]
    print(f"baseline written to {args.out} in {secs} s")
    print((args.out / "table_baseline.md").read_text())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
