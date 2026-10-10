"""Withdrawn exploratory sensitivity (plan, Amendment 4). Laptop rubric shares from laptop trials 1 to 4.

It was run with the earlier laptop definitions, whose errors in trials 5 and 6 caused the inconsistency it probed.
Kept for the record. Its results are in results/superseded.

In laptop trials 5 and 6 the designers' rubric scores contradict the coded attributes (verification/README.md).
The rubric shares of every laptop model come from a regression of the rubric scores on the coded attributes over
all six trials, so those two trials can move every laptop result. This experiment re-estimates the laptop shares
from trials 1 to 4 only, keeps everything else as in the plan, and reruns R0 and H1.

Usage
    python -m overload.experiments.sens_laptop_rubric --data PATH_TO_CSV --out results/superseded/laptop_rubric_1to4
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..data import _SHARE_ATTRIBUTES, load_study, rubric_shares
from ..engine import precompute_bases, provenance, run_replicate, write_json
from ..models import baseline_models, screening_models, set_size_models
from . import baseline, h1_narrowing

TRIALS_USED = (0, 1, 2, 3)
_STATE: dict = {}


def load_tasks(csv: Path) -> dict:
    tasks = load_study(csv, laptop="aamas")
    laptop = tasks["laptop"]
    shares, r2 = rubric_shares([laptop.trials[j] for j in TRIALS_USED], _SHARE_ATTRIBUTES["laptop"])
    laptop.shares, laptop.share_r2 = shares, r2
    return tasks


def _worker(bs: list[int]) -> list[dict]:
    t, m, b_ = _STATE["tasks"], _STATE["models"], _STATE["bases"]
    return [run_replicate(t, m, b_, b, concentration=(h1_narrowing.REF,)) for b in bs]


def run(csv: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    tasks = load_tasks(csv)
    models = baseline_models() + set_size_models() + screening_models()
    bases = precompute_bases(tasks, models)
    _STATE.update(tasks=tasks, models=models, bases=bases)
    t0 = time.time()
    point = run_replicate(tasks, models, bases, -1, concentration=(h1_narrowing.REF,))
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
    laptop = tasks["laptop"]
    summary = {
        "laptop_shares": dict(zip(laptop.attributes, laptop.shares.tolist(), strict=True)),
        "laptop_share_r2": laptop.share_r2,
        "R0": baseline.summarize_results(tasks, [m.name for m in baseline_models()], point, reps)["laptop"],
        "H1": h1_narrowing.summarize_h1(tasks, point, reps)["laptop"],
        "provenance": provenance(
            {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/superseded/laptop_rubric_1to4"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.data, args.out, args.replicates, args.jobs)
    print(f"written to {args.out} in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
