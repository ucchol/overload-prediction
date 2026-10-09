"""Compare the exploratory laptop rubric sensitivity (shares from laptop trials 1 to 4) with its independent version.

Usage
    python verification/compare_laptop_rubric_1to4.py --data PATH_TO_CSV --independent PATH/sens_laptop_1to4.json \
        --results results/exploratory/laptop_rubric_1to4 [--rerun DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_independent import Ledger, compare_h1, compare_r0, identical  # noqa: E402

from overload.experiments.sens_laptop_rubric import load_tasks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--independent", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results/exploratory/laptop_rubric_1to4"))
    ap.add_argument("--rerun", type=Path)
    args = ap.parse_args()

    tasks = {"laptop": load_tasks(args.data)["laptop"]}
    ind = {"laptop": json.loads(args.independent.read_text())}
    summary = json.loads((args.results / "summary.json").read_text())
    point = json.loads((args.results / "point.json").read_text())
    reps = {}
    with (args.results / "replicates.jsonl").open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["b"] in {0, 1, 499, 999}:
                reps[rec["b"]] = rec

    print("Independent reimplementation against the package, laptop shares from trials 1 to 4\n")
    ledger = Ledger()
    compare_r0(ledger, tasks, {"laptop": summary["R0"]}, point, reps, ind)
    compare_h1(ledger, tasks, {"laptop": summary["H1"]}, ind)
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
