"""Run the independent Amendment 12 analysis: point values on the original sample, then B bootstrap replicates.

Partial results are written after every replicate to <out>.partial and the final file replaces it at the end.
Usage: python -I run_time.py --B 20 --out independent_results.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import time_core as tc  # noqa: E402
from overload.bootstrap import holm, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "independent_results.json"))
    a = ap.parse_args()
    partial = a.out + ".partial"

    t0 = time.time()
    data = tc.load_all()
    prep = {k: tc.prepare(v) for k, v in data.items()}
    ones = {k: np.ones(len(prep[k]["y"])) for k in tc.SETS}

    point = tc.analyse(prep, ones, want_lams=True)
    for name in tc.SETS:
        d = prep[name]
        point[name]["S2_coef"] = tc.wls(d["X"], d["y"], ones[name]).tolist()
        point[name]["fold_lambdas_sr_cl"] = [list(map(float, p)) for p in point[name]["fold_lambdas_sr_cl"]]
    out = {
        "plan": "ANALYSIS_PLAN.md Amendment 12",
        "feature_order": ["log2 n", "best", "gap best minus second", "mean of rest"],
        "B": a.B,
        "point": point,
        "replicates": {name: {"T1": [], "T2": [], "T3": []} for name in tc.SETS},
    }
    print(f"point done in {time.time() - t0:.0f} s", flush=True)
    for name in tc.SETS:
        p = point[name]
        print(f"  {name}: T1 {p['T1']:.4f} T2 {p['T2']:.4f} T3 {p['T3']:.4f}", flush=True)

    def dump(path):
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)

    dump(partial)
    for b in range(a.B):
        res = tc.analyse(prep, tc.replicate_weights(prep, b))
        for name in tc.SETS:
            for t in ("T1", "T2", "T3"):
                out["replicates"][name][t].append(res[name][t])
        out["replicates_done"] = b + 1
        dump(partial)
        print(f"replicate {b} done at {time.time() - t0:.0f} s", flush=True)

    tests = {}
    for t in ("T1", "T2", "T3"):
        summ = {name: summarize(point[name][t], out["replicates"][name][t]) for name in tc.SETS}
        adj = holm({name: summ[name]["p"] for name in tc.SETS})
        for name in tc.SETS:
            summ[name]["p_holm"] = adj[name]
            summ[name]["supported"] = bool(
                summ[name]["percentile"][0] > 0 and summ[name]["basic"][0] > 0 and adj[name] < 0.05
            )
        tests[t] = summ
    out["tests"] = tests
    dump(a.out)
    os.remove(partial)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
