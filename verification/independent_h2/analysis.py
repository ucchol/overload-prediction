"""Run the blind H2 (Amendment 11) analysis with a resumable participant bootstrap."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

DATASETS = ("snack", "reutskaja")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    part_dir = os.path.join(here, "partial")
    os.makedirs(part_dir, exist_ok=True)

    sn, P_s = core.load_snack(os.path.join(a.repo, "data/snack"), os.path.join(a.repo, "data/snack_gaze"))
    re_, P_r = core.load_reutskaja(os.path.join(a.repo, "data/reutskaja/fixations_data.dta"))
    data = {"snack": (sn, P_s), "reutskaja": (re_, P_r)}

    point_path = os.path.join(part_dir, "point.json")
    if os.path.exists(point_path):
        with open(point_path) as f:
            point = json.load(f)
    else:
        point = {d: core.analyse(g, np.ones(P, int)) for d, (g, P) in data.items()}
        with open(point_path, "w") as f:
            json.dump(point, f)
    print("point done", flush=True)

    rep_path = os.path.join(part_dir, "replicates.jsonl")
    done = {}
    if os.path.exists(rep_path):
        with open(rep_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["b"]] = r
    for b in range(a.B):
        if b in done:
            continue
        rng = np.random.default_rng(110000 + b)
        counts = {}
        for d in DATASETS:  # snack first, then Reutskaja
            P = data[d][1]
            counts[d] = np.bincount(rng.integers(0, P, size=P), minlength=P)
        r = {"b": b}
        for d in DATASETS:
            r[d] = core.analyse(data[d][0], counts[d])
        with open(rep_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        done[b] = r
        print(f"replicate {b} done", flush=True)

    reps = [done[b] for b in range(a.B)]
    res = {"amendment": 11, "B": a.B, "seed_base": 110000,
           "notes": "score = weighted mean log2 q(chosen) over all trials, leave one set size out; "
                    "gains in bits per choice; see VERIFY_BRIEF_H2.md",
           "datasets": {}}
    for d in DATASETS:
        keys = list(point[d].keys())
        res["datasets"][d] = {k: core.summarize(point[d][k], [r[d][k] for r in reps]) for k in keys}
        ds = res["datasets"][d]
        ds["H2_supported"] = bool(ds["return_gain"]["clear_positive"] and
                                  ds["return_minus_dwell"]["clear_positive"])
    res["H2_supported_overall"] = all(res["datasets"][d]["H2_supported"] for d in DATASETS)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for d in DATASETS:
        ds = res["datasets"][d]
        for k in ("CL_log2_score", "CL_bits_above_uniform", "return_gain", "dwell_gain", "last_gain",
                  "return_minus_dwell", "return_beta", "dwell_beta", "last_beta"):
            s = ds[k]
            print(f"{d:10s} {k:22s} {s['point']:+.4f} pct [{s['percentile'][0]:+.4f}, {s['percentile'][1]:+.4f}]"
                  f" basic [{s['basic'][0]:+.4f}, {s['basic'][1]:+.4f}] clear {s['clear']}")
        print(d, "H2 supported", ds["H2_supported"])
    print("overall", res["H2_supported_overall"])


if __name__ == "__main__":
    main()
