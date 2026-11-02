"""Compare the blind reimplementation of Amendment 12 with the package. Point values from POINT_JSON (the package's
point.json or a point computation), replicates 0 to B-1 from the package's replicates.jsonl when it exists."""
import json, sys
from pathlib import Path
pt = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/time_structure/point.json"))
d = json.load(open("verification/independent_time/independent_results.json"))
SETS = list(d["point"])
worst, worst_k = 0.0, ""
for s in SETS:
    for law in ("S1", "S2", "S3", "S4"):
        dk = abs(d["point"][s][law] - pt[f"{s}|R2|{law}"])
        if dk > worst: worst, worst_k = dk, f"{s} R2 {law}"
    for t in ("T1", "T2", "T3", "S4_minus_S2"):
        key = f"{s}|{t}"
        dk = abs(d["point"][s][t] - pt[key])
        if dk > worst: worst, worst_k = dk, f"{s} {t}"
    for j, f in enumerate(("log2n", "best", "gap", "mean_rest")):
        dk = abs(d["point"][s]["transfer_coef"][j] - pt[f"{s}|transfer_coef|{f}"])
        if dk > worst: worst, worst_k = dk, f"{s} transfer {f}"
print(f"point values, {len(SETS)} sets, max difference {worst:.1e} ({worst_k})")
rep = Path("results/time_structure/replicates.jsonl")
if rep.exists():
    R = [json.loads(l) for l in open(rep)]
    w2 = 0.0
    for s in SETS:
        for t in ("T1", "T2", "T3"):
            for b, v in enumerate(d["replicates"][s][t]):
                w2 = max(w2, abs(v - R[b][f"{s}|{t}"]))
    print(f"replicates 0 to {len(d['replicates'][SETS[0]]['T1']) - 1}, T1 to T3, max difference {w2:.1e}")
