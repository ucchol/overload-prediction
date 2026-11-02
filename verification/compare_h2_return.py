"""Compare the blind reimplementation of Amendment 11 (H2) with results/h2_return, point values and replicates 0 to 49."""
import json, sys
pt = json.load(open("results/h2_return/point.json"))
reps = [json.loads(l) for l in open("results/h2_return/replicates.jsonl")]
d = json.load(open("verification/independent_h2/independent_results.json"))["datasets"]
sizes = {"snack": (9, 16, 25, 36), "reutskaja": (4, 9, 16)}
worst, n, skipped = 0.0, 0, []
for ds, ns in sizes.items():
    m = {"CL_log2_score": "bits|CL", "return_gain": "gain|return", "dwell_gain": "gain|dwell", "last_gain": "gain|last",
         "return_minus_dwell": "return_minus_dwell", "return_beta": "beta|return", "dwell_beta": "beta|dwell"}
    if ds == "snack":
        m["last_beta"] = "beta|last"  # Reutskaja last cue has no finite optimum, its beta is where each optimizer stops
    else:
        skipped.append("reutskaja last beta")
    for k in ns:
        m[f"first_return_chosen_n{k}"] = f"share_return_chosen|n={k}"
        m[f"no_return_n{k}"] = f"share_no_return|n={k}"
    for a, b in m.items():
        e = d[ds][a]
        key = f"{ds}|{b}"
        diffs = [abs(e["point"] - pt[key])] + [abs(x - r[key]) for x, r in zip(e["replicates"], reps)]
        scale = max(1.0, abs(pt[key]))
        worst = max(worst, max(diffs) / scale)
        n += 1
print(f"{n} measures, original sample and replicates 0 to {len(d['snack']['return_gain']['replicates']) - 1}, "
      f"max relative difference {worst:.1e}; not compared: {skipped}")
sys.exit(0 if worst < 1e-5 else 1)
