"""Part S of Amendment 13 with the benchmark time scale the amendment states (options inspected up to the final
selection), recomputed from the stored replicates of a run. The raw coefficients of every replicate are unchanged, so
S1 is each replicate's raw coefficient minus the new benchmark, summarized as before.

Usage, python3 -I scripts/rational_benchmark_s_text.py RUN_DIR OUT_JSON
RUN_DIR holds the slices of one run (slice_*.json). Nothing participant level is written.
"""
import json
import sys
from pathlib import Path

import numpy as np

from overload.bootstrap import summarize
from overload.cdm import load_cdm
from overload.experiments import rational_benchmark as rb

run, out = Path(sys.argv[1]), Path(sys.argv[2])
root = Path(__file__).resolve().parents[1] / "data"
panels, groups = rb.load_panels(root)
data = load_cdm(root / "cdm")
ones = {g: np.ones(P) for g, P in groups.items()}
stand = {g: rb.stopping(data, e, ones[g])[1] for e, g in ((0, "cdm_standard"), (1, "cdm_process"))}
bench = rb.stopping_benchmark(data, panels, stand, scale_at="final")
raw = {}
for f in sorted(run.glob("slice_*.json"), key=lambda q: int(q.stem.split("_")[1])):
    sl = json.loads(f.read_text())["boots"]
    for k, v in sl.items():
        if k.startswith("S/") and not k.endswith("_minus_benchmark"):
            raw.setdefault(k, []).extend(v)
res = {"benchmark": bench, "S1": {}}
for e, g in ((0, "cdm_standard"), (1, "cdm_process")):
    point, _ = rb.stopping(data, e, ones[g], stand[g])
    for k in ("value", "log_time", "log2_n"):
        b = bench[g][k]["mean"]
        res["S1"][f"{g}/{k}_minus_benchmark"] = summarize(point[k] - b, np.array(raw[f"S/{g}/{k}"]) - b)
out.write_text(json.dumps(res, indent=1))
for k, v in res["S1"].items():
    print(k.ljust(42), "%+.4f [%+.4f, %+.4f]" % (v["point"], *v["percentile"]), "clear" if v["clear"] else "")
