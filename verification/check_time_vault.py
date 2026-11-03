"""Separate recomputation of the key vault numbers of the decision time models, written without the package's
r2, center or the evaluation script. For every set and model, the within person R2 from pandas group means, and the
bootstrap percentile interval of the R2 difference from S2 for the key models, by resampling people explicitly
(rows copied), same seeds.
Compared with results of scripts/time_models_vault_eval.py.
Usage, python3 -I verification/check_time_vault.py VAULT_DIR EVAL_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

vdir, ev = Path(sys.argv[1]), json.load(open(sys.argv[2]))["sets"]
KEY = ["S1", "TLS_looks", "TLS_tfit", "S2+TLS", "S2+TLS_tfit", "Wald_n", "Wald_n_cal", "LBA_n_cal", "S2+Wald_n", "Ceiling2"]
worst_r2, worst_ci = 0.0, 0.0
for s, rec in ev.items():
    z = np.load(vdir / f"{s}_pred.npz")
    models = [k for k in z.files if k not in ("logt", "pid", "n", "choice")]
    df = pd.DataFrame({"pid": z["pid"], "y": z["logt"], **{m: z[m] for m in models}})
    cols = ["y"] + models
    dm = df[cols] - df.groupby("pid")[cols].transform("mean")
    dm["pid"] = df["pid"]
    for m in models:
        r = 1 - ((dm["y"] - dm[m]) ** 2).sum() / (dm["y"] ** 2).sum()
        worst_r2 = max(worst_r2, abs(r - rec["models"][m]["R2"]))
    people = np.unique(df["pid"])
    rows = {p: np.flatnonzero(df["pid"].to_numpy() == p) for p in people}
    Y = dm["y"].to_numpy()
    for m in [m for m in KEY if m in models and m != "S2"]:
        e2, em = (Y - dm["S2"].to_numpy()) ** 2, (Y - dm[m].to_numpy()) ** 2
        d = []
        for b in range(1000):
            idx = np.random.default_rng(130500 + b).integers(0, people.size, people.size)
            r = np.concatenate([rows[people[i]] for i in idx])  # rows copied for each draw of a person
            d.append((e2[r].sum() - em[r].sum()) / (Y[r] ** 2).sum())
        pc = np.percentile(d, [2.5, 97.5])
        worst_ci = max(worst_ci, float(np.max(np.abs(pc - rec["models"][m]["minus_S2"]["percentile"]))))
    print(s, "checked", len(models), "models", flush=True)
print("largest R2 difference", worst_r2, "largest interval end difference", worst_ci)
