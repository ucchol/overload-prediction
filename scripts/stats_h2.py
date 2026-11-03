"""Statistics layer for H2, the return rule (docs/statistics_plan.md).

Per trial held out log2 probability of the chosen item under CL, CL plus the return cue and CL plus the dwell cue,
each set size predicted from fits to the others (Amendment 11). Then
1. a linear mixed model of the per trial difference with a random intercept per participant (REML),
2. per participant mean differences, Cohen's dz, a one sample t test and the share of participants above zero.
The mean differences equal the gains of Amendment 11 (checked against results/h2_return).
Usage, python3 -I scripts/stats_h2.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from overload.experiments import h2_return as H  # noqa: E402

import statsmodels.formula.api as smf  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
data = {"snack": H.load_snack_gaze(Path("data/snack"), Path("data/snack_gaze"))[0],
        "reutskaja": H.load_reutskaja_gaze(Path("data/reutskaja"))[0]}


def per_trial(blocks, cue):
    w = [np.ones(b.choice.size) for b in blocks]
    ll, person = [], []
    for k, b in enumerate(blocks):
        others = [x for j, x in enumerate(blocks) if j != k]
        th = H.fit(others, [w[j] for j in range(len(blocks)) if j != k], cue)
        lam, beta = th
        Z = b.cue[cue] if cue else np.zeros_like(b.V)
        U = lam * b.V + beta * Z
        lse = np.log(np.exp(U - U.max(1, keepdims=True)).sum(1)) + U.max(1)
        ll.append((U[np.arange(b.choice.size), b.choice] - lse) / np.log(2)); person.append(b.person)
    return np.concatenate(ll), np.concatenate(person)


rows = []
stored = json.load(open("results/h2_return/point.json")) if Path("results/h2_return/point.json").exists() else {}
for ds, blocks in data.items():
    llc, person = per_trial(blocks, None)
    llr, _ = per_trial(blocks, "return")
    lld, _ = per_trial(blocks, "dwell")
    for label, d in (("return minus CL", llr - llc), ("return minus dwell", llr - lld), ("dwell minus CL", lld - llc)):
        df = pd.DataFrame({"d": d, "person": person})
        m = smf.mixedlm("d ~ 1", df, groups=df["person"]).fit(reml=True)
        pm = df.groupby("person")["d"].mean().to_numpy()
        t = stats.ttest_1samp(pm, 0.0)
        rows.append({"dataset": ds, "contrast": label, "mean_bits_per_trial": float(d.mean()),
                     "lmm_estimate": float(m.params["Intercept"]), "lmm_se": float(m.bse["Intercept"]),
                     "lmm_z": float(m.tvalues["Intercept"]), "lmm_p": float(m.pvalues["Intercept"]),
                     "person_mean": float(pm.mean()), "cohen_dz": float(pm.mean() / pm.std(ddof=1)),
                     "t": float(t.statistic), "df": int(pm.size - 1), "p_t": float(t.pvalue),
                     "share_above_zero": float(np.mean(pm > 0))})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
json.dump({"rows": rows, "stored_point": stored}, open(out / "h2_lmm.json", "w"), indent=1)
