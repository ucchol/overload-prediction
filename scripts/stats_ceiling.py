"""Statistics layer for the machine learning ceiling. Mean per trial held out log2 score difference (model minus
the network ceiling) with participant clustered standard errors (the random intercept variance of a mixed model
sits at its boundary here, so the mixed model is not used), per dataset (the four sums
panels pooled, people identified within experiment). Usage, python3 -I scripts/stats_ceiling.py OUT"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
Z = np.load("results/ml_ceiling/trial_scores.npz")
sets = {"snack": ["snack"], "reutskaja": ["reutskaja"], "cdm": ["cdm_standard_3", "cdm_standard_7", "cdm_process_3", "cdm_process_7"]}
rows = []
for ds, names in sets.items():
    for m in ("CL", "SR", "GBM"):
        d = np.concatenate([Z[f"{n}|{m}"] - Z[f"{n}|NN"] for n in names])
        pid = np.concatenate([np.array([f"{n.split('_')[1] if n.startswith('cdm') else n}_{p}" for p in Z[f"{n}|participant"]]) for n in names])
        df = pd.DataFrame({"d": d, "pid": pid})
        f = smf.ols("d ~ 1", df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df["pid"])[0]})
        pm = df.groupby("pid")["d"].mean()
        rows.append({"dataset": ds, "contrast": f"{m} minus network", "mean_bits": float(d.mean()),
                     "estimate": float(f.params["Intercept"]), "clustered_se": float(f.bse["Intercept"]),
                     "p": float(f.pvalues["Intercept"]), "cohen_dz": float(pm.mean() / pm.std(ddof=1)),
                     "share_people_below_ceiling": float(np.mean(pm < 0))})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()})
json.dump(rows, open(out / "ceiling_lmm.json", "w"), indent=1)
