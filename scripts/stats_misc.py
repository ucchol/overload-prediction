"""Statistics layer for H1 within one study (Amendment 10) and decision time from structure (Amendment 12).

1. Lotteries. Logistic regression of a miss (expected value below the menu's best) on centered log2 n, cost and
   their interaction, standard errors clustered by menu (the file has no participant identifier). Odds ratios per
   doubling at each cost, and the ratio of the high and low cost odds ratios.
2. Time. Per dataset, a linear mixed model of log seconds on the four structure features (values on 0 to 1),
   random intercept and random slope of log2 n per participant, by REML. Standardized coefficients (per SD of the
   feature, in SDs of within person log time), percent change in time per doubling, and marginal and conditional
   R2 after Nakagawa and Schielzeth.
Usage, python3 -I scripts/stats_misc.py DEV_CSV OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from overload.experiments import h1_within as A  # noqa: E402
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from overload import time_models as tm  # noqa: E402

import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

dev, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
res = {}

# ------------------------------------------------------------------ lotteries
counts = A.load(Path("data/aguiar"))
rows = []
for c in A.COSTS:
    for m, cnt in counts[c].items():
        v = A.values(m); best = A.best_mask(v)
        for j, k in enumerate(cnt):
            rows += [{"miss": float(not best[j]), "log2n": np.log2(A.size(m)), "cost": c, "menu": m}] * int(k)
df = pd.DataFrame(rows)
df["x"] = df["log2n"] - df["log2n"].mean()
df["cost"] = pd.Categorical(df["cost"], categories=["low", "medium", "high"])
g = smf.glm("miss ~ x * cost", df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df["menu"]})
lot = {"coef": {k: float(v) for k, v in g.params.items()}, "se": {k: float(v) for k, v in g.bse.items()},
       "p": {k: float(v) for k, v in g.pvalues.items()}, "n_choices": int(len(df)), "menus": int(df["menu"].nunique())}
V = g.cov_params()
for c in ("low", "medium", "high"):
    names = ["x"] + ([f"x:cost[T.{c}]"] if c != "low" else [])
    est = sum(g.params[n] for n in names)
    se = float(np.sqrt(sum(V.loc[a, b] for a in names for b in names)))
    lot[f"or_per_doubling_{c}"] = [float(np.exp(est)), float(np.exp(est - 1.96 * se)), float(np.exp(est + 1.96 * se))]
lot["ratio_of_or_high_vs_low"] = [float(np.exp(g.params["x:cost[T.high]"])),
                                  float(np.exp(g.params["x:cost[T.high]"] - 1.96 * g.bse["x:cost[T.high]"])),
                                  float(np.exp(g.params["x:cost[T.high]"] + 1.96 * g.bse["x:cost[T.high]"])),
                                  float(g.pvalues["x:cost[T.high]"])]
res["lotteries"] = lot
print("lotteries", json.dumps({k: v for k, v in lot.items() if k.startswith(("or_", "ratio"))}), flush=True)

# ------------------------------------------------------------------ time
S = load_all(dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
tres = {}
for name, ts in S.items():
    T = tm.from_timeset(ts, SCALE[name])
    F = tm.structure(T)
    d = pd.DataFrame(F, columns=["log2n", "best", "gap", "mean_rest"])
    d["y"] = T.logt; d["pid"] = T.pid
    d = d.dropna()
    varying = [c for c in ["log2n", "best", "gap", "mean_rest"] if d[c].std() > 1e-9]
    # mean of the rest equals best minus gap with two options, keep the identifiable set
    if name == "krajbich2010":
        varying = [c for c in varying if c != "mean_rest"]
    form = "y ~ " + " + ".join(varying)
    re = "~log2n" if d["log2n"].nunique() > 1 else "~1"
    def best_fit(f, r):
        cands = []
        for meth in ("lbfgs", "bfgs", "powell", "nm"):
            try:
                mm = smf.mixedlm(f, d, groups=d["pid"], re_formula=r).fit(reml=True, method=meth, maxiter=5000)
                if np.isfinite(mm.llf) and np.all(np.isfinite(mm.bse_fe)):
                    cands.append(mm)
            except Exception:
                pass
        return max(cands, key=lambda mm: mm.llf) if cands else None
    m = best_fit(form, re)
    if m is None:
        re = "~1"; m = best_fit(form, re)
    fe = m.fe_params
    X = d[varying].to_numpy()
    pred_fixed = X @ fe[varying].to_numpy()
    var_f = float(np.var(pred_fixed))
    # random effect variance averaged over observations (intercept and slope)
    cov = np.asarray(m.cov_re)
    Zr = np.column_stack([np.ones(len(d))] + ([d["log2n"].to_numpy()] if cov.shape[0] > 1 else []))
    var_r = float(np.mean(np.einsum("ni,ij,nj->n", Zr, cov, Zr)))
    var_e = float(m.scale)
    ywithin = d["y"] - d.groupby("pid")["y"].transform("mean")
    rec = {"random": re, "n_trials": int(len(d)),
           "R2_marginal": var_f / (var_f + var_r + var_e), "R2_conditional": (var_f + var_r) / (var_f + var_r + var_e),
           "terms": {}}
    for c in varying:
        xw = d[c] - d.groupby("pid")[c].transform("mean")
        rec["terms"][c] = {"estimate": float(fe[c]), "se": float(m.bse_fe[c]), "z": float(m.tvalues[c]),
                           "p": float(m.pvalues[c]), "std_beta_within": float(fe[c] * xw.std() / ywithin.std())}
    if "log2n" in varying:
        rec["percent_time_per_doubling"] = [float(100 * (np.exp(fe["log2n"]) - 1)),
                                            float(100 * (np.exp(fe["log2n"] - 1.96 * m.bse_fe["log2n"]) - 1)),
                                            float(100 * (np.exp(fe["log2n"] + 1.96 * m.bse_fe["log2n"]) - 1))]
    if d["log2n"].nunique() > 1:
        m1 = best_fit("y ~ log2n", re)
        rec["percent_time_per_doubling_unadjusted"] = [float(100 * (np.exp(m1.fe_params["log2n"] + k * 1.96 * m1.bse_fe["log2n"]) - 1)) for k in (0, -1, 1)]
    tres[name] = rec
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "terms"}),
          {c: (round(t["estimate"], 3), round(t["p"], 4), round(t["std_beta_within"], 3)) for c, t in rec["terms"].items()},
          flush=True)
res["time_structure_lmm"] = tres
json.dump(res, open(out / "misc_stats.json", "w"), indent=1)
