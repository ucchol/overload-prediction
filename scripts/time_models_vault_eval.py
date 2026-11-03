"""Vault statistics for the decision time models (docs/time_models_plan.md, docs/statistics_plan.md).

Reads the vault predictions written by scripts/time_models_run.py --stage vault, and for every set and model gives
the held out R2 with a participant bootstrap interval, the difference from S1 and S2 (percentile and basic
intervals), a linear mixed model of per trial squared error differences against S2 with Holm's correction across
sets, Cohen's dz, BF01 where a model is not clearly above S2, the share of the ceiling, and the satisfactory verdict.
Usage, python3 -I scripts/time_models_vault_eval.py VAULT_DIR DEV_R1_DIR OUT
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm

sys.path.insert(0, "src")
from overload import time_models as tm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

warnings.filterwarnings("ignore")
SETS = ["dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016"]
B = 1000
vdir, r1dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)


def bf01(est, se, scale):
    like0 = norm.pdf(est, 0, se)
    like1, _ = integrate.quad(lambda d: norm.pdf(est, d, se) * 2 * norm.pdf(d, 0, scale), 0, 50 * (scale + se),
                              limit=400, epsabs=0, epsrel=1e-11)
    return float(like0 / like1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = p.size
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


res = {}
lmm_p = {}
for s in SETS:
    z = dict(np.load(vdir / f"{s}_pred.npz"))
    y, pid = z.pop("logt"), z.pop("pid"); z.pop("n"); z.pop("choice")
    _, pid = np.unique(pid, return_inverse=True); P = pid.max() + 1
    models = sorted(z)
    yc = tm.center(y, pid)
    C = {m: tm.center(z[m], pid) for m in models}
    # per person sums, so a bootstrap draw is a weighted sum over people
    tot = np.bincount(pid, weights=yc ** 2, minlength=P)
    sse = {m: np.bincount(pid, weights=(yc - C[m]) ** 2, minlength=P) for m in models}
    point = {m: 1 - sse[m].sum() / tot.sum() for m in models}
    W = np.zeros((B, P))
    for b in range(B):
        idx = np.random.default_rng(130500 + b).integers(0, P, P)
        W[b] = np.bincount(idx, minlength=P)
    boot = {m: 1 - (W @ sse[m]) / (W @ tot) for m in models}
    dev1 = json.load(open(r1dir / f"{s}_summary.json"))["r2"]
    scale = max(dev1["S2"] - dev1["S1"], 0.01)
    ceil = max(point.get("Ceiling", -1), point.get("Ceiling2", -1))
    rows = {}
    for m in models:
        r = {"R2": point[m], "R2_ci": np.percentile(boot[m], [2.5, 97.5]).tolist()}
        for ref in ("S1", "S2"):
            d = boot[m] - boot[ref]; pt = point[m] - point[ref]
            pc = np.percentile(d, [2.5, 97.5]); bc = [2 * pt - pc[1], 2 * pt - pc[0]]
            clear_up = bool(pc[0] > 0 and bc[0] > 0); clear_down = bool(pc[1] < 0 and bc[1] < 0)
            r[f"minus_{ref}"] = {"point": pt, "percentile": pc.tolist(), "basic": bc, "se": float(d.std(ddof=1)),
                                 "clear_above": clear_up, "clear_below": clear_down}
        if m != "S2":
            dt = (yc - C["S2"]) ** 2 - (yc - C[m]) ** 2
            df = pd.DataFrame({"d": dt, "pid": pid})
            try:
                f = smf.mixedlm("d ~ 1", df, groups=df["pid"]).fit(reml=True)
                r["lmm"] = {"estimate": float(f.params["Intercept"]), "se": float(f.bse["Intercept"]),
                            "z": float(f.tvalues["Intercept"]), "p": float(f.pvalues["Intercept"])}
            except Exception as e:  # recorded, not hidden
                r["lmm"] = {"error": str(e)}
            pm = np.bincount(pid, weights=dt, minlength=P) / np.bincount(pid, minlength=P)
            r["cohen_dz_vs_S2"] = float(pm.mean() / pm.std(ddof=1)) if pm.std() > 0 else None
            r["share_people_better_than_S2"] = float(np.mean(pm > 0))
            lmm_p.setdefault(m, {})[s] = r["lmm"].get("p", np.nan)
            if not r["minus_S2"]["clear_above"]:
                e, se = r["minus_S2"]["point"], r["minus_S2"]["se"]
                r["BF01_vs_S2"] = {"scale": scale, "BF01": bf01(e, se, scale), "half": bf01(e, se, scale / 2),
                                   "double": bf01(e, se, scale * 2)}
        r["share_of_ceiling"] = point[m] / ceil if ceil > 0 else None
        rows[m] = r
    res[s] = {"prior_scale": scale, "ceiling_R2": ceil,
              "S2_clear_above_S1": rows["S2"]["minus_S1"]["clear_above"], "models": rows}

models = sorted(set.intersection(*[set(res[s]["models"]) for s in SETS]))
for m in lmm_p:
    sets = [s for s in SETS if s in lmm_p[m]]
    adj = holm([lmm_p[m][s] if np.isfinite(lmm_p[m][s]) else 1.0 for s in sets])
    for s, a in zip(sets, adj):
        res[s]["models"][m]["lmm"]["p_holm"] = float(a)
verdict = {}
for m in models:
    c1 = all(res[s]["models"][m]["minus_S1"]["clear_above"] for s in SETS if res[s]["S2_clear_above_S1"])
    c2 = not any(res[s]["models"][m]["minus_S2"]["clear_below"] for s in SETS)
    verdict[m] = {"above_S1_where_S2_is": c1, "not_below_S2": c2, "satisfactory": bool(c1 and c2)}
json.dump({"sets": res, "verdict": verdict}, open(out / "vault_eval.json", "w"), indent=1, default=float)

lines = ["| Model | " + " | ".join(SETS) + " | Satisfactory |", "|---" * (len(SETS) + 2) + "|"]
for m in models:
    cells = []
    for s in SETS:
        r = res[s]["models"][m]; t = f"{r['R2']:+.3f}"
        if m != "S2":
            t += " ↑" if r["minus_S2"]["clear_above"] else (" ↓" if r["minus_S2"]["clear_below"] else "")
        cells.append(t)
    lines.append(f"| {m} | " + " | ".join(cells) + f" | {'yes' if verdict[m]['satisfactory'] else 'no'} |")
lines.append("")
lines.append("S2 clearly above S1 in: " + ", ".join(s for s in SETS if res[s]["S2_clear_above_S1"]))
open(out / "vault_table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
