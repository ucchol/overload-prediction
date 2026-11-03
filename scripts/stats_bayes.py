"""Bayes factors for results that were not supported, and the smallest detectable effect of every test
(docs/statistics_plan.md).

BF01 compares no effect with a half normal prior in the predicted direction (Dienes 2014), given the estimate and
its bootstrap standard error (the SD of the stored replicates, or the 95 percent percentile width over 3.92 when
replicates were not stored). The detectable effect at 80 percent power, two sided 0.05, is 2.80 standard errors.
Usage, python3 -I scripts/stats_bayes.py OUT
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)


def reps(path):
    p = Path(path) / "replicates.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p)]


def se_of(key, summ, rp):
    if rp is not None and key in rp[0]:
        v = np.array([r[key] for r in rp], float)
        return float(np.nanstd(v, ddof=1)), "replicates"
    lo, hi = summ["percentile"]
    return float((hi - lo) / 3.919928), "interval"


def bf01(est, se, scale, sign=+1):
    """No effect against a half normal effect in direction sign with the given scale."""
    e = sign * est
    like0 = norm.pdf(e, 0, se)
    f = lambda d: norm.pdf(e, d, se) * 2 * norm.pdf(d, 0, scale)
    like1, _ = integrate.quad(f, 0, np.inf, limit=200)
    return float(like0 / like1)


def word(b):
    if b > 10: return "strong for no effect"
    if b > 3: return "moderate for no effect"
    if b < 1 / 10: return "strong for the effect"
    if b < 1 / 3: return "moderate for the effect"
    return "inconclusive"


rows = []


def add(test, est, se, src, hyp, scale, scale_source, sign=+1, supported=None):
    r = {"test": test, "estimate": est, "se": se, "se_source": src, "detectable_80": 2.8 * se,
         "hypothesized": hyp, "underpowered": bool(hyp is not None and 2.8 * se > abs(hyp)), "supported": supported}
    if scale is not None:
        b = bf01(est, se, scale, sign)
        r.update({"prior_scale": scale, "prior_source": scale_source, "BF01": b, "BF01_half_scale": bf01(est, se, scale / 2, sign),
                  "BF01_double_scale": bf01(est, se, scale * 2, sign),
                  "reading": word(b) if sign * est >= 0 or b < 3 else "strong against the hypothesis, estimate in the opposite direction" if b > 10 else "moderate against the hypothesis, estimate in the opposite direction"})
    rows.append(r)


# H1 within, lotteries
s = json.load(open("results/h1_within/summary.json")); rp = reps("results/h1_within")
se, src = se_of("H1w", s["H1w"], rp)
add("H1w lotteries, growth high minus low", s["H1w"]["point"], se, src, 0.036, 0.036, "plan simulation, Amendment 10", +1, False)

# H2
s = json.load(open("results/h2_return/summary.json")); rp = reps("results/h2_return")
for ds, sc in (("snack", 1.07), ("reutskaja", 0.38)):
    k = f"{ds}|return_minus_dwell"; se, src = se_of(k, s[k], rp)
    add(f"H2 {ds}, return minus dwell", s[k]["point"], se, src, sc, sc, "plan simulation, Amendment 11", +1, False)
    k = f"{ds}|gain|return"; se, src = se_of(k, s[k], rp)
    add(f"H2 {ds}, return gain over CL", s[k]["point"], se, src, None, None, None, +1, s[k]["clear"])

# H3 forecasting, O1 and O2
s = json.load(open("results/overload_premonition/summary.json"))["datasets"]
o2_reut = s["reutskaja"]["sr_auc_minus_near_best"]["point"]
for ds in ("reutskaja", "cdm"):
    e = s[ds]["observed_minus_comp"]; se, src = se_of("x", e, None)
    add(f"O1 {ds}, people minus compensation", e["point"], se, src, None, None, None, +1, e["clear"])
    e = s[ds]["sr_auc_minus_near_best"]; se, src = se_of("x", e, None)
    sup = bool(e["clear"] and e["point"] > 0)
    add(f"O2 {ds}, SR AUC minus near best", e["point"], se, src, o2_reut, None if sup else o2_reut,
        None if sup else "O2 effect in snacks in three seconds", +1, sup)

# time, Amendment 12
s = json.load(open("results/time_structure/summary.json")); rp = reps("results/time_structure")
sets = ("dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016")
sup = {t: {d: bool(s[f"{d}|{t}"]["clear"] and s[f"{d}|{t}"]["point"] > 0 and s[f"{d}|{t}"].get("p_holm", 1.0) < 0.05) for d in sets} for t in ("T1", "T2", "T3")}
dec = {}
smallest = {}
for t in ("T1", "T2"):
    vals = [s[f"{d}|{t}"]["point"] for d in sets if dec.get(f"{d}|{t}", sup[t][d])]
    smallest[t] = min(vals) if vals else None
for t in ("T1", "T2", "T3"):
    for d in sets:
        k = f"{d}|{t}"; e = s[k]; se, src = se_of(k, e, rp)
        is_sup = bool(dec.get(k, sup[t][d]))
        scale = smallest["T1"] if t in ("T1", "T3") else smallest["T2"]
        add(f"{t} {d}", e["point"], se, src, scale, None if is_sup else scale,
            None if is_sup else f"smallest supported {'T1' if t != 'T2' else 'T2'} gain", +1, is_sup)

json.dump({"rows": rows, "smallest_supported": smallest}, open(out / "bayes_sensitivity.json", "w"), indent=1)
for r in rows:
    print(f"{r['test']:<42} est {r['estimate']:+.4f} se {r['se']:.4f} MDE80 {r['detectable_80']:.4f}"
          + (f"  BF01 {r['BF01']:.3g} ({r['reading']}) [{r['BF01_half_scale']:.3g}, {r['BF01_double_scale']:.3g}]" if 'BF01' in r else "")
          + f"  supported {r['supported']}")
