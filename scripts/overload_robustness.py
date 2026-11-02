"""Exploratory robustness of the overload measure (after Amendment 8), people and held out CL.

Variants of the outcome, each with its growth per doubling of the set (and per 10 more options for the linear
version), for people and for CL fitted to the other set sizes, and people minus CL.
  best       a miss is a choice below the best value (the measure of Amendment 8)
  near_best  a miss is a choice more than one unit below the best (one rating point or one dollar)
  value_lost the value lost, best minus chosen, as a share of the dataset's value range
  linear     the best miss, slope on n per 10 options instead of log2 n
Datasets as in Amendment 8 (snack, Reutskaja, the four panels of Caplin, Dean and Martin averaged). Participant
bootstrap, numpy.random.default_rng(130000 + b), groups in the order of overload_premonition.groups.
Usage, python3 -I scripts/overload_robustness.py OUT [B]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from overload.bootstrap import summarize  # noqa: E402
from overload.cdm import fit_softmax, softmax_masked  # noqa: E402
from overload.experiments.overload_premonition import groups  # noqa: E402

RANGE = {"snack": 6.0, "reutskaja": 10.0, "cdm": 34.0}
VARIANTS = ("best", "near_best", "value_lost", "linear")


def per_trial(panel, q):
    """People's outcome and CL's expected outcome per trial, for every variant."""
    v = panel.values
    top = np.nanmax(v, axis=1)
    chosen = v[np.arange(v.shape[0]), panel.choice]
    rng_ = RANGE["cdm" if panel.name.startswith("cdm") else panel.name]
    vv = np.where(np.isfinite(v), v, -np.inf)
    out = {}
    out["best"] = ((chosen < top).astype(float), 1.0 - np.where(vv == top[:, None], q, 0).sum(axis=1))
    out["near_best"] = ((chosen < top - 1).astype(float), np.where(vv < top[:, None] - 1, q, 0).sum(axis=1))
    lost_q = np.where(np.isfinite(v), (top[:, None] - np.nan_to_num(v)) * q, 0).sum(axis=1)
    out["value_lost"] = ((top - chosen) / rng_, lost_q / rng_)
    out["linear"] = out["best"]
    return out


def growth(x, y, w):
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def panel_growth(panel, w):
    q = np.zeros_like(panel.values)
    for s in np.unique(panel.n):
        test, train = panel.n == s, panel.n != s
        lam = fit_softmax(panel.values[train], panel.choice[train], w[train])
        q[test] = np.nan_to_num(softmax_masked(lam * panel.values[test]))
    res = {}
    for var, (y, f) in per_trial(panel, q).items():
        sizes = np.unique(panel.n)
        ry = np.array([w[panel.n == s] @ y[panel.n == s] / w[panel.n == s].sum() for s in sizes])
        rf = np.array([w[panel.n == s] @ f[panel.n == s] / w[panel.n == s].sum() for s in sizes])
        x = sizes / 10.0 if var == "linear" else np.log2(sizes)
        one = np.ones(len(sizes))  # Amendment 8 averages over set sizes equally
        res[var] = (growth(x, ry, one), growth(x, rf, one))
    return res


def measures(G, b):
    rng = None if b < 0 else np.random.default_rng(130000 + b)
    per_ds = {}
    for g, (panels, P) in G.items():
        mult = np.ones(P) if rng is None else np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
        ds = "cdm" if g.startswith("cdm") else g
        for p in panels:
            per_ds.setdefault(ds, []).append(panel_growth(p, mult[p.participant]))
    out = {}
    for ds, R in per_ds.items():
        for var in VARIANTS:
            people = float(np.mean([r[var][0] for r in R]))
            cl = float(np.mean([r[var][1] for r in R]))
            out[f"{ds}|{var}|people"] = people
            out[f"{ds}|{var}|CL"] = cl
            out[f"{ds}|{var}|people_minus_CL"] = people - cl
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    out = Path(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
    point = measures(G, -1)
    reps = [measures(G, b) for b in range(B)]
    s = {k: summarize(v, [r[k] for r in reps]) for k, v in point.items()}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(out / "summary.json", "w"), indent=1, sort_keys=True)
    f = lambda e: f"{e['point']:+.3f} [{e['percentile'][0]:+.3f}, {e['percentile'][1]:+.3f}]" + (" *" if e["clear"] else "")
    names = {"snack": "Snacks at a glance", "reutskaja": "Snacks in three seconds", "cdm": "Sums worked out"}
    lab = {"best": "Missed the best (main measure)", "near_best": "Missed every near best option",
           "value_lost": "Value lost, share of the scale", "linear": "Missed the best, per 10 more options"}
    L = ["# Robustness of the overload measure (exploratory)", "",
         "Growth per doubling of the set (per 10 options for the linear row). People minus CL held out, star when clear.", "",
         "| Measure | " + " | ".join(names[d] for d in ("snack", "reutskaja", "cdm")) + " |", "|---|---|---|---|"]
    for var in VARIANTS:
        L.append(f"| {lab[var]}, people | " + " | ".join(f(s[f'{d}|{var}|people']) for d in ("snack", "reutskaja", "cdm")) + " |")
        L.append(f"| {lab[var]}, people minus CL | " + " | ".join(f(s[f'{d}|{var}|people_minus_CL']) for d in ("snack", "reutskaja", "cdm")) + " |")
    L += ["", f"Participant bootstrap, B = {B}, seeds 130000 + b."]
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
