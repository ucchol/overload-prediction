"""Exploratory model and parameter recovery on the real displays of the overload datasets.

For each dataset (snack, Reutskaja, the four panels of Caplin, Dean and Martin) and each generating model (CL, sample
until return, DbS_avg, MLBA_avg), choices are simulated on the real displays from the model's sensitivity fitted to
the real choices, scaled by a random factor exp(U(-0.7, 0.7)) so that recovery is checked over a range. Every model is
then fitted to the simulated choices (one parameter each) and the one with the highest likelihood is recorded. CL and
sample until return choose almost alike (Amendment 5, design check), so their mutual confusion is expected and says
that choices alone cannot separate the two. Seeds numpy.random.default_rng(150000 + 1000 * g + r).
Usage, python3 -I scripts/model_recovery.py OUT [R]
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from overload.cdm import Displays, levels, softmax_masked, sr_levels, static_scores  # noqa: E402
from overload.experiments.overload_premonition import groups  # noqa: E402
from overload.overload_forecast import miss_forecasts  # noqa: E402

MODELS = ("CL", "SR", "DbS", "MLBA")


def probabilities(panel, model, lam):
    v = panel.values
    if model == "SR":
        lv = levels(v + panel.shift, np.zeros(v.shape[0], int))
        disp = Displays.of(lv)
        logp_u, _ = sr_levels(lam, disp.unique)
        lvl = np.where(np.isfinite(v), np.rint(v + panel.shift) - 1, 0).astype(int)
        p = np.where(np.isfinite(v), np.exp(logp_u[disp.index][np.arange(v.shape[0])[:, None], lvl]), 0.0)
    else:
        S = {"CL": v, "DbS": static_scores(v, "DbS_avg"), "MLBA": static_scores(v, "MLBA_avg", mlba_scale=panel.mlba_scale)}[model]
        p = np.nan_to_num(softmax_masked(lam * S))
    return p / p.sum(axis=1, keepdims=True)


def fit_all(panel):
    ll = {}
    _, lam = miss_forecasts(panel, np.ones(panel.n.size), MODELS, fitted=True, loglik=ll)
    return {m: float(lam[m][0]) for m in MODELS}, {m: float(ll[m].sum()) for m in MODELS}


if __name__ == "__main__":
    np.seterr(all="ignore")
    out = Path(sys.argv[1]); R = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
    datasets = {"snack": G["snack"][0], "reutskaja": G["reutskaja"][0], "cdm": G["cdm_standard"][0] + G["cdm_process"][0]}
    true_fit = {ds: [fit_all(p)[0] for p in panels] for ds, panels in datasets.items()}
    res = {}
    for ds, panels in datasets.items():
        conf = {g: {m: 0 for m in MODELS} for g in MODELS}
        prec = {g: {"true": [], "recovered": []} for g in MODELS}
        for gi, g in enumerate(MODELS):
            for r in range(R):
                rng = np.random.default_rng(150000 + 1000 * gi + r + 100000 * list(datasets).index(ds))
                factor = float(np.exp(rng.uniform(-0.7, 0.7)))
                tot = {m: 0.0 for m in MODELS}
                rec_lam = []
                for k, p in enumerate(panels):
                    lam = true_fit[ds][k][g] * factor
                    P = probabilities(p, g, lam)
                    u = rng.random(P.shape[0])[:, None]
                    ch = (np.cumsum(P, axis=1) < u).sum(axis=1)
                    sim = dataclasses.replace(p, choice=np.minimum(ch, p.n - 1))
                    lam_hat, ll = fit_all(sim)
                    for m in MODELS:
                        tot[m] += ll[m]
                    rec_lam.append((lam, lam_hat[g]))
                conf[g][max(tot, key=tot.get)] += 1
                prec[g]["true"].append(float(np.mean([a for a, _ in rec_lam])))
                prec[g]["recovered"].append(float(np.mean([b for _, b in rec_lam])))
        res[ds] = {"confusion_percent": {g: {m: 100.0 * c / R for m, c in row.items()} for g, row in conf.items()},
                   "parameter": {g: {"rank_correlation": float(np.corrcoef(np.argsort(np.argsort(d["true"])), np.argsort(np.argsort(d["recovered"])))[0, 1]),
                                     "median_ratio": float(np.median(np.array(d["recovered"]) / np.array(d["true"])))}
                                 for g, d in prec.items()}}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "summary.json", "w"), indent=1, sort_keys=True)
    names = {"snack": "Snacks at a glance", "reutskaja": "Snacks in three seconds", "cdm": "Sums worked out"}
    L = ["# Model and parameter recovery on the real displays (exploratory)", "",
         f"{R} simulated datasets per generating model and dataset. Rows, generating model. Columns, share of datasets in which each model fitted best (percent).", ""]
    for ds in datasets:
        L += [f"## {names[ds]}", "", "| Generating model | " + " | ".join(MODELS) + " | Rank correlation of sensitivity | Median recovered / true |",
              "|---|" + "---|" * (len(MODELS) + 2)]
        for g in MODELS:
            c = res[ds]["confusion_percent"][g]; pr = res[ds]["parameter"][g]
            L.append(f"| {g} | " + " | ".join(f"{c[m]:.0f}" for m in MODELS) + f" | {pr['rank_correlation']:.2f} | {pr['median_ratio']:.2f} |")
        L.append("")
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
