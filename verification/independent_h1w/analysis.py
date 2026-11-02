"""Run the independent H1w analysis (Amendment 10) and write independent_results.json.

Usage: python -I analysis.py --data DATA_DIR [--B 1000] [--out PATH]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--B", type=int, default=1000)
ap.add_argument("--out", default=os.path.join(HERE, "independent_results.json"))
args = ap.parse_args()

counts = core.load_counts(args.data)
t0 = time.time()
r_hat, lam_crra, _ = core.crra_fit(counts["low"])
srtabs = core.SRTable(core.EV)
point = core.analyse(counts, r_hat, srtabs)

# Scalar quantities to resample, as name -> function of an analyse() result.
names = {"H1w": lambda o: o["H1w"], "H1w_medium_minus_low": lambda o: o["H1w_medium_minus_low"],
         "H1w_crra": lambda o: o["H1w_crra"]}
per_cost = ["growth_observed", "growth_CL", "growth_SR", "growth_observed_crra", "people_minus_CL",
            "people_minus_SR", "bits_CL", "bits_SR", "bits_above_uniform_CL", "bits_above_uniform_SR"]
for c in core.COSTS:
    for k in per_cost:
        names[f"{c}.{k}"] = (lambda c, k: lambda o: o[c][k])(c, k)
    for j, n in enumerate(core.SIZES):
        names[f"{c}.miss_rate.n{n}"] = (lambda c, j: lambda o: o[c]["rate"][j])(c, j)
        names[f"{c}.miss_rate_CL.n{n}"] = (lambda c, j: lambda o: o[c]["rate_CL"][j])(c, j)
        names[f"{c}.miss_rate_SR.n{n}"] = (lambda c, j: lambda o: o[c]["rate_SR"][j])(c, j)
        names[f"{c}.miss_rate_crra.n{n}"] = (lambda c, j: lambda o: o[c]["rate_crra"][j])(c, j)

reps = {k: [] for k in names}
reps["crra_r_refit"] = []
for b in range(args.B):
    cb = core.resample(counts, b)
    ob = core.analyse(cb, r_hat, srtabs)  # r held at the original fit
    for k, f in names.items():
        reps[k].append(f(ob))
    reps["crra_r_refit"].append(core.crra_fit(cb["low"])[0])  # descriptive only
    if (b + 1) % 100 == 0:
        print(f"replicate {b + 1}/{args.B}, {time.time() - t0:.0f} s", flush=True)

res = {"B": args.B, "seed_rule": "numpy.random.default_rng(100000 + b)", "intervals": "95% percentile and basic; clear when both exclude zero"}
for k, f in names.items():
    res[k] = core.summary(f(point), reps[k])
res["crra_r"] = core.summary(r_hat, reps["crra_r_refit"])
res["crra_r"]["note"] = "point = ML r on low cost choices; interval from refitting r per replicate (descriptive); H1w_crra holds r at the point value"
res["crra_lambda"] = lam_crra
res["crra_utility_ranking_options_best_first"] = [int(i) for i in np.argsort(-core.crra_values(r_hat))]
res["fitted_sensitivities_original"] = {c: {"CL": point[c]["lambda_CL"], "SR": point[c]["lambda_SR"]} for c in core.COSTS}
res["choices_per_size"] = {c: point[c]["n_choices"].astype(int).tolist() for c in core.COSTS}
res["share_of_choices_per_size_percent"] = {c: (100 * point[c]["n_choices"] / point[c]["n_choices"].sum()).round(2).tolist() for c in core.COSTS}
with open(args.out, "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", args.out)
for k in ["H1w", "H1w_medium_minus_low", "H1w_crra"]:
    print(k, res[k])
print("crra r", res["crra_r"])
for c in core.COSTS:
    print(c, {k: round(res[f"{c}.{k}"]["point"], 4) for k in per_cost},
          [round(res[f"{c}.miss_rate.n{n}"]["point"], 4) for n in core.SIZES])
