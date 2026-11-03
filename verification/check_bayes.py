"""Separate check of scripts/stats_bayes.py. BF01 for a half normal prior recomputed in closed form,
p(e | H1) = 2 N(e; 0, sqrt(se^2 + s^2)) Phi(e s / (se sqrt(se^2 + s^2))), against the stored numerical values."""
import json, sys
import numpy as np
from scipy.stats import norm
d = json.load(open(sys.argv[1]))["rows"]
worst = 0.0
for r in d:
    if "BF01" not in r:
        continue
    sign = 1.0
    for key, s in (("BF01", r["prior_scale"]), ("BF01_half_scale", r["prior_scale"] / 2), ("BF01_double_scale", r["prior_scale"] * 2)):
        e, se = sign * r["estimate"], r["se"]
        tot = np.sqrt(se ** 2 + s ** 2)
        h1 = 2 * norm.pdf(e, 0, tot) * norm.cdf(e * s / (se * tot))
        bf = norm.pdf(e, 0, se) / h1
        worst = max(worst, abs(np.log(bf) - np.log(r[key])))
print("largest difference in log BF01", worst)
