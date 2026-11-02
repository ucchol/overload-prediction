"""Compare the blind reimplementation of Amendment 10 (H1w) with results/h1_within."""
import json, sys
s = json.load(open("results/h1_within/summary.json"))
d = json.load(open("verification/independent_h1w/independent_results.json"))
pairs = [(k, k) for k in ("H1w", "H1w_medium_minus_low", "H1w_crra")]
names = [("growth|observed", "growth_observed"), ("growth|CL", "growth_CL"), ("growth|SR", "growth_SR"),
         ("growth|observed_crra", "growth_observed_crra"), ("observed_minus|CL", "people_minus_CL"),
         ("observed_minus|SR", "people_minus_SR"), ("bits|CL", "bits_CL"), ("bits|SR", "bits_SR")]
names += [(f"miss_rate|n={n}", f"miss_rate.n{n}") for n in range(2, 7)]
pairs += [(f"{c}|{a}", f"{c}.{b}") for c in ("low", "medium", "high") for a, b in names]
v = lambda x: x["point"] if isinstance(x, dict) else x
dp = di = 0.0
bad = []
for a, b in pairs:
    x, y = s[a], d[b]
    dp = max(dp, abs(v(x) - v(y)))
    if isinstance(x, dict) and isinstance(y, dict) and "percentile" in x and "percentile" in y:
        di = max(di, *(abs(p - q) for p, q in zip(x["percentile"], y["percentile"])))
        if x["clear"] != y["clear"]:
            bad.append(a)
print(f"{len(pairs)} values, max point difference {dp:.1e}, max interval difference {di:.1e}, clear calls differing {bad}")
sys.exit(0 if dp < 1e-6 and di < 1e-6 and not bad else 1)
