"""Independent point check of results/h1_within_decompose from the raw Aguiar files, without package code.
Lotteries are read from the screens in Figure 3 of Aguiar and colleagues (2023)."""
import itertools, json, sys
import numpy as np, pandas as pd
prizes = {0: {12: 1}, 1: {50: .5, 0: .5}, 2: {30: .5, 10: .5}, 3: {50: .25, 30: .25, 10: .25, 0: .25},
          4: {50: .25, 48: .2, 14: .15, 0: .4}, 5: {30: .25, 10: .25, 48: .2, 14: .15, 0: .15}}
s = json.load(open("results/h1_within_decompose/summary.json"))
r = json.load(open("results/h1_within/summary.json"))["crra_r_low_cost"]
u = lambda x: (x + 1) ** (1 - r) / (1 - r)
val = {"ev": {k: sum(x * p for x, p in d.items()) for k, d in prizes.items()},
       "crra": {k: sum(u(x) * p for x, p in d.items()) for k, d in prizes.items()}}
subsets = [()] + [c for k in range(1, 6) for c in itertools.combinations(range(1, 6), k)]
worst = 0.0
for c in ("low", "medium", "high"):
    df = pd.read_csv(f"data/aguiar/menu_choice_{c}.csv"); df = df[df.x1 > 1]
    menu = df.x1.map(lambda m: (0,) + subsets[m - 1])
    for tag, v in val.items():
        best = menu.map(lambda M: max(v[o] for o in M))
        g = pd.DataFrame({"n": menu.map(len), "d": df.x2 == 0, "w": (df.x2 != 0) & (df.x2.map(v) < best - 1e-9)})
        t = g.groupby("n").agg(N=("d", "size"), d=("d", "mean"), w=("w", "mean")); t["wl"] = t.w / (1 - t.d)
        x = np.log2(t.index.values)
        def sl(y, w):
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
        for k, y, w in (("default", t.d, t.N), ("wrong_lottery", t.w, t.N), ("wrong_given_lottery", t.wl, t.N * (1 - t.d))):
            worst = max(worst, abs(sl(y, w) - s[tag][f"{c}|growth|{k}"]["point"]))
print(f"18 growth values, max difference {worst:.1e}")
sys.exit(0 if worst < 1e-9 else 1)
