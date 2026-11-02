"""Exploratory, after Amendment 10. Splits the misses of H1w into choices of the default (sure 12 tokens,
chosen when a participant skips the task) and choices of a lottery below the best, by cost and menu size.
Same data, measures and resampling as overload.experiments.h1_within."""
import json, sys
from pathlib import Path
import numpy as np
from overload.experiments.h1_within import COSTS, load, size, values, best_mask, slope, resample, options, crra_util

def parts(counts, util=None):
    out = {}
    for n in range(2, 7):
        ms = [m for m in counts if size(m) == n]
        N = sum(counts[m].sum() for m in ms)
        d = sum(counts[m][0] for m in ms)  # option 0, the default, is first in every menu
        miss_lot = 0.0
        for m in ms:
            v = values(m) if util is None else util[options(m)]
            b = best_mask(v); b0 = b.copy(); b0[0] = True  # default counted separately
            miss_lot += counts[m][~b0].sum()
        out[n] = dict(N=N, default=d / N, wrong_lottery=miss_lot / N, wrong_given_lottery=miss_lot / (N - d))
    return out

def growth_of(p, key, weight="N"):
    x = np.log2(list(p)); y = np.array([p[n][key] for n in p])
    w = np.array([p[n]["N"] if key != "wrong_given_lottery" else p[n]["N"] * (1 - p[n]["default"]) for n in p])
    return slope(x, y, w)

def measures(data, util=None):
    r = {}
    for c in COSTS:
        p = parts(data[c], util)
        for k in ("default", "wrong_lottery", "wrong_given_lottery"):
            r[f"{c}|growth|{k}"] = growth_of(p, k)
            for n in p:
                r[f"{c}|{k}|n={n}"] = p[n][k]
    for k in ("default", "wrong_lottery", "wrong_given_lottery"):
        r[f"high_minus_low|{k}"] = r[f"high|growth|{k}"] - r[f"low|growth|{k}"]
    return r

def summ(pt, reps):
    a = np.array(reps); lo, hi = np.percentile(a, [2.5, 97.5])
    basic = (2 * pt - hi, 2 * pt - lo)
    clear = bool((lo > 0 or hi < 0) and (basic[0] > 0 or basic[1] < 0))
    return dict(point=pt, percentile=[lo, hi], basic=list(basic), clear=clear)

if __name__ == "__main__":
    data = load(Path(sys.argv[1])); out = Path(sys.argv[2]); B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    r_crra = json.load(open("results/h1_within/summary.json"))["crra_r_low_cost"]
    res = {}
    for tag, util in (("ev", None), ("crra", crra_util(r_crra))):
        pt = measures(data, util)
        reps = [measures(resample(data, b), util) for b in range(B)]
        res[tag] = {k: summ(v, [x[k] for x in reps]) for k, v in pt.items()}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "summary.json", "w"), indent=1, sort_keys=True)
    f = lambda e: f"{e['point']:+.3f} [{e['percentile'][0]:+.3f}, {e['percentile'][1]:+.3f}]" + (" *" if e["clear"] else "")
    L = ["# H1w misses split into default and wrong lottery (exploratory, after Amendment 10)", ""]
    for tag in ("ev", "crra"):
        s = res[tag]
        L += [f"## Best option by {'expected value' if tag == 'ev' else f'CRRA utility, r = {r_crra:.3f}'}", "",
              "| Growth per doubling | Low cost | Medium cost | High cost | High minus low |", "|---|---|---|---|---|"]
        for k, lab in (("default", "Default chosen"), ("wrong_lottery", "Wrong lottery chosen"),
                       ("wrong_given_lottery", "Wrong lottery, among lottery choices")):
            L.append(f"| {lab} | " + " | ".join(f(s[f'{c}|growth|{k}']) for c in COSTS) + f" | {f(s[f'high_minus_low|{k}'])} |")
        L += ["", "| Share by menu size | " + " | ".join(f"{c} default, {c} wrong lottery" for c in COSTS) + " |", "|---" * 4 + "|"]
        for n in range(2, 7):
            L.append(f"| {n} | " + " | ".join(f"{100*s[f'{c}|default|n={n}']['point']:.0f}%, {100*s[f'{c}|wrong_lottery|n={n}']['point']:.0f}%" for c in COSTS) + " |")
        L.append("")
    L.append(f"Brackets give 95% percentile intervals from resampling choices within menu and cost (B = {B}, seeds 100000 + b). A star marks a value whose percentile and basic intervals both exclude zero.")
    (out / "table.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
