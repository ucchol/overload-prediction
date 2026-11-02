"""H1 across datasets in one figure, and the equivalence check for snacks at a glance (descriptive).

Reads stored results only. Writes results/h1_synthesis/{fig_h1_synthesis.png,.pdf,summary.json,table.md}.

Equivalence bound. Scheibehenne, Greifeneder and Todd (2010) treat a standardized difference of d = 0.2 between
small and large sets as the level that indicates choice overload. For a miss (a binary outcome) with overall rate p,
d = 0.2 is a difference of 0.2 * sqrt(p (1 - p)) in the miss rate. Spread over the two doublings from 9 to 36 snacks,
the bound is that difference divided by 2 per doubling. The two one sided test (Lakens 2017) checks whether the
90 percent percentile interval of the growth lies inside the bound.
Usage, python3 -I scripts/h1_synthesis.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
OUT = R / "h1_synthesis"
OUT.mkdir(parents=True, exist_ok=True)


def entry(folder, *path):
    e = json.load(open(R / folder / "summary.json"))
    for k in path:
        e = e[k]
    return e


across = [("Snacks at a glance", "overload_snack_all", "snack"), ("Snacks in three seconds", "overload_premonition", "reutskaja"),
          ("Sums worked out", "overload_premonition", "cdm")]
rows = []
for label, folder, ds in across:
    g = {m: entry(folder, "datasets", ds, "growth", m) for m in ("observed", "CL", "SR")}
    rows.append((label, g))
within = [(f"Lotteries, {c} cost", entry("h1_within", f"{c}|growth|observed"), entry("h1_within", f"{c}|growth|CL"),
           entry("h1_within", f"{c}|growth|SR")) for c in ("low", "medium", "high")]

# ---------------------------------------------------------------- equivalence for snacks at a glance
reps = [json.loads(l)["datasets"]["snack"]["growth"]["observed"] for l in open(R / "overload_snack_all" / "replicates.jsonl")]
p_rates = json.load(open(R / "overload_snack_all" / "point.json"))["panels"]["snack"]["rate"]["observed"]
point = rows[0][1]["observed"]["point"]
lo90, hi90 = np.percentile(reps, [5, 95])
basic90 = (2 * point - hi90, 2 * point - lo90)
p_bar = float(np.mean(p_rates))  # mean over the four set sizes
bound = 0.2 * np.sqrt(p_bar * (1 - p_bar)) / 2.0
equivalent = bool(lo90 > -bound and hi90 < bound and basic90[0] > -bound and basic90[1] < bound)
eq = {"growth": point, "interval90_percentile": [float(lo90), float(hi90)], "interval90_basic": list(map(float, basic90)),
      "mean_miss_rate": p_bar, "bound": float(bound), "equivalent": equivalent,
      "note": "descriptive, the interval was known before the bound was set"}

# ---------------------------------------------------------------- figure
INK, SR_C, CL_C = "#222222", "#1f5fbf", "#d9822b"
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, gridspec_kw={"width_ratios": [3, 3]})
for ax, title, items in ((axes[0], "Across tasks, cost of judging rises left to right", [(l, g["observed"], g["CL"], g["SR"]) for l, g in rows]),
                         (axes[1], "Within one lottery study", within)):
    x = np.arange(len(items))
    for i, (label, obs, cl, sr) in enumerate(items):
        lo, hi = obs["percentile"]
        ax.plot([i, i], [lo, hi], color=INK, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(i, obs["point"], "o", color=INK, ms=8, zorder=4, label="People" if i == 0 else None)
        ax.plot(i + 0.22, sr["point"], "s", mfc="white", mec=SR_C, mew=2, ms=7, zorder=4, label="Sample until return, forecast" if i == 0 else None)
        ax.plot(i + 0.22, cl["point"], "D", mfc="white", mec=CL_C, mew=2, ms=6, zorder=4, label="Logit, forecast" if i == 0 else None)
    ax.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax.set_xticks(x + 0.08)
    ax.set_xticklabels([it[0].replace(", ", "\n").replace(" in ", "\nin ").replace(" at ", "\nat ").replace(" worked", "\nworked") for it in items], fontsize=8.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].axhspan(-bound, bound, xmin=0.0, xmax=0.30, color="#bbbbbb", alpha=0.35, lw=0, zorder=0)
axes[0].set_ylabel("Growth of missed best options\nper doubling of the set", fontsize=9)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, 0.005, "Bars, 95% intervals from the participant bootstrap (lotteries, resampling within menu and cost). Grey band, equivalence bound.",
         fontsize=7.5, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "fig_h1_synthesis.png", dpi=200)
fig.savefig(OUT / "fig_h1_synthesis.pdf")

summary = {"across": {l: {m: g[m]["point"] for m in g} | {"interval": g["observed"]["percentile"]} for l, g in rows},
           "within": {l: {"observed": o["point"], "interval": o["percentile"], "CL": c["point"], "SR": s["point"]} for l, o, c, s in within},
           "equivalence_snack": eq}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
L = ["# H1 across datasets and the equivalence check for snacks at a glance", "",
     "| Data | People's growth per doubling [95%] | Logit forecast | Sample until return forecast |", "|---|---|---|---|"]
for l, g in rows:
    L.append(f"| {l} | {g['observed']['point']:+.3f} [{g['observed']['percentile'][0]:+.3f}, {g['observed']['percentile'][1]:+.3f}] | {g['CL']['point']:+.3f} | {g['SR']['point']:+.3f} |")
for l, o, c, s in within:
    L.append(f"| {l} | {o['point']:+.3f} [{o['percentile'][0]:+.3f}, {o['percentile'][1]:+.3f}] | {c['point']:+.3f} | {s['point']:+.3f} |")
L += ["", f"Equivalence, snacks at a glance (descriptive). Bound +-{bound:.3f} per doubling, from d = 0.2 (Scheibehenne et al. 2010) "
      f"at a mean miss rate of {100 * p_bar:.0f} percent over two doublings. 90% interval {lo90:+.3f} to {hi90:+.3f} (basic "
      f"{basic90[0]:+.3f} to {basic90[1]:+.3f}). Inside the bound: {equivalent}."]
(OUT / "table.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
