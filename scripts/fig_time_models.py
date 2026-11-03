"""Figures for the decision time models on the vault (docs/time_models_plan.md).

1. Held out R2 of within person log time per set for six models, with participant bootstrap intervals.
2. Calibration on the vault (bins with at least 20 trials), trials binned by the predicted time relative to the person's usual time (deciles),
   mean predicted against mean observed, as percent slower or faster than the person's usual time.
Usage, python3 -I scripts/fig_time_models.py VAULT_DIR EVAL_JSON OUT
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "src")
from overload import time_models as tm  # noqa: E402

vdir, ev, out = Path(sys.argv[1]), json.load(open(sys.argv[2]))["sets"], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SETS = ["dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016"]
LAB = {"dev_file": "Our files", "dev_laptop": "Our laptops", "snack": "Snacks at a glance",
       "reutskaja": "Snacks in 3 s", "cdm": "Sums", "krajbich2010": "Krajbich 2010",
       "krajbich2011": "Krajbich 2011", "folke2016": "Folke 2016"}
MOD = [("S1", "Set size only", "#9aa0a6"), ("S2", "Structure, 4 features", "#4c78a8"),
       ("TLS_looks", "Two look satisficing, 1 parameter", "#f58518"), ("S2+TLS", "Structure plus satisficing", "#e45756"),
       ("LBA_n_cal", "LBA race, growing threshold", "#54a24b"), ("Ceiling2", "Boosted trees ceiling", "#b279a2")]

fig, ax = plt.subplots(figsize=(11, 4.2))
w = 0.13
for j, (m, lab, col) in enumerate(MOD):
    x = np.arange(len(SETS)) + (j - 2.5) * w
    r = np.array([ev[s]["models"][m]["R2"] for s in SETS])
    ci = np.array([ev[s]["models"][m]["R2_ci"] for s in SETS])
    ax.bar(x, r, w * 0.9, color=col, label=lab)
    ax.errorbar(x, r, yerr=[r - ci[:, 0], ci[:, 1] - r], fmt="none", ecolor="#333", lw=0.8, capsize=1.5)
ax.axhline(0, color="#333", lw=0.8)
ax.set_xticks(np.arange(len(SETS))); ax.set_xticklabels([LAB[s] for s in SETS], fontsize=9)
ax.set_ylabel("Held out R2, within person log time")
ax.set_title("Vault participants, models fitted on the other participants")
ax.legend(fontsize=8, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(out / "time_models_vault_r2.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(12, 3.4), sharey=False)
for a, s in zip(axs, ["snack", "reutskaja", "krajbich2010", "krajbich2011"]):
    z = np.load(vdir / f"{s}_pred.npz"); pid = np.unique(z["pid"], return_inverse=True)[1]
    y = tm.center(z["logt"], pid)
    for m, lab, col in [MOD[0], MOD[3], MOD[2]]:
        p = tm.center(z[m], pid)
        q = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
        keep = np.bincount(b, minlength=10) >= 20  # bins with ties of a discrete law can be nearly empty
        px = np.array([p[b == k].mean() for k in range(10)]); oy = np.array([y[b == k].mean() for k in range(10)])
        a.plot(100 * (np.exp(px[keep]) - 1), 100 * (np.exp(oy[keep]) - 1), "o-", color=col, ms=4, lw=1.5, label=lab)
    lim = a.get_xlim(); a.plot(lim, lim, color="#999", lw=0.8, ls="--")
    a.set_title(LAB[s], fontsize=10); a.set_xlabel("Predicted, % vs usual time", fontsize=8)
    a.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("Observed, % vs usual time", fontsize=8)
axs[0].legend(fontsize=7, frameon=False)
fig.suptitle("Vault calibration, trials in deciles of predicted time", fontsize=10)
fig.tight_layout(); fig.savefig(out / "time_models_vault_calibration.png", dpi=200); plt.close(fig)
print("saved", out)
