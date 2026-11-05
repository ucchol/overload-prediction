"""Figure for the decision time write up. Time against the value of the best option on the vault, with the effect of
set size removed, observed and as predicted by sample until return's looks, two look satisficing's looks and the
structure law.

For every vault trial, the centered log time (and each model's centered prediction) has its mean at that trial's set
size removed, pooled over vault participants. Trials are then binned in quintiles of the best option's value (0 to 1)
and each point is a bin's mean, as percent against the person's usual time at that set size.
Usage, python3 -I scripts/fig_time_bestvalue.py DEV_CSV VAULT_DIR OUT
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from overload import time_models as tm  # noqa: E402
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from overload.figures import AXIS, GRID, INK, INK_2, SR_COLOR, _letters, _save, _style  # noqa: E402
from time_models_run import vault_mask  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
dev, vdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
S = load_all(dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
TLS_C = "#c2185b"
SETS = [("snack", "Snacks at a glance"), ("reutskaja", "Snacks in three seconds"), ("krajbich2010", "Two foods"),
        ("krajbich2011", "Three foods")]
LINES = [("S2", "Structure law", INK, "-"), ("SR_looks", "Sample until return looks", SR_COLOR, "-"),
         ("TLS_looks", "Two look satisficing looks", TLS_C, "-")]
B = 1000

fig, axes = plt.subplots(1, len(SETS), figsize=(7.2, 2.6))
for ax, (name, lab) in zip(axes, SETS):
    T = tm.from_timeset(S[name], SCALE[name])
    V = T.take(vault_mask(int(T.pid.max()) + 1)[T.pid])
    z = np.load(vdir / f"{name}_pred.npz")
    assert np.allclose(z["logt"], V.logt)
    pid = np.unique(V.pid, return_inverse=True)[1]
    best = tm.structure(V)[:, 1]

    def resid(v):
        c = tm.center(v, pid)
        for k in np.unique(V.n):
            m = V.n == k
            c[m] = c[m] - c[m].mean()
        return c

    q = np.quantile(best, np.linspace(0, 1, 6))
    b = np.clip(np.searchsorted(q, best, side="right") - 1, 0, 4)
    keep = [k for k in range(5) if (b == k).sum() >= 20]
    xb = np.array([best[b == k].mean() for k in keep])
    pc = lambda v: 100 * (np.exp(v) - 1)
    y = resid(z["logt"])
    P = pid.max() + 1
    W = np.array([np.bincount(np.random.default_rng(130500 + r).integers(0, P, P), minlength=P) for r in range(B)])
    obs, lo, hi = [], [], []
    for k in keep:
        m = b == k
        sw = np.bincount(pid[m], weights=y[m], minlength=P); c = np.bincount(pid[m], minlength=P)
        bt = (W @ sw) / np.maximum(W @ c, 1)
        obs.append(y[m].mean()); lo.append(np.percentile(bt, 2.5)); hi.append(np.percentile(bt, 97.5))
    obs, lo, hi = map(np.array, (obs, lo, hi))
    ax.errorbar(xb, pc(obs), yerr=[pc(obs) - pc(lo), pc(hi) - pc(obs)], fmt="o", color=INK, ms=4, mfc="white",
                mew=1.3, elinewidth=1.1, capsize=0, zorder=5)
    print(name, "best", np.round(xb, 2).tolist(), "observed", np.round(pc(obs), 1).tolist(),
          "interval", np.round(pc(lo), 1).tolist(), np.round(pc(hi), 1).tolist())
    for mdl, _, col, ls in LINES:
        p = resid(z[mdl])
        yy = pc(np.array([p[b == k].mean() for k in keep]))
        print("   ", mdl, np.round(yy, 1).tolist())
        ax.plot(xb, yy, ls, color=col, lw=1.5, zorder=4)
    _style(ax)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title(lab, loc="left", fontsize=7.5, fontweight="bold", color=INK)
    ax.set_xlabel("Value of the best option", color=INK_2, fontsize=7)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Time against the person's usual\ntime at that set size, percent", color=INK_2)
handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="", label="Observed, vault, 95% interval")] + \
          [Line2D([], [], color=c, ls=ls, lw=1.5, label=l) for _, l, c, ls in LINES]
fig.legend(handles=handles, frameon=False, fontsize=6.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           labelcolor=INK_2)
_letters(axes)
fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
print(_save(fig, out / "figT5_bestvalue"))
