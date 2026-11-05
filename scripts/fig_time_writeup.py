"""Figures for the decision time write up (paper/time_section), drawn from stored vault results.

figT2_setsize      mean time at each set size on the vault, observed and as predicted by four models
figT3_vault        held out R2 on the vault for the main models, and each model's difference from the structure law
figT4_calibration  trials in deciles of predicted time against observed time, on the vault
Also prints the calibration slope of each main model in each set (observed centered log time regressed on predicted).
Usage, python3 -I scripts/fig_time_writeup.py VAULT_DIR EVAL_JSON OUT
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, "src")
from overload import time_models as tm  # noqa: E402
from overload.figures import AXIS, GRID, INK, INK_2, MUTED, SR_COLOR, _letters, _save, _style  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
vdir, ev, out = Path(sys.argv[1]), json.load(open(sys.argv[2]))["sets"], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SETS = [("dev_file", "Our study, files"), ("dev_laptop", "Our study, laptops"), ("snack", "Snacks at a glance"),
        ("reutskaja", "Snacks in three seconds"), ("cdm", "Sums worked out"), ("krajbich2010", "Two foods"),
        ("krajbich2011", "Three foods"), ("folke2016", "Three snacks")]
LAB = dict(SETS)
TLS_C, RACE_C = "#c2185b", "#00838f"
MODELS = [("S1", "Set size law", MUTED, "o", "white"), ("S2", "Structure law", INK, "o", None),
          ("SR_looks", "Sample until return looks", SR_COLOR, "s", None),
          ("TLS_looks", "Two look satisficing looks", TLS_C, "o", None),
          ("S2+TLS", "Structure plus satisficing", TLS_C, "D", "white"),
          ("LBA_n_cal", "LBA, rising threshold", RACE_C, "^", None),
          ("Ceiling2", "Boosted trees ceiling", MUTED, "s", None)]
B = 1000


def ci_line(ax, y, pt, lo, hi, color, marker, mfc=None, size=4.2):
    ax.plot([lo, hi], [y, y], color=color, lw=1.3, solid_capstyle="round", zorder=3)
    ax.plot(pt, y, marker, color=color, ms=size, mfc=mfc or color, mew=1.2, zorder=4)


def load(s):
    z = np.load(vdir / f"{s}_pred.npz")
    pid = np.unique(z["pid"], return_inverse=True)[1]
    return z, pid


# ------------------------------------------------------------------ T3, vault scores
def fig_vault():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True, gridspec_kw={"width_ratios": [1.35, 1]})
    a, b = axes
    ys = np.arange(len(SETS))[::-1]
    off = np.linspace(0.33, -0.33, len(MODELS))
    diff_models = [m for m in MODELS if m[0] not in ("S1", "S2")]
    off_b = np.linspace(0.28, -0.28, len(diff_models))
    for y, (s, _) in zip(ys, SETS):
        for (m, lab, col, mk, mfc), o in zip(MODELS, off):
            r = ev[s]["models"][m]
            ci_line(a, y + o, r["R2"], *r["R2_ci"], col, mk, mfc)
        for (m, lab, col, mk, mfc), o in zip(diff_models, off_b):
            d = ev[s]["models"][m]["minus_S2"]
            ci_line(b, y + o, d["point"], *d["percentile"], col, mk, mfc)
            if d["clear_above"] or d["clear_below"]:
                b.text(max(d["percentile"][1], -0.13) + 0.004, y + o, "clear", fontsize=5.5, color=col, va="center")
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.axvline(0, color=AXIS, lw=0.8)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color=GRID, lw=0.5)
    a.set_yticks(ys, labels=[s[1] for s in SETS])
    a.set_xlim(-0.12, 0.23)
    b.set_xlim(-0.14, 0.06)
    a.set_xlabel("Held out share of within person\nvariance in log time (vault)", color=INK_2)
    b.set_xlabel("Difference from the structure law", color=INK_2)
    a.set_title("How well each model times choices", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    b.set_title("Against the structure law", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, mfc=mfc or c, lw=1.2, label=lab) for _, lab, c, mk, mfc in MODELS],
               frameon=False, fontsize=6.6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2,
               handletextpad=0.3, columnspacing=1.0)
    _letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8)
    return _save(fig, out / "figT3_vault")


# ------------------------------------------------------------------ T2, set size
def fig_setsize():
    sets = ["snack", "reutskaja", "cdm", "dev_file", "dev_laptop"]
    lines = [("S2", "Structure law", INK, "-"), ("TLS_looks", "Two look satisficing looks", TLS_C, "-"),
             ("LBA", "LBA, fixed threshold", RACE_C, "--"), ("LBA_n", "LBA, threshold rises with set size", RACE_C, "-")]
    fig, grid = plt.subplots(2, 3, figsize=(7.2, 4.6))
    axes = list(grid.ravel()[:len(sets)])
    leg_ax = grid.ravel()[-1]
    for ax, s in zip(axes, sets):
        z, pid = load(s)
        y = tm.center(z["logt"], pid)
        n = z["n"]; sizes = np.unique(n); P = pid.max() + 1
        W = np.array([np.bincount(np.random.default_rng(130500 + b).integers(0, P, P), minlength=P) for b in range(B)])
        def means(v):
            return np.array([v[n == k].mean() for k in sizes])
        def boot(v):
            # mean at each size over the drawn people, trials weighted by their person's draw count
            out_ = np.zeros((B, sizes.size))
            for j, k in enumerate(sizes):
                m = n == k
                sw = np.bincount(pid[m], weights=v[m], minlength=P); c = np.bincount(pid[m], minlength=P)
                out_[:, j] = (W @ sw) / np.maximum(W @ c, 1)
            return np.percentile(out_, [2.5, 97.5], axis=0)
        pc = lambda v: 100 * (np.exp(v) - 1)
        mo = means(y); lo, hi = boot(y)
        ax.errorbar(sizes, pc(mo), yerr=[pc(mo) - pc(lo), pc(hi) - pc(mo)], fmt="o", color=INK, ms=4, mfc="white",
                    mew=1.3, elinewidth=1.1, capsize=0, zorder=5)
        for m, lab, col, ls in lines:
            ax.plot(sizes, pc(means(tm.center(z[m], pid))), ls, color=col, lw=1.4, zorder=4)
        _style(ax)
        ax.set_xscale("log", base=2)
        ticks = [k for k in sizes if not (s == "dev_laptop" and k == 30)]
        ax.set_xticks(ticks, labels=[str(k) for k in ticks]); ax.minorticks_off()
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(LAB[s], loc="left", fontsize=7.5, fontweight="bold", color=INK)
        ax.set_xlabel("Options in the trial", color=INK_2, fontsize=7)
        ax.tick_params(labelsize=7)
    for ax in (grid[0, 0], grid[1, 0]):
        ax.set_ylabel("Time against the person's\nusual time, percent", color=INK_2)
    handles = [Line2D([], [], color=INK, marker="o", mfc="white", ls="", label="Observed, vault, 95% interval")] + \
              [Line2D([], [], color=c, ls=ls, lw=1.4, label=lab) for _, lab, c, ls in lines]
    leg_ax.axis("off")
    leg_ax.legend(handles=handles, frameon=False, fontsize=7.2, loc="center left", labelcolor=INK_2)
    _letters(axes)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    return _save(fig, out / "figT2_setsize")


# ------------------------------------------------------------------ T4, calibration
def fig_calibration():
    sets = ["snack", "reutskaja", "krajbich2010", "krajbich2011"]
    models = [("S2", "Structure law", INK, "o"), ("TLS_looks", "Two look satisficing looks", TLS_C, "o"),
              ("S2+TLS", "Structure plus satisficing", TLS_C, "D")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    for ax, s in zip(axes, sets):
        z, pid = load(s)
        y = tm.center(z["logt"], pid)
        lims = []
        for m, lab, col, mk in models:
            p = tm.center(z[m], pid)
            q = np.quantile(p, np.linspace(0, 1, 11)); bins = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 9)
            keep = np.bincount(bins, minlength=10) >= 20
            px = np.array([p[bins == k].mean() if keep[k] else np.nan for k in range(10)])
            oy = np.array([y[bins == k].mean() if keep[k] else np.nan for k in range(10)])
            ok = np.isfinite(px)
            ax.plot(100 * (np.exp(px[ok]) - 1), 100 * (np.exp(oy[ok]) - 1), "-", color=col, lw=1.1, zorder=3)
            ax.plot(100 * (np.exp(px[ok]) - 1), 100 * (np.exp(oy[ok]) - 1), mk, color=col, ms=3.6,
                    mfc="white" if mk == "D" else col, mew=1.1, zorder=4)
            lims += list(100 * (np.exp(px[ok]) - 1)) + list(100 * (np.exp(oy[ok]) - 1))
        lo, hi = min(lims) - 3, max(lims) + 3
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.9, ls="--", zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        _style(ax); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(LAB[s], loc="left", fontsize=7.2, fontweight="bold", color=INK)
        ax.set_xlabel("Predicted, percent", color=INK_2, fontsize=7)
        ax.tick_params(labelsize=6.8)
    axes[0].set_ylabel("Observed, percent against\nthe person's usual time", color=INK_2)
    fig.legend(handles=[Line2D([], [], color=c, marker=mk, mfc="white" if mk == "D" else c, lw=1.1, label=lab)
                        for _, lab, c, mk in models],
               frameon=False, fontsize=6.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    _letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    return _save(fig, out / "figT4_calibration")


def slopes():
    res = {}
    for s, _ in SETS:
        z, pid = load(s)
        y = tm.center(z["logt"], pid)
        res[s] = {}
        for m in ("S2", "TLS_looks", "TLS_tfit", "S2+TLS", "LBA_n_cal", "Wald_n", "Ceiling2"):
            p = tm.center(z[m], pid)
            res[s][m] = float(p @ y / (p @ p)) if p @ p > 0 else None
    json.dump(res, open(out / "calibration_slopes.json", "w"), indent=1)
    for s, r in res.items():
        print(s, {k: (round(v, 2) if v is not None else None) for k, v in r.items()})


if __name__ == "__main__":
    np.seterr(all="ignore")
    fig_vault(); fig_setsize(); fig_calibration(); slopes()
    print("saved", out)
