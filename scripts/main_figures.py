"""Main figures of the paper, drawn from stored results only.

Figure 1  the datasets, their set sizes, what each records and which question it serves
Figure 2  H1 (see scripts/h1_synthesis.py, copied here as fig2_h1)
Figure 3  H2, the return rule against dwell time, pooled and per participant
Figure 4  H3, forecasting overload at unseen set sizes, rates, trials and participants
Figure 5  decision time from trial structure (Amendment 12)
Writes results/main_figures/fig*.png and .pdf. Usage, python3 -I scripts/main_figures.py
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, "src")
from overload.figures import AXIS, GRID, INK, INK_2, MUTED, SERIES, SR_COLOR, _letters, _mean_panels, _save, _style  # noqa: E402

R = Path("results")
OUT = R / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
CL_C, DWELL_C, TRANSFER_C = SERIES["CL"], "#eb6834", "#eb6834"


def load(p):
    return json.load(open(R / p))


def ci(ax, y, e, color, marker, size=5, mfc=None, x_is_value=True):
    lo, hi = e["percentile"]
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=3)
    ax.plot(e["point"], y, marker, color=color, ms=size, mfc=mfc or color, mew=1.4, zorder=4)


# ------------------------------------------------------------------ Figure 1, datasets
def fig1():
    rows = [
        ("Our study, files", "Rubric", [8, 16, 32], "choices, time, ratings", "H1 H3 time", "This study"),
        ("Our study, laptops", "Rubric", [8, 16, 30, 32], "choices, time, ratings", "H1 H3 time", "This study"),
        ("Snacks at a glance", "Glance", [9, 16, 25, 36], "choices, time, gaze", "H1 H2 H3 time", "Thomas et al. 2021"),
        ("Snacks in three seconds", "Rushed", [4, 9, 16], "choices, time, gaze", "H1 H2 H3 time", "Reutskaja et al. 2011"),
        ("Sums worked out", "Worked out", [10, 20, 40], "choices, time, search path", "H1 H3 time", "Caplin, Dean, Martin 2011"),
        ("Lotteries at three costs", "Varied", [2, 3, 4, 5, 6], "choices", "H1 within", "Aguiar et al. 2023"),
        ("Two foods", "Glance", [2], "choices, time, gaze", "time", "Krajbich et al. 2010"),
        ("Three foods", "Glance", [3], "choices, time, gaze", "time", "Krajbich, Rangel 2011"),
        ("Three snacks", "Glance", [3], "choices, time, gaze", "time", "Folke et al. 2016"),
    ]
    tier = {"Rubric": "#52514e", "Glance": "#1baf7a", "Rushed": "#eb6834", "Worked out": "#c2185b", "Varied": "#2a78d6"}
    fig = plt.figure(figsize=(7.2, 3.4))
    ax = fig.add_axes((0.30, 0.13, 0.30, 0.80))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    for i, (name, t, sizes, rec, use, src) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot(sizes, [y] * len(sizes), "-", color=GRID, lw=3, zorder=1)
        ax.plot(sizes, [y] * len(sizes), "o", color=tier[t], ms=5.5, zorder=3)
        ax.text(1.62, y, name, ha="right", va="center", fontsize=7.5, color=INK)
        ax.text(1.62, y - 0.36, src, ha="right", va="center", fontsize=6, color=MUTED)
        ax.text(52, y, rec, ha="left", va="center", fontsize=6.8, color=INK_2)
        ax.text(52, y - 0.36, use, ha="left", va="center", fontsize=6.3, color=MUTED, style="italic")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(1.7, 46)
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Options in the set", color=INK_2)
    lab = {"Rubric": "Judged on a rubric", "Glance": "Judged at a glance", "Rushed": "Judged in a rush",
           "Worked out": "Worked out", "Varied": "Cost varied by design"}
    handles = [Line2D([], [], marker="o", ls="", color=c, label=lab[k]) for k, c in tier.items()]
    fig.legend(handles=handles, frameon=False, fontsize=6.8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01), labelcolor=INK_2,
               handletextpad=0.2, columnspacing=1.0)
    fig.text(0.62, 0.955, "Records and use", fontsize=7, color=INK_2, fontweight="bold")
    fig.text(0.02, 0.955, "Dataset and source", fontsize=7, color=INK_2, fontweight="bold")
    fig.subplots_adjust(bottom=0.2)
    ax.set_position((0.30, 0.20, 0.30, 0.73))
    return _save(fig, OUT / "fig1_datasets")


# ------------------------------------------------------------------ Figure 3, H2
def fig3():
    s = load("h2_return/summary.json")
    ind = load("individual_level/summary.json")
    names = {"snack": "Snacks at a glance", "reutskaja": "Snacks in\nthree seconds"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1, 1.15]})
    a, b, c = axes
    # a, gain over CL from each gaze cue
    for j, ds in enumerate(("snack", "reutskaja")):
        y = 1 - j
        ci(a, y + 0.12, s[f"{ds}|gain|return"], SR_COLOR, "s")
        ci(a, y - 0.12, s[f"{ds}|gain|dwell"], DWELL_C, "o")
    a.set_yticks([1, 0], labels=[names["snack"], names["reutskaja"]])
    a.set_xlabel("Gain over CL, bits per choice", color=INK_2)
    a.set_xlim(0, 1.6)
    a.set_title("Which gaze cue predicts the choice", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    # b, share of trials where the first item looked at twice was chosen, by set size
    for ds, marker, color in (("snack", "o", INK), ("reutskaja", "s", MUTED)):
        ks = sorted([k for k in s if k.startswith(f"{ds}|share_return_chosen|")], key=lambda k: int(k.split("n=")[1]))
        n = np.array([int(k.split("n=")[1]) for k in ks], float)
        p = np.array([s[k]["point"] for k in ks]); lo = np.array([s[k]["percentile"][0] for k in ks]); hi = np.array([s[k]["percentile"][1] for k in ks])
        b.fill_between(n, lo, hi, color=GRID, lw=0)
        b.plot(n, p, marker=marker, color=color, lw=1.4, ms=4.5, label=names[ds].replace("\n", " "))
    b.set_xscale("log", base=2)
    b.set_xticks([4, 9, 16, 36], labels=["4", "9", "16", "36"])
    b.minorticks_off()
    b.set_ylim(0, 1)
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    b.set_xlabel("Options in the set", color=INK_2)
    b.set_ylabel("Chose the first item\nlooked at twice", color=INK_2)
    b.set_title("The return rule, as stated", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    b.text(4, 0.86, "In three seconds", fontsize=6.5, color=MUTED)
    b.text(9, 0.33, "At a glance", fontsize=6.5, color=INK)
    # c, per participant
    for j, ds in enumerate(("snack", "reutskaja")):
        y = 1 - j
        ci(c, y + 0.12, ind[f"H2|{ds}|share_dwell_beats_return"], DWELL_C, "o")
        ci(c, y - 0.12, ind[f"H2|{ds}|share_return_beats_CL"], SR_COLOR, "s")
    c.set_yticks([1, 0], labels=[names["snack"], names["reutskaja"]])
    c.set_xlim(0, 104)
    c.axvline(50, color=AXIS, lw=0.8, ls="--")
    c.set_xlabel("Participants, percent", color=INK_2)
    c.set_title("Person by person", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    for ax in axes:
        _style(ax)
    for ax in (a, c):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_ylim(-0.5, 1.5)
    fig.legend(handles=[Line2D([], [], color=SR_COLOR, marker="s", label="Return cue (a), return beats CL (c)"),
                        Line2D([], [], color=DWELL_C, marker="o", label="Dwell cue (a), dwell beats return (c)")],
               frameon=False, fontsize=6.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), labelcolor=INK_2)
    _letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.2)
    return _save(fig, OUT / "fig3_h2_return")


# ------------------------------------------------------------------ Figure 4, H3
def fig4():
    conf = load("overload_premonition/summary.json")
    snack = load("overload_snack_all/summary.json")
    ind = load("individual_level/summary.json")
    P = conf["panels"]
    panels = [("Snacks at a glance", snack["panels"]["snack"]), ("Snacks in three seconds", P["reutskaja"]),
              ("Sums worked out", _mean_panels([P[k] for k in ("cdm_standard_3", "cdm_process_3", "cdm_standard_7", "cdm_process_7")]))]
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.62, wspace=0.38)
    top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax, (title, p) in zip(top, panels):
        _style(ax)
        x = np.array(p["sizes"], float)
        lo, hi = np.array(p["rate_interval"]["observed"])
        ax.fill_between(x, lo, hi, color=GRID, lw=0, zorder=1)
        ax.plot(x, p["rate"]["observed"], color=INK, lw=1.8, marker="o", ms=5, zorder=4)
        ax.plot(x, p["rate"]["CL"], color=CL_C, lw=1.1, marker="o", ms=3.5, zorder=3)
        ax.plot(x, p["rate"]["SR"], color=SR_COLOR, lw=1.1, marker="s", ms=3.5, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(x, labels=[str(int(v)) for v in x])
        ax.minorticks_off()
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Options in the set", color=INK_2)
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", color=INK)
    top[0].set_ylabel("Missed the best option", color=INK_2)
    # d, which trials end in a miss
    d = fig.add_subplot(gs[1, 0:2])
    _style(d)
    d.grid(axis="y", visible=False); d.grid(axis="x", color=GRID, lw=0.6)
    sets = [("Snacks at a glance", snack["datasets"]["snack"]["auc"]), ("Snacks in three seconds", conf["datasets"]["reutskaja"]["auc"]),
            ("Sums worked out", conf["datasets"]["cdm"]["auc"])]
    for j, (name, a) in enumerate(sets):
        y = 2 - j
        ci(d, y + 0.14, a["SR"], SR_COLOR, "s")
        ci(d, y - 0.14, a["near_best"], MUTED, "o", mfc="white")
    d.axvline(0.5, color=AXIS, lw=0.8, ls="--")
    d.set_yticks([2, 1, 0], labels=[s[0].replace(" in ", "\nin ").replace(" at ", "\nat ").replace(" worked", "\nworked") for s in sets])
    d.set_ylim(-0.6, 2.6)
    d.set_xlim(0.45, 0.8)
    d.set_xlabel("Area under the curve for trials that end in a miss", color=INK_2)
    d.set_title("Which trials end in a miss", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    # e, participants
    e = fig.add_subplot(gs[1, 2])
    _style(e)
    e.grid(axis="y", visible=False); e.grid(axis="x", color=GRID, lw=0.6)
    for j, ds in enumerate(("snack", "reutskaja")):
        y = 1 - j
        ci(e, y + 0.12, ind[f"H3|{ds}|rank_corr_observed_SR"], SR_COLOR, "s")
        ci(e, y - 0.12, ind[f"H3|{ds}|rank_corr_observed_CL"], CL_C, "o")
    e.axvline(0, color=AXIS, lw=0.8, ls="--")
    e.set_yticks([1, 0], labels=["Snacks at\na glance", "Snacks in\nthree seconds"])
    e.set_ylim(-0.5, 1.5)
    e.set_xlim(-0.1, 0.8)
    e.set_xlabel("Rank correlation, own growth\nand forecast growth", color=INK_2)
    e.set_title("Who overloads", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=INK, marker="o", lw=1.8, label="People"),
                        Line2D([], [], color=CL_C, marker="o", label="CL forecast"),
                        Line2D([], [], color=SR_COLOR, marker="s", label="Sample until return forecast"),
                        Line2D([], [], color=MUTED, marker="o", mfc="white", label="Count of near best options")],
               frameon=False, fontsize=6.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=INK_2)
    _letters(top + [d, e])
    fig.subplots_adjust(top=0.9, bottom=0.11, left=0.17, right=0.98)
    return _save(fig, OUT / "fig4_h3_forecast")


# ------------------------------------------------------------------ Figure 5, time
def fig5():
    t = load("time_structure/summary.json")
    sets = [("dev_file", "Our study, files"), ("dev_laptop", "Our study, laptops"), ("snack", "Snacks at a glance"),
            ("reutskaja", "Snacks in three seconds"), ("cdm", "Sums worked out"), ("krajbich2010", "Two foods"),
            ("krajbich2011", "Three foods"), ("folke2016", "Three snacks")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1, 1]})
    a, b, c = axes
    ys = np.arange(len(sets))[::-1]
    for y, (k, _) in zip(ys, sets):
        ci(a, y + 0.22, t[f"{k}|R2|S1"], MUTED, "o", mfc="white")
        ci(a, y, t[f"{k}|R2|S2"], INK, "o")
        ci(a, y - 0.22, t[f"{k}|T2"], TRANSFER_C, "D", size=4.5)
        if t[f"{k}|T2"].get("supported"):
            a.text(t[f"{k}|T2"]["percentile"][1] + 0.006, y - 0.22, "supported", fontsize=5.8, color=TRANSFER_C, va="center")
        ci(b, y, t[f"{k}|T1"], INK, "o")
        if t[f"{k}|T1"].get("supported"):
            b.text(t[f"{k}|T1"]["percentile"][1] + 0.006, y, "supported", fontsize=5.8, color=INK, va="center")
        ci(c, y + 0.12, t[f"{k}|T3"], SR_COLOR, "s", size=4.5)
        ci(c, y - 0.12, t[f"{k}|S4_minus_S2"], CL_C, "o", size=4)
    for ax in axes:
        _style(ax)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
        ax.axvline(0, color=AXIS, lw=0.8)
    a.set_yticks(ys, labels=[s[1] for s in sets])
    a.set_xlim(-0.06, 0.2); b.set_xlim(-0.04, 0.16); c.set_xlim(-0.03, 0.03)
    a.set_xlabel("Held out share of within person\nvariance in log time", color=INK_2)
    b.set_xlabel("Structure minus counting options", color=INK_2)
    c.set_xlabel("Added to structure", color=INK_2)
    a.set_title("How well time is predicted", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    b.set_title("Beyond set size (T1)", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    c.set_title("Model signals (T3)", loc="left", fontsize=7.5, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], color=MUTED, marker="o", mfc="white", label="Counting options"),
                        Line2D([], [], color=INK, marker="o", label="Trial structure"),
                        Line2D([], [], color=TRANSFER_C, marker="D", label="Structure law from the other sets"),
                        Line2D([], [], color=SR_COLOR, marker="s", label="Search model's looks"),
                        Line2D([], [], color=CL_C, marker="o", label="CL entropy")],
               frameon=False, fontsize=6.6, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.01), labelcolor=INK_2,
               handletextpad=0.2, columnspacing=0.9)
    _letters(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    return _save(fig, OUT / "fig5_time")


if __name__ == "__main__":
    np.seterr(all="ignore")
    made = fig1() + fig3() + fig4() + fig5()
    for ext in ("png", "pdf"):
        src = R / "h1_synthesis" / f"fig_h1_synthesis.{ext}"
        if src.exists():
            shutil.copy(src, OUT / f"fig2_h1.{ext}")
    print("\n".join(str(p) for p in made))
