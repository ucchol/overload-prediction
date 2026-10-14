"""Figures of the results, standalone PNG and PDF files.

Colors follow a validated categorical palette (all pairs pass the color vision deficiency checks), and every
series also differs in marker shape, so identity never rests on color alone. Observed behavior is drawn in
ink, models in color. Text uses ink tones only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
SERIES = {"CL": "#2a78d6", "DbS_avg": "#eb6834", "MLBA_avg": "#1baf7a"}
MARKERS = {"CL": "o", "DbS_avg": "D", "MLBA_avg": "^"}
LABELS = {"CL": "Conditional logit", "DbS_avg": "Decision by sampling", "MLBA_avg": "MLBA"}
TASK_TITLES = {"file": "File selection", "laptop": "Laptop purchase"}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure_h1(summary: dict, out: Path) -> list[Path]:
    """Narrowing with set size. Top, observed and predicted concentration. Bottom, the set size scaling test."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey="row")
    fig.patch.set_facecolor(SURFACE)

    for col, task in enumerate(("file", "laptop")):
        # Behavior: effective options as a share of the set, people against held out CL, per trial
        ax = axes[0, col]
        _style(ax)
        for t in summary[task]["B1"]["per_trial"]:
            low = t["trial"].split("_")[1].startswith("L")
            x = t["options"] * (0.93 if low else 1.07)
            obs, mod = 100 * t["observed_effective_share"], 100 * t["model_effective_share"]
            ax.plot([x, x], [obs, mod], color=MUTED, linewidth=0.8, zorder=1)
            shape = "o" if low else "s"
            ax.scatter(x, mod, s=34, marker=shape, color=SERIES["CL"], edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.scatter(x, obs, s=34, marker=shape, color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32], labels=["8", "16", "32"])
        ax.minorticks_off()
        ax.set_xlim(6.3, 40)
        ax.set_ylim(0, 100)
        ax.set_title(TASK_TITLES[task], color=INK, fontsize=9, loc="left", fontweight="bold")
        b1 = summary[task]["B1"]
        m, s = b1["mean_gap"], b1["slope_gap_per_doubling"]
        ax.text(
            0.02,
            0.04,
            f"Gap {100 * m['point']:+.1f} points [{100 * m['percentile'][0]:.1f}, {100 * m['percentile'][1]:.1f}]\n"
            f"per doubling {100 * s['point']:+.1f} [{100 * s['percentile'][0]:.1f}, {100 * s['percentile'][1]:.1f}]",
            transform=ax.transAxes,
            color=INK_2,
            fontsize=7.5,
            va="bottom",
        )
        if col == 0:
            ax.set_ylabel("Effective options, % of the set", color=INK_2)
        ax.set_xlabel("Options in the set", color=INK_2)

        # Algorithm: held out completeness at each beta minus beta 0, three model families
        ax = axes[1, col]
        _style(ax)
        ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
        cells = summary[task]["A1a"]["cells"]
        betas = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for k, fam in enumerate(("CL", "DbS_avg", "MLBA_avg")):
            off = (k - 1) * 0.07
            ys, lo, hi = [], [], []
            for beta in betas:
                c = cells[f"{fam}|beta={beta:+.1f}"]
                d = c.get("minus_beta0", {"point": 0.0, "percentile": [0.0, 0.0]})
                ys.append(d["point"])
                lo.append(d["point"] - d["percentile"][0])
                hi.append(d["percentile"][1] - d["point"])
            xs = np.array(betas) + off
            ax.errorbar(
                xs,
                ys,
                yerr=[lo, hi],
                fmt="none",
                ecolor=SERIES[fam],
                elinewidth=1.0,
                capsize=0,
                zorder=2,
            )
            ax.plot(xs, ys, color=SERIES[fam], linewidth=1.4, zorder=2)
            ax.scatter(xs, ys, s=30, marker=MARKERS[fam], color=SERIES[fam], edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.set_xticks(betas, labels=["−1", "−0.5", "0", "+0.5", "+1"])
        ax.set_xlim(-1.25, 1.25)
        if col == 0:
            ax.set_ylabel("Held out completeness minus\nconstant sensitivity, points", color=INK_2)
        ax.set_xlabel("Sensitivity scaling with set size, beta\n(negative, noisier with more options)", color=INK_2)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=6, label="Participants, brief display"),
        Line2D([], [], marker="s", linestyle="", color=INK, markersize=6, label="Participants, full display"),
        Line2D([], [], marker="o", linestyle="", color=SERIES["CL"], markersize=6, label="Conditional logit, held out"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        ncol=3,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fam_handles = [
        Line2D([], [], marker=MARKERS[f], color=SERIES[f], linewidth=1.4, markersize=6, label=LABELS[f])
        for f in ("CL", "DbS_avg", "MLBA_avg")
    ]
    axes[1, 1].legend(handles=fam_handles, frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_2)
    for ax, letter in zip(axes.ravel(), "abcd", strict=True):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", ha="right")
    fig.tight_layout(h_pad=2.2, rect=(0, 0, 1, 0.96))
    paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300, facecolor=SURFACE)
    fig.savefig(paths[1], facecolor=SURFACE)
    plt.close(fig)
    return paths


def figure_h1_confirmatory(summary: dict, out: Path) -> list[Path]:
    """Confirmatory H1 on the snack data. Attention, sensitivity by set size and the set size scaling test."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    fig.patch.set_facecolor(SURFACE)
    sizes = np.array(summary["descriptives"]["set_sizes"], float)
    d = summary["descriptives"]

    def size_axis(ax: plt.Axes) -> None:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes, labels=[f"{int(n)}" for n in sizes])
        ax.minorticks_off()
        ax.set_xlim(7.5, 43)
        ax.set_xlabel("Items on screen", color=INK_2)

    # a, the share of the display looked at
    ax = axes[0]
    _style(ax)
    looked = d["share_looked_at"]
    y = np.array([100 * e["point"] for e in looked])
    lo = y - np.array([100 * e["percentile"][0] for e in looked])
    hi = np.array([100 * e["percentile"][1] for e in looked]) - y
    ax.errorbar(sizes, y, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.0, capsize=0, zorder=2)
    ax.plot(sizes, y, color=INK, linewidth=1.2, zorder=2)
    ax.scatter(sizes, y, s=30, marker="o", color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    size_axis(ax)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Items looked at, % of the display", color=INK_2)
    ax.set_title("Attention narrows", color=INK, fontsize=9, loc="left", fontweight="bold")

    # b, CL's sensitivity fitted to each set size, with the fitted elasticity
    ax = axes[1]
    _style(ax)
    c3 = summary["C3"]
    grid = np.geomspace(8.5, 38, 60)
    for key, version, marker, face, style in (
        ("all_items_C3_trials", "all", "o", SERIES["CL"], "-"),
        ("seen_items_C3_trials", "seen", "s", SURFACE, "--"),
    ):
        rows = d["lambda_alone"][key]
        y = np.array([e["point"] for e in rows])
        lo = y - np.array([e["percentile"][0] for e in rows])
        hi = np.array([e["percentile"][1] for e in rows]) - y
        x = sizes * (0.96 if version == "all" else 1.04)
        lam, beta = c3[f"lambda_{version}"], c3[f"beta_{version}"]["point"]
        ax.plot(grid, lam * ((grid - 1) / 7) ** beta, color=SERIES["CL"], linewidth=1.2, linestyle=style, zorder=1)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=SERIES["CL"], elinewidth=1.0, capsize=0, zorder=2)
        ax.scatter(x, y, s=30, marker=marker, facecolor=face, edgecolor=SERIES["CL"], linewidth=1.2, zorder=3)
    size_axis(ax)
    ax.set_ylim(0, 3.0)
    ax.set_ylabel("Sensitivity to liking, CL", color=INK_2)
    ax.set_title("Sensitivity rises", color=INK, fontsize=9, loc="left", fontweight="bold")
    b_all, b_seen = c3["beta_all"], c3["beta_seen"]
    ax.text(
        0.96,
        0.05,
        "Elasticity with set size\n"
        f"all {b_all['point']:+.2f} [{b_all['percentile'][0]:.2f}, {b_all['percentile'][1]:.2f}]\n"
        f"looked at {b_seen['point']:+.2f} [{b_seen['percentile'][0]:.2f}, {b_seen['percentile'][1]:.2f}]",
        transform=ax.transAxes,
        color=INK_2,
        fontsize=7,
        va="bottom",
        ha="right",
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=SERIES["CL"], markerfacecolor=SERIES["CL"], label="All items"),
            Line2D(
                [], [], marker="s", color=SERIES["CL"], markerfacecolor=SURFACE, linestyle="--", label="Looked-at items"
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
        labelcolor=INK_2,
    )

    # c, held-out likelihood against the set size scaling, all items
    ax = axes[2]
    _style(ax)
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
    betas = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cells = summary["C2"]["cells"]
    for k, fam in enumerate(("CL", "DbS_avg", "MLBA_avg")):
        off = (k - 1) * 0.07
        ys, lo, hi = [], [], []
        for beta in betas:
            c = cells[f"{fam}|beta={beta:+.1f}|all"]
            pc = c["minus_beta0"]["percent"] if "minus_beta0" in c else {"point": 0.0, "percentile": [0.0, 0.0]}
            ys.append(pc["point"])
            lo.append(pc["point"] - pc["percentile"][0])
            hi.append(pc["percentile"][1] - pc["point"])
        xs = np.array(betas) + off
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=SERIES[fam], elinewidth=1.0, capsize=0, zorder=2)
        ax.plot(xs, ys, color=SERIES[fam], linewidth=1.4, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=28,
            marker=MARKERS[fam],
            color=SERIES[fam],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
            label=LABELS[fam],
        )
    ax.set_xticks(betas, labels=["−1", "−0.5", "0", "+0.5", "+1"])
    ax.set_xlim(-1.25, 1.25)
    ax.set_xlabel("Set size scaling, beta", color=INK_2)
    ax.set_ylabel("Held-out likelihood against beta 0, %", color=INK_2)
    ax.set_title("Noise predicts worst", color=INK, fontsize=9, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK_2)

    for ax, letter in zip(axes, "abc", strict=True):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", ha="right")
    fig.tight_layout(w_pad=1.6)
    paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300, facecolor=SURFACE)
    fig.savefig(paths[1], facecolor=SURFACE)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------- overload, time and search
SR_COLOR = "#4a3aa7"  # the proposed search model, validated with the three model colors above (all pairs)
MODEL_STYLE = {
    "CL": (SERIES["CL"], "o", "CL"),
    "DbS_avg": (SERIES["DbS_avg"], "D", "DbS and MLBA"),
    "MLBA_avg": (SERIES["MLBA_avg"], "^", "MLBA"),
    "EBA": (MUTED, "v", "Elimination by aspects"),
    "SR": (SR_COLOR, "s", "Sample until return"),
}


def _save(fig: plt.Figure, out: Path) -> list[Path]:
    paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=300, facecolor=SURFACE)
    fig.savefig(paths[1], facecolor=SURFACE)
    plt.close(fig)
    return paths


def _letters(axes: Sequence[plt.Axes]) -> None:
    for ax, letter in zip(axes, "abcdefgh", strict=False):
        ax.text(-0.02, 1.05, letter, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", ha="right")


def _options_axis(ax: plt.Axes, label: bool = True) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32], labels=["8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(6.3, 40)
    if label:
        ax.set_xlabel("Options in the trial", color=INK_2)


def figure_overload(summary: dict, out: Path) -> list[Path]:
    """Where people felt overloaded. Overload against each person's own average, regret and time, by trial."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharey="row")
    fig.patch.set_facecolor(SURFACE)
    M = summary["measures"]
    rows = (
        ("overload_vs_average", "Overload against own average,\nscale points (1 to 5 scale)", 1.0),
        ("regret", "Regret, points\n(0 means a best option)", 1.0),
        ("geo_seconds", "Time on the choice, s\n(geometric mean)", 1.0),
    )
    for col, task in enumerate(("file", "laptop")):
        trials = [t for t in summary["trials"] if t["task"] == task]
        for row, (name, ylabel, scale) in enumerate(rows):
            ax = axes[row, col]
            _style(ax)
            if name == "overload_vs_average":
                ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)
            for t in trials:
                e = M[f"trial|{t['key']}|{name}"]
                low = t["information"] == "low"
                x = (32 if t["options"] == 30 else t["options"]) * (0.93 if low else 1.07)
                y = scale * e["point"]
                lo, hi = (scale * v for v in e["percentile"])
                ax.plot([x, x], [lo, hi], color=INK, linewidth=1.0, zorder=2)
                ax.scatter(
                    x, y, s=34, marker="o" if low else "s", color=INK, edgecolor=SURFACE, linewidth=1.2, zorder=3
                )
            _options_axis(ax, label=row == 2)
            if col == 0:
                ax.set_ylabel(ylabel, color=INK_2)
            if row == 0:
                ax.set_title(TASK_TITLES[task], color=INK, fontsize=9, loc="left", fontweight="bold")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=INK, markersize=6, label="Brief information"),
        Line2D([], [], marker="s", linestyle="", color=INK, markersize=6, label="Full information"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper center", ncol=2, labelcolor=INK_2)
    _letters(axes.ravel())
    fig.tight_layout(h_pad=1.6, rect=(0, 0, 1, 0.97))
    return _save(fig, out)


def figure_time(summary: dict, out: Path) -> list[Path]:
    """People's typical decision time on each trial against each model's held out time law, and the typical error."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    fig.patch.set_facecolor(SURFACE)
    T = summary["time"]
    shown = ("constant", "CL", "DbS_avg", "SR")
    style = {name: MODEL_STYLE[name] for name in ("CL", "DbS_avg", "SR")}
    style["constant"] = (MUTED, "v", "No model (constant)")
    for col, task in enumerate(("file", "laptop")):
        ax = axes[col]
        _style(ax)
        x = np.arange(6)
        ax.scatter(x, T[task]["seconds"], s=40, marker="o", color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=4)
        for k, name in enumerate(shown):
            color, marker, _ = style[name]
            ax.scatter(
                x + (k - 1.5) * 0.12,
                T[task]["held_out"][name],
                s=22,
                marker=marker,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=3,
            )
        labels = [f"{n}\n{'brief' if j % 2 == 0 else 'full'}" for j, n in enumerate((8, 8, 16, 16, 32, 32))]
        ax.set_xticks(x, labels=labels, fontsize=6.5)
        ax.set_ylim(0, 160 if task == "file" else 60)
        ax.set_xlabel("Options and information on the trial", color=INK_2)
        ax.set_title(TASK_TITLES[task], color=INK, fontsize=9, loc="left", fontweight="bold")
        if col == 0:
            ax.set_ylabel("Typical decision time, s", color=INK_2)
    handles = [Line2D([], [], marker="o", linestyle="", color=INK, label="People")] + [
        Line2D([], [], marker=style[n][1], linestyle="", color=style[n][0], label=style[n][2]) for n in shown
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=6.5,
        loc="upper center",
        ncol=5,
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, 1.0),
    )
    # c, held out time error
    ax = axes[2]
    _style(ax)
    names = ("constant", "CL", "DbS_avg", "EBA", "SR")
    labels = ("No model", "CL", "DbS, MLBA", "EBA", "Sample\nuntil return")
    for k, task in enumerate(("file", "laptop")):
        for i, name in enumerate(names):
            e = T[task]["error"][name]
            color = INK if name == "constant" else MODEL_STYLE[name][0]
            y = i + (k - 0.5) * 0.32
            ax.plot(e["percentile"], [y, y], color=color, linewidth=1.0, zorder=2)
            ax.scatter(
                e["point"], y, s=26, marker="o" if task == "file" else "s", color=color, edgecolor=SURFACE, zorder=3
            )
    ax.set_yticks(range(len(names)), labels=labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_xlabel("Typical error of held out time, %\ncircles files, squares laptops", color=INK_2)
    _letters(axes)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 0.99, 0.93))
    return _save(fig, out)


def figure_process_choices(summary: dict, out: Path) -> list[Path]:
    """Held out completeness of every model. The search process changes little when the values stay the same."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    order = ("QR", "CL", "SR", "DbS_avg", "SR_DbS", "MLBA_avg", "SR_MLBA", "EBA", "Accept_read")
    labels = {
        "QR": "QR",
        "CL": "CL",
        "SR": "Sample until return, CL values",
        "DbS_avg": "DbS",
        "SR_DbS": "Sample until return, DbS values",
        "MLBA_avg": "MLBA",
        "SR_MLBA": "Sample until return, MLBA values",
        "EBA": "Elimination by aspects",
        "Accept_read": "Accept in reading order",
    }
    colors = {"CL": SERIES["CL"], "DbS_avg": SERIES["DbS_avg"], "MLBA_avg": SERIES["MLBA_avg"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    C = summary["choices"]
    for col, task in enumerate(("file", "laptop")):
        ax = axes[col]
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.axvline(0, color=AXIS, linewidth=1.0)
        for i, m in enumerate(order):
            e = C[task]["held_out"][m]
            color = SR_COLOR if m.startswith("SR") else colors.get(m, MUTED)
            ax.plot(e["percentile"], [i, i], color=color, linewidth=1.2, zorder=2)
            ax.scatter(
                e["point"], i, s=30, marker="s" if m.startswith("SR") else "o", color=color, edgecolor=SURFACE, zorder=3
            )
            ax.text(e["percentile"][1] + 1.5, i, f"{e['point']:.0f}", va="center", fontsize=6.5, color=INK_2)
        ax.set_yticks(range(len(order)), labels=[labels[m] for m in order])
        ax.invert_yaxis()
        ax.set_xlabel("Held out completeness, %", color=INK_2)
        ax.set_title(TASK_TITLES[task], color=INK, fontsize=9, loc="left", fontweight="bold")
    axes[0].set_xlim(-65, 50)
    axes[1].set_xlim(-5, 85)
    _letters(axes)
    fig.tight_layout(w_pad=1.0)
    return _save(fig, out)


def figure_premonition(summary: dict, out: Path) -> list[Path]:
    """Overload index of each trial against forecasts made before anyone chose."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    P = summary["premonition"]
    for col, task in enumerate(("file", "laptop")):
        ax = axes[col]
        _style(ax)
        y = np.array(P[task]["overload"])
        x = np.array(P[task]["forecasts"]["looks_SR"])
        for j in range(y.size):
            ax.scatter(x[j], y[j], s=34, marker="o" if j % 2 == 0 else "s", color=SR_COLOR, edgecolor=SURFACE, zorder=3)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 10)
        ax.plot(xs, a + b * xs, color=SR_COLOR, linewidth=1.0, alpha=0.7, zorder=2)
        rho = P[task]["by_forecast"]["looks_SR"]["spearman"]
        rho_n = P[task]["by_forecast"]["options"]["spearman"]
        ax.text(
            0.03,
            0.95,
            f"Rank agreement with looks {rho['point']:+.2f}\nwith number of options {rho_n['point']:+.2f}",
            transform=ax.transAxes,
            fontsize=7,
            color=INK_2,
            va="top",
        )
        ax.set_xlabel("Expected looks before a choice, sample until return", color=INK_2)
        if col == 0:
            ax.set_ylabel("Overload index (1 to 5)", color=INK_2)
        ax.set_title(TASK_TITLES[task], color=INK, fontsize=9, loc="left", fontweight="bold")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=SR_COLOR, label="Brief information"),
        Line2D([], [], marker="s", linestyle="", color=SR_COLOR, label="Full information"),
    ]
    axes[1].legend(handles=handles, frameon=False, fontsize=7, loc="center right", labelcolor=INK_2)
    _letters(axes)
    fig.tight_layout(w_pad=1.4)
    return _save(fig, out)


# ----------------------------------------------------------------------------- confirmatory search tests (Amendment 5)
FORECAST_LABELS = {
    "looks_SR": ("Looks, sample until return", SR_COLOR, "s"),
    "entropy_CL": ("Spread of choices, CL", SERIES["CL"], "o"),
    "entropy_DbS_avg": ("Spread of choices, DbS", SERIES["DbS_avg"], "D"),
    "entropy_MLBA_avg": ("Spread of choices, MLBA", SERIES["MLBA_avg"], "^"),
    "top_rating": ("Rating of the best snack", INK, "o"),
    "rating_sd": ("Spread of the ratings", INK, "o"),
    "top_count": ("Ties at the best rating", INK, "o"),
}


def _snack_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_xticks([9, 16, 25, 36], labels=["9", "16", "25", "36"])
    ax.minorticks_off()
    ax.set_xlim(7.5, 43)
    ax.set_xlabel("Snacks on display", color=INK_2)


def figure_search_confirmatory(summary: dict, out: Path) -> list[Path]:
    """Confirmatory tests of the search model on the snack data. Growth of search, slow decisions and choices."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(7.2, 6.2))
    fig.patch.set_facecolor(SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, :]), fig.add_subplot(grid[0, 1])]
    sizes = np.array([9.0, 16.0, 25.0, 36.0])
    s1 = summary["S1"]

    # a, search grows like the square root of the set size
    ax = axes[0]
    _style(ax)
    ax.set_yscale("log")
    ax.plot(sizes, sizes, color=MUTED, linewidth=1.0, linestyle="--", zorder=1)
    for key, color, marker, x in (("observed", INK, "o", sizes * 0.985), ("predicted", SR_COLOR, "s", sizes * 1.015)):
        y = np.array(s1[f"{key}_by_size"])
        lo, hi = np.array(s1[f"{key}_by_size_interval"])
        ax.plot(x, y, color=color, linewidth=1.2, zorder=2)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=color, elinewidth=1.0, capsize=0, zorder=2)
        ax.scatter(x, y, s=30, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    e_obs, e_pred = s1["observed_elasticity"], s1["predicted_elasticity"]
    ax.text(
        0.97,
        0.04,
        f"Growth with set size (elasticity)\npeople {e_obs['point']:.2f} [{e_obs['percentile'][0]:.2f}, "
        f"{e_obs['percentile'][1]:.2f}]\nsample until return {e_pred['point']:.2f} "
        f"[{e_pred['percentile'][0]:.2f}, {e_pred['percentile'][1]:.2f}]\nlooking at every snack 1",
        transform=ax.transAxes,
        fontsize=7,
        color=INK_2,
        va="bottom",
        ha="right",
    )
    _snack_axis(ax)
    ax.set_yticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])
    ax.set_ylim(0.9, 45)
    ax.set_ylabel("Distinct snacks looked at", color=INK_2)
    ax.set_title("Search grows like the square root", color=INK, fontsize=9, loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=INK, label="People"),
            Line2D([], [], marker="s", color=SR_COLOR, label="Sample until return"),
            Line2D([], [], color=MUTED, linestyle="--", label="Looking at every snack"),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
        labelcolor=INK_2,
    )

    # c, what foresees a slow decision (wide bottom panel)
    ax = axes[1]
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.axvline(0, color=AXIS, linewidth=1.0)
    S3 = summary["S3"]["correlation"]
    names = list(FORECAST_LABELS)
    for i, name in enumerate(names):
        label, color, marker = FORECAST_LABELS[name]
        e = S3[name]
        ax.plot(e["percentile"], [i, i], color=color, linewidth=1.2, zorder=2)
        ax.scatter(e["point"], i, s=30, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_yticks(range(len(names)), labels=[FORECAST_LABELS[n][0] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(-0.4, 0.2)
    ax.set_xlabel("Rank correlation with response time", color=INK_2)
    ax.set_title("What foresees a slow decision", color=INK, fontsize=9, loc="left", fontweight="bold")

    # b, choices of unseen set sizes
    ax = axes[2]
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    tc = summary["choices"]["times_chance"]
    order = ("SR", "CL", "DbS_avg", "MLBA_avg")
    labels = {"SR": "Sample until return", "CL": "CL", "DbS_avg": "DbS", "MLBA_avg": "MLBA"}
    for i, m in enumerate(order):
        color, marker, _ = MODEL_STYLE[m]
        e = tc[m]
        ax.plot(e["percentile"], [i, i], color=color, linewidth=1.2, zorder=2)
        ax.scatter(e["point"], i, s=30, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.text(e["percentile"][1] + 0.08, i, f"{e['point']:.2f}", va="center", fontsize=7, color=INK_2)
    ax.axvline(1.0, color=AXIS, linewidth=1.0)
    ax.set_yticks(range(len(order)), labels=[labels[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0.8, 5.2)
    ax.set_xlabel("Likelihood of the choice, times chance", color=INK_2)
    ax.set_title("Choices as well as the best", color=INK, fontsize=9, loc="left", fontweight="bold")

    _letters([axes[0], axes[2], axes[1]])
    fig.tight_layout(h_pad=2.4, w_pad=2.0)
    return _save(fig, out)
