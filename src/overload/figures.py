"""Figures of the results, standalone PNG and PDF files.

Colors follow a validated categorical palette (all pairs pass the color vision deficiency checks), and every
series also differs in marker shape, so identity never rests on color alone. Observed behavior is drawn in
ink, models in color. Text uses ink tones only.
"""

from __future__ import annotations

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
    axes[0, 1].legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK_2)
    fam_handles = [
        Line2D([], [], marker=MARKERS[f], color=SERIES[f], linewidth=1.4, markersize=6, label=LABELS[f])
        for f in ("CL", "DbS_avg", "MLBA_avg")
    ]
    axes[1, 1].legend(handles=fam_handles, frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_2)
    for ax, letter in zip(axes.ravel(), "abcd", strict=True):
        ax.text(-0.02, 1.04, letter, transform=ax.transAxes, color=INK, fontsize=10, fontweight="bold", ha="right")
    fig.tight_layout(h_pad=2.2)
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
