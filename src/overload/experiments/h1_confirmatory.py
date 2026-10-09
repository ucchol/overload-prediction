"""Confirmatory test of H1 on the open snack choice data, ANALYSIS_PLAN.md Amendment 3.

C1  narrowing across screen positions, the concentration gap between observed choice shares per position and the
    held-out CL prediction, its mean over set sizes and its slope on log2 n, with the centered bootstrap.
C2  set size scaling, held-out CL at beta = +0.5 and -0.5 against beta = 0 over all items, Holm corrected.
C3  narrowing against noise, CL's sensitivity elasticity over all items and over the looked-at items.

Usage
    python -m overload.experiments.h1_confirmatory --snack data/snack --out results/h1_confirmatory --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import holm, summarize
from ..engine import provenance, write_json
from ..metrics import effective_number, sampled_effective_number
from ..models import BETAS
from ..snack import (
    FAMILIES,
    SET_SIZES,
    VERSIONS,
    SnackData,
    elasticity,
    held_out,
    included_rows,
    load_snack,
    size_alone,
)
from .h1_narrowing import _centered, _slope

SEED_OFFSET = 20000
TASK_INDEX = 2
DRAWS = 2000
REF = "CL|beta=+0.0|all"
PRIMARY = {"CL|beta=+0.5|all": +1, "CL|beta=-0.5|all": -1}

_STATE: dict = {}


def cell(family: str, beta: float, version: str) -> str:
    return f"{family}|beta={beta:+.1f}|{version}"


def draw(n_participants: int, b: int) -> np.ndarray | None:
    if b < 0:
        return None
    return np.random.default_rng(SEED_OFFSET + b).integers(0, n_participants, size=n_participants)


def run_replicate(data: SnackData, b: int) -> dict:
    """Every measure of Amendment 3 for replicate b (b = -1 is the original sample)."""
    w = data.weights(draw(len(data.participants), b))
    rows = {v: included_rows(data, v) for v in VERSIONS}
    rec: dict = {"b": b, "cells": {}}
    q_ref = None
    for version in VERSIONS:
        for family in FAMILIES:
            for beta in BETAS:
                ho = held_out(data, w, family, beta, version, rows[version], want_q=cell(family, beta, version) == REF)
                rec["cells"][cell(family, beta, version)] = {
                    "L": ho["L"],
                    "by_size": ho["by_size"],
                    "lambda": ho["lambda"],
                }
                if cell(family, beta, version) == REF:
                    q_ref = ho["q"]

    # C1, choices across positions against the held-out CL prediction
    E_obs, E_samp_q, E_samp_p = [], [], []
    for k, block in enumerate(data.blocks):
        wk = w[k]
        n_choices = int(round(float(wk.sum())))
        p = np.bincount(block.choice, weights=wk, minlength=block.n) / wk.sum()
        q = (wk[:, None] * q_ref[k]).sum(axis=0) / wk.sum()
        E_obs.append(effective_number(p))
        E_samp_q.append(sampled_effective_number(q, n_choices, DRAWS, (1, TASK_INDEX, k, b + 1)))
        if b < 0:
            E_samp_p.append(sampled_effective_number(p, n_choices, DRAWS, (2, TASK_INDEX, k, 0)))
    rec["C1"] = {"E_obs": E_obs, "E_samp_q": E_samp_q}  # no trial or participant counts are recorded
    if b < 0:
        rec["C1"]["E_samp_p"] = E_samp_p
        rec["C1"]["p"] = [
            (np.bincount(blk.choice, weights=w[k], minlength=blk.n) / w[k].sum()).tolist()
            for k, blk in enumerate(data.blocks)
        ]

    # C3, sensitivity elasticity on the trials whose chosen item was looked at
    beta_all, lam_all = elasticity(data, w, "all", rows["seen"])
    beta_seen, lam_seen = elasticity(data, w, "seen", rows["seen"])
    rec["C3"] = {"beta_all": beta_all, "lambda_all": lam_all, "beta_seen": beta_seen, "lambda_seen": lam_seen}

    # Descriptives
    rec["lambda_alone"] = {
        "all_items_all_trials": size_alone(data, w, "all", rows["all"]),
        "all_items_C3_trials": size_alone(data, w, "all", rows["seen"]),
        "seen_items_C3_trials": size_alone(data, w, "seen", rows["seen"]),
    }
    rec["share_looked_at"] = [
        float(w[k] @ (blk.seen.sum(axis=1) / blk.n) / w[k].sum()) for k, blk in enumerate(data.blocks)
    ]
    rec["share_chosen_looked_at"] = [float(w[k] @ blk.chosen_seen / w[k].sum()) for k, blk in enumerate(data.blocks)]
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(_STATE["data"], b) for b in bs]


def run(folder: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    data = load_snack(folder)
    _STATE["data"] = data
    t0 = time.time()
    point = run_replicate(data, -1)
    order = list(range(replicates))
    if jobs > 1 and replicates:
        size = max(1, replicates // (4 * jobs))
        groups = [order[i : i + size] for i in range(0, len(order), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            reps = [r for chunk in pool.map(_worker, groups) for r in chunk]
    else:
        reps = _worker(order)
    reps.sort(key=lambda r: r["b"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "point.json", point)
    summary = summarize_confirmatory(point, reps)
    summary["provenance"] = provenance(
        {
            "replicates": replicates,
            "seconds": round(time.time() - t0, 1),
            "status": "confirmatory, Amendment 3",
            "snack_commit": (folder / "COMMIT").read_text().strip() if (folder / "COMMIT").exists() else "unknown",
        }
    )
    write_json(out / "summary.json", summary)
    write_tables(summary, out)
    return summary


# ----------------------------------------------------------------------------- summaries
def _pct(dl: float) -> float:
    """Percent change in the likelihood per choice for a log score difference in bits."""
    return 100.0 * (2.0**dl - 1.0)


def _with_percent(entry: dict) -> dict:
    out = dict(entry)
    out["percent"] = {
        "point": _pct(entry["point"]),
        "percentile": [_pct(v) for v in entry["percentile"]],
        "basic": [_pct(v) for v in entry["basic"]],
    }
    return out


def summarize_c2(point: dict, reps: list[dict]) -> dict:
    cells: dict = {}
    for version in VERSIONS:
        for family in FAMILIES:
            base = cell(family, 0.0, version)
            for beta in BETAS:
                m = cell(family, beta, version)
                c = point["cells"][m]
                entry = {
                    "L": summarize(c["L"], [r["cells"][m]["L"] for r in reps]),
                    "times_chance": 2.0 ** c["L"],
                    "times_chance_by_size": {str(n): 2.0**v for n, v in zip(SET_SIZES, c["by_size"], strict=True)},
                    "lambda_held_out": c["lambda"],
                }
                if beta != 0.0:
                    entry["minus_beta0"] = _with_percent(
                        summarize(
                            c["L"] - point["cells"][base]["L"],
                            [r["cells"][m]["L"] - r["cells"][base]["L"] for r in reps],
                        )
                    )
                if family != "CL":
                    ref = cell("CL", beta, version)
                    entry["minus_CL"] = _with_percent(
                        summarize(
                            c["L"] - point["cells"][ref]["L"],
                            [r["cells"][m]["L"] - r["cells"][ref]["L"] for r in reps],
                        )
                    )
                cells[m] = entry
    adjusted = holm({m: cells[m]["minus_beta0"]["p"] for m in PRIMARY})
    primary = {}
    for m, sign in PRIMARY.items():
        d = cells[m]["minus_beta0"]
        clear = bool(d["clear"] and sign * d["point"] > 0)
        primary[m] = {
            "predicted_sign": sign,
            "clear_in_predicted_direction": clear,
            "p": d["p"],
            "p_holm": adjusted[m],
            "supports_H1": bool(clear and adjusted[m] < 0.05),
        }
    return {"cells": cells, "primary": primary, "H1_supported_C2": all(v["supports_H1"] for v in primary.values())}


def summarize_c1(point: dict, reps: list[dict]) -> dict:
    n = np.asarray(SET_SIZES, float)
    x = np.log2(n)
    pc = point["C1"]
    G = np.asarray(pc["E_obs"]) / n - np.asarray(pc["E_samp_q"]) / n
    G0 = np.asarray(pc["E_samp_p"]) / n - np.asarray(pc["E_samp_q"]) / n
    errs = np.array([np.asarray(r["C1"]["E_obs"]) / n - np.asarray(r["C1"]["E_samp_q"]) / n - G0 for r in reps])
    mean = _centered(float(G.mean()), errs.mean(axis=1))
    slope = _centered(_slope(x, G), np.array([_slope(x, e) for e in errs]))
    per_size = [
        {
            "set_size": int(n[k]),
            "observed_effective_share": float(pc["E_obs"][k] / n[k]),
            "model_effective_share": float(pc["E_samp_q"][k] / n[k]),
            "gap": float(G[k]),
        }
        for k in range(len(SET_SIZES))
    ]
    supported = bool(mean["clear"] and mean["point"] < 0 and slope["clear"] and slope["point"] < 0)
    return {"mean_gap": mean, "slope_gap_per_doubling": slope, "per_size": per_size, "H1_supported_C1": supported}


def summarize_c3(point: dict, reps: list[dict]) -> dict:
    p3 = point["C3"]
    out = {
        "beta_all": summarize(p3["beta_all"], [r["C3"]["beta_all"] for r in reps]),
        "beta_seen": summarize(p3["beta_seen"], [r["C3"]["beta_seen"] for r in reps]),
        "beta_all_minus_seen": summarize(
            p3["beta_all"] - p3["beta_seen"], [r["C3"]["beta_all"] - r["C3"]["beta_seen"] for r in reps]
        ),
        "lambda_all": p3["lambda_all"],
        "lambda_seen": p3["lambda_seen"],
    }
    d, s = out["beta_all_minus_seen"], out["beta_seen"]
    out["narrowing_supported"] = bool(d["clear"] and d["point"] < 0)
    out["noise_supported"] = bool(s["clear"] and s["point"] < 0)
    out["at_bound"] = {k: bool(abs(p3[k]) >= 3.0 - 1e-6) for k in ("beta_all", "beta_seen")}
    return out


def summarize_descriptives(point: dict, reps: list[dict]) -> dict:
    out: dict = {"set_sizes": list(SET_SIZES)}
    for key in ("share_looked_at", "share_chosen_looked_at"):
        out[key] = [summarize(point[key][k], [r[key][k] for r in reps]) for k in range(len(SET_SIZES))]
    out["lambda_alone"] = {
        name: [summarize(vals[k], [r["lambda_alone"][name][k] for r in reps]) for k in range(len(SET_SIZES))]
        for name, vals in point["lambda_alone"].items()
    }
    return out


def summarize_confirmatory(point: dict, reps: list[dict]) -> dict:
    return {
        "C1": summarize_c1(point, reps),
        "C2": summarize_c2(point, reps),
        "C3": summarize_c3(point, reps),
        "descriptives": summarize_descriptives(point, reps),
    }


# ----------------------------------------------------------------------------- tables
def _f(e: dict, scale: float = 1.0, digits: int = 1) -> str:
    s = f"{e['point'] * scale:+.{digits}f}"
    lo, hi = (v * scale for v in e["percentile"])
    return s + f" [{lo:.{digits}f}, {hi:.{digits}f}]" + (" *" if e.get("clear") else "")


def _pf(e: dict) -> str:
    pc = e["percent"]
    return f"{pc['point']:+.1f}% [{pc['percentile'][0]:.1f}, {pc['percentile'][1]:.1f}]" + (" *" if e["clear"] else "")


def write_tables(summary: dict, out: Path) -> None:
    lines = ["# H1 confirmatory, open snack choice data (Amendment 3)", ""]
    c1 = summary["C1"]
    lines += [
        "## C1, choices across screen positions",
        "",
        "| Items | Observed | CL held out | Gap |",
        "|---|---|---|---|",
    ]
    for row in c1["per_size"]:
        lines.append(
            f"| {row['set_size']} | {100 * row['observed_effective_share']:.1f} | "
            f"{100 * row['model_effective_share']:.1f} | {100 * row['gap']:+.1f} |"
        )
    lines += [
        "",
        f"- Mean gap {_f(c1['mean_gap'], 100.0)} points, slope per doubling {_f(c1['slope_gap_per_doubling'], 100.0)}"
        f" points, supports H1 {c1['H1_supported_C1']}.",
        "",
        "## C2, set size scaling, held-out likelihood relative to chance",
        "",
        "| Model | Version | beta | Times chance | 9 | 16 | 25 | 36 | Minus beta 0 | Minus CL |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    c2 = summary["C2"]
    for version in VERSIONS:
        for family in FAMILIES:
            for beta in BETAS:
                e = c2["cells"][cell(family, beta, version)]
                by = e["times_chance_by_size"]
                lines.append(
                    f"| {family} | {version} | {beta:+.1f} | {e['times_chance']:.2f} | "
                    + " | ".join(f"{by[str(n)]:.2f}" for n in SET_SIZES)
                    + f" | {_pf(e['minus_beta0']) if 'minus_beta0' in e else ''} | "
                    f"{_pf(e['minus_CL']) if 'minus_CL' in e else ''} |"
                )
    lines += ["", "Primary tests:", ""]
    for m, d in c2["primary"].items():
        lines.append(
            f"- {m} minus beta 0, predicted {'positive' if d['predicted_sign'] > 0 else 'negative'}. Clear in that "
            f"direction {d['clear_in_predicted_direction']}, Holm p {d['p_holm']:.3f}, supports H1 {d['supports_H1']}."
        )
    lines += [f"- C2 supports H1: {c2['H1_supported_C2']}", ""]
    c3 = summary["C3"]
    lines += [
        "## C3, narrowing against noise, CL's sensitivity elasticity",
        "",
        f"- All items {_f(c3['beta_all'], 1.0, 3)}, looked-at items {_f(c3['beta_seen'], 1.0, 3)}, "
        f"difference {_f(c3['beta_all_minus_seen'], 1.0, 3)}.",
        f"- Narrowing supported {c3['narrowing_supported']}, noise supported {c3['noise_supported']}.",
        "",
        "## Descriptives",
        "",
        "| Items | Share looked at | Chosen item looked at | CL sensitivity, all items | Looked-at items |",
        "|---|---|---|---|---|",
    ]
    d = summary["descriptives"]
    for k, n in enumerate(SET_SIZES):
        la = d["lambda_alone"]["all_items_C3_trials"][k]
        ls = d["lambda_alone"]["seen_items_C3_trials"][k]
        lines.append(
            f"| {n} | {100 * d['share_looked_at'][k]['point']:.1f}"
            f" | {100 * d['share_chosen_looked_at'][k]['point']:.1f}"
            f" | {la['point']:.3f} [{la['percentile'][0]:.3f}, {la['percentile'][1]:.3f}]"
            f" | {ls['point']:.3f} [{ls['percentile'][0]:.3f}, {ls['percentile'][1]:.3f}] |"
        )
    lines += [
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples, and a star marks a value whose "
        "percentile and basic intervals both exclude zero. C1 intervals are centered as in Amendment 2. Differences "
        "in held-out log score are shown as percent changes in the likelihood of the chosen item.",
    ]
    (out / "table_h1_confirmatory.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snack", type=Path, default=Path("data/snack"))
    ap.add_argument("--out", type=Path, default=Path("results/h1_confirmatory"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.snack, args.out, args.replicates, args.jobs)
    print(f"written to {args.out} in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
