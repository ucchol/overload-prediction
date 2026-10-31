"""Fitted and held out predictions of human overload on the development data (exploratory).

Question. On the file and laptop displays, how well does each model predict, display by display, how often people
missed the best option and how much they regretted their choice, and does that prediction track how overloaded
people felt?

Why it matters. Only these data hold felt overload. The open datasets show overload in choice quality, so this is the
bridge between what the models predict and what people report.

How. Every model of the AAMAS comparison and the search model, one sensitivity each, with the designers' rubric
weights, and mixed logit (MXL, CL with a log normal sensitivity across people, two parameters, predicted by its
population average). Fitted, the parameters fitted to the task's six displays. Held out, each display predicted from
the fit to the other five. For every display, the model's probability of missing the best option (one minus its
probability of the options with the top rubric score) and its expected regret (the rubric shortfall of every option
weighted by its probabilities).
Against people, per task:
- the change from 8 to 32 options (mean of the two information levels), people's and each model's,
- the mean absolute error over the six displays, in misses and in regret points,
- the rank correlation over the six displays of the model's forecast with people's misses, regret and felt overload,
- choice accuracy, bits per choice above a uniform guess, mean over the six displays.

Uncertainty. 1000 participant resamples, the engine's resampling (seeds 10000 + b), every model refitted.

Usage
    python -m overload.experiments.overload_dev --data PATH_TO_CSV --out results/overload_dev --jobs 2
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .. import mixed_logit
from ..bootstrap import resample_indices, summarize
from ..cdm import fit_softmax
from ..data import TASKS, load_study
from ..engine import precompute_bases, provenance, run_replicate, write_json
from ..fit import _predict
from ..metrics import log_score
from ..responses import load_responses
from .process_dev import models

ENGINE_MODELS = ("QR", "CL", "DbS_avg", "MLBA_avg", "EBA", "SR")
MODELS = (*ENGINE_MODELS, "MXL")
MODES = ("fitted", "held_out")
_STATE: dict = {}


def _people(task: str, persons: np.ndarray | None) -> dict:
    R = _STATE["responses"][task]
    over, regret = R["overload"], R["regret"]
    if persons is not None:
        over, regret = over[persons], regret[persons]
    return {"overload": over.mean(0).tolist(), "regret": regret.mean(0).tolist(), "miss": (regret > 0).mean(0).tolist()}


def _mxl_rows(task_name: str) -> dict:
    """Every choice of the task as a row: the CL scores of its display (padded), the option chosen, the person."""
    task, base = _STATE["tasks"][task_name], _STATE["bases"][task_name]["CL"]
    width = max(b.size for b in base)
    S = np.full((len(base), width), np.nan)
    for j, b in enumerate(base):
        S[j, : b.size] = b
    trial, choice, person = [], [], []
    for r, obs in enumerate(task.panel):
        for j, o in obs:
            trial.append(j)
            choice.append(o)
            person.append(r)
    return {"S": S, "trial": np.array(trial), "choice": np.array(choice), "person": np.array(person)}


def _mxl(task_name: str, persons: np.ndarray | None) -> dict:
    """Mixed logit fitted to all six displays and to each five, population average predictions per display."""
    R = _STATE["mxl"][task_name]
    P = len(_STATE["tasks"][task_name].panel)
    mult = np.ones(P) if persons is None else np.bincount(persons, minlength=P).astype(float)
    w = mult[R["person"]]
    S = R["S"][R["trial"]]
    out = {}
    for key, drop in [("fitted", None), *[(j, j) for j in range(R["S"].shape[0])]]:
        ww = w if drop is None else np.where(R["trial"] == drop, 0.0, w)
        start = fit_softmax(S[ww > 0], R["choice"][ww > 0], ww[ww > 0])
        out[key] = mixed_logit.fit_mxl(S, R["choice"], R["person"], ww, start=start)
    return out


def _predictions(task_name: str, rec: dict, persons: np.ndarray | None) -> dict:
    task, bases = _STATE["tasks"][task_name], _STATE["bases"][task_name]
    by_name = {m.name: m for m in _STATE["models"]}
    counts = task.counts(persons)
    mxl = _mxl(task_name, persons)
    S = _STATE["mxl"][task_name]["S"]
    out: dict = {}
    for mode in MODES:
        out[mode] = {}
        for name in MODELS:
            miss, regret, bits = [], [], []
            for j, trial in enumerate(task.trials):
                if name == "MXL":
                    mu, sigma = mxl["fitted"] if mode == "fitted" else mxl[j]
                    q = mixed_logit.predict_mxl(mu, sigma, S[j : j + 1])[0, : trial.U.size]
                    bits.append(log_score(counts[j], q))
                else:
                    theta = rec[name]["theta"] if mode == "fitted" else rec[name]["theta_ho"][j]
                    q = _predict(by_name[name], theta, bases[name][j])
                    bits.append(rec[name]["fit" if mode == "fitted" else "ho"][j])
                best = trial.U.max() == trial.U
                miss.append(float(1.0 - q[best].sum()))
                regret.append(float(q @ (trial.U.max() - trial.U)))
            out[mode][name] = {"miss": miss, "regret": regret, "bits": bits}
    out["mxl_params"] = {str(k): list(v) for k, v in mxl.items()}
    return out


def replicate(b: int) -> dict:
    rec = run_replicate(_STATE["tasks"], _STATE["models"], _STATE["bases"], b)
    idx = resample_indices(_STATE["tasks"], b)
    return {"b": b, **{t: {"people": _people(t, idx[t]), "models": _predictions(t, rec[t], idx[t])} for t in TASKS}}


def _worker(bs: Sequence[int]) -> list[dict]:
    return [replicate(b) for b in bs]


def _growth(x: Sequence[float]) -> float:
    """32 minus 8 options, mean of the two information levels (displays 8L, 8H, 16L, 16H, 32L, 32H)."""
    return (x[4] + x[5] - x[0] - x[1]) / 2.0


def _rank(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def measures(rep: dict) -> dict:
    out = {}
    for t in TASKS:
        P = rep[t]["people"]
        for k in ("overload", "regret", "miss"):
            out[f"{t}|people|growth|{k}"] = _growth(P[k])
        for mode in MODES:
            for name in MODELS:
                F = rep[t]["models"][mode][name]
                for k in ("miss", "regret"):
                    key = f"{t}|{mode}|{name}|{k}"
                    out[f"{key}|growth"] = _growth(F[k])
                    out[f"{key}|growth_error"] = _growth(F[k]) - _growth(P[k])
                    out[f"{key}|mae"] = float(np.mean(np.abs(np.array(F[k]) - np.array(P[k]))))
                    for y in ("overload", "regret", "miss"):
                        out[f"{key}|rank_with_{y}"] = _rank(F[k], P[y])
                out[f"{t}|{mode}|{name}|bits"] = float(np.mean(F["bits"]))
                if name != "SR":
                    out[f"{t}|{mode}|SR_minus_{name}|bits"] = float(
                        np.mean(rep[t]["models"][mode]["SR"]["bits"])
                    ) - float(np.mean(F["bits"]))
    return out


def run(csv: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    tasks = load_study(csv)
    ms = [m for m in models() if m.name in ENGINE_MODELS]
    bases = precompute_bases(tasks, ms)
    R = load_responses(csv)
    responses = {}
    for task in TASKS:
        S = R[R["task"] == task].sort_values(["person", "trial"])
        n_p = S["person"].nunique()
        responses[task] = {
            "overload": S["overload"].to_numpy(float).reshape(n_p, 6),
            "regret": S["regret"].to_numpy(float).reshape(n_p, 6),
        }
    _STATE.update(tasks=tasks, models=ms, bases=bases, responses=responses)
    _STATE["mxl"] = {t: _mxl_rows(t) for t in TASKS}
    t0 = time.time()
    point = replicate(-1)
    order = list(range(replicates))
    if jobs > 1 and replicates:
        size = max(1, replicates // (4 * jobs))
        groups = [order[i : i + size] for i in range(0, len(order), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            reps = [r for c in pool.map(_worker, groups) for r in c]
    else:
        reps = _worker(order)
    reps.sort(key=lambda r: r["b"])
    pm, rm = measures(point), [measures(r) for r in reps]
    summary = {"measures": {}, "point": {t: point[t] for t in TASKS}}
    for k, v in pm.items():
        vals = [r[k] for r in rm]
        if np.isfinite(v) and np.all(np.isfinite(vals)):
            summary["measures"][k] = summarize(v, vals)
        else:
            summary["measures"][k] = {"point": v, "percentile": [float("nan")] * 2, "clear": False}
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    (out / "table_overload_dev.md").write_text(table(summary))
    return summary


def _f(e: dict, d: int = 2) -> str:
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{' *' if e.get('clear') else ''}"


def table(s: dict) -> str:
    M, Pt = s["measures"], s["point"]
    names = {"file": "File selection", "laptop": "Laptop purchase"}
    L = ["# Fitted and held out predictions of human overload (development data, exploratory)", ""]
    for t in TASKS:
        P = Pt[t]["people"]
        L += [
            f"## {names[t]}",
            "",
            "Displays 8 brief, 8 full, 16 brief, 16 full, 32 brief, 32 full.",
            "",
            f"- People missed the best: {', '.join(f'{x:.0%}' for x in P['miss'])}. Regret: "
            f"{', '.join(f'{x:.1f}' for x in P['regret'])}. Felt overload: "
            f"{', '.join(f'{x:.2f}' for x in P['overload'])}.",
            f"- From 8 to 32 options, people: missed {_f(M[f'{t}|people|growth|miss'])}, regret "
            f"{_f(M[f'{t}|people|growth|regret'], 1)} points, felt overload "
            f"{_f(M[f'{t}|people|growth|overload'])}.",
            "",
            "| Model | Prediction | Missed the best by display | 8 to 32, minus people's | Mean error | "
            "Rank with misses | Rank with regret | Rank with felt overload | Bits per choice |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name in MODELS:
            for mode in MODES:
                F = Pt[t]["models"][mode][name]["miss"]
                key = f"{t}|{mode}|{name}|miss"
                L.append(
                    f"| {name} | {mode.replace('_', ' ')} | {', '.join(f'{x:.0%}' for x in F)} | "
                    f"{_f(M[key + '|growth_error'])} | {M[key + '|mae']['point']:.3f} | "
                    f"{_f(M[key + '|rank_with_miss'])} | {_f(M[key + '|rank_with_regret'])} | "
                    f"{_f(M[key + '|rank_with_overload'])} | {_f(M[f'{t}|{mode}|{name}|bits'], 3)} |"
                )
        L.append("")
    L += [
        "Brackets give 95% percentile intervals from 1000 participant resamples. Six displays per task, so rank "
        "correlations are coarse. Bits per choice above a uniform guess, mean of the six displays. MXL is mixed logit, "
        "CL with a log normal sensitivity across people, predicted by its population average.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/overload_dev"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    run(a.data, a.out, a.replicates, a.jobs)
    print((a.out / "table_overload_dev.md").read_text())


if __name__ == "__main__":
    main()
