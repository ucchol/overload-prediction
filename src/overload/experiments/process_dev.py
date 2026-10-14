"""How people decide, in the development data. Choices, decision time and overload, model by model (exploratory).

Three questions, answered with the same fits.

1. Choices. Does the proposed search model, sample until return, predict unseen trials as well as the
   established models? Every model has one free parameter and fixed rubric weights, and each trial is
   predicted from a fit to the other five trials of its task, as in R0. The comparisons that matter hold the
   values fixed and change only the process (SR against CL, SR_DbS against DbS_avg, SR_MLBA against
   MLBA_avg), then ask whether the search model is as accurate as the best established model. The rejected
   alternative, accept in reading order, is reported beside them.

2. Time. Does the work each model's algorithm needs explain how long people took? A model's effort is the
   number of elementary steps its algorithm needs on a trial (``overload.effort``). Its time law, a start up
   time plus a time per step, both nonnegative, is fitted to people's typical times by least squares.
   - In sample, on all six trials of a task, the fit shows how much of people's time the model's work can
     carry and what time per step that implies.
   - Held out, the constants are fitted on five trials and the sixth trial's time is predicted. The typical
     error is the geometric mean of predicted over observed time, as a percent. A model whose effort does not
     track people's time does no better than a constant.
   People's time divided by a model's effort (``per_step`` in the summary) is recorded but not used. It is not
   a fit, since it assumes that every second goes to the model's steps, while the fitted laws put most of
   people's time into the start up constant.

3. Premonition. Before anyone chooses, can a model tell which trials will overload people, and on which they
   will choose poorly? Each forecast is computed from the trial's options and a fit to the other trials only,
   never from the trial's own choices or ratings.
   - Feelings. Forecasts are the number of options, the attributes used, each model's predicted spread of
     choices (entropy, in bits), each model's effort and the search model's expected number of looks. A
     forecast is judged by how well it orders the trials by their overload index (Spearman correlation) and by
     the error of a held out linear calibration, fitted on the other trials and applied to the left out one,
     against a forecast that ignores the trial. This is done within each task (six trials) and over both
     tasks together (twelve trials, one calibration line), where the number of options cannot tell a file
     display from a laptop display of the same size.
   - Choice quality. Each model's expected regret, the rubric shortfall of every option weighted by the
     model's held out choice probabilities, is compared directly with people's mean regret, in regret
     points and without any calibration.

Typical time is the geometric mean of the seconds on the choice page. The overload index is from
``overload.responses``. Uncertainty comes from 1000 participant resamples drawn as in R0 (seed 10000 + b, file
participants before laptop participants), with every model refitted in every replicate. Development data,
so everything here is exploratory.

Usage
    python -m overload.experiments.process_dev --data PATH_TO_CSV --out results/process_dev --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

from ..bootstrap import resample_indices, summarize
from ..data import TASKS, load_study
from ..effort import effort, used_attributes
from ..engine import precompute_bases, provenance, run_replicate, subset_completeness, write_json
from ..fit import _predict
from ..metrics import entropy_bits
from ..models import Model, baseline_models, screening_models, search_models
from ..responses import load_responses
from ..search import sample_until_return_length

FIXATION_SECONDS = 0.25
TIME_MODELS = ("QR", "CL", "DbS_avg", "MLBA_avg", "EBA", "SR", "SR_DbS", "Accept_read")
ENTROPY_MODELS = ("QR", "CL", "DbS_avg", "MLBA_avg", "SR", "SR_DbS")
REGRET_MODELS = ("QR", "CL", "DbS_avg", "MLBA_avg", "EBA", "SR", "SR_DbS")
SCOPES = ("file", "laptop", "both")
PROCESS_PAIRS = (("SR", "CL"), ("SR_DbS", "DbS_avg"), ("SR_MLBA", "MLBA_avg"), ("Accept_read", "CL"))
BEST_ESTABLISHED = ("CL", "DbS_avg", "MLBA_avg")

_STATE: dict = {}


def models() -> list[Model]:
    return baseline_models() + [m for m in screening_models() if m.name == "EBA"] + search_models()


# ----------------------------------------------------------------------------- one replicate
def _people(task: str, persons: np.ndarray | None) -> dict:
    """Typical seconds, mean overload index and mean regret per trial for the (resampled) participants."""
    R = _STATE["responses"]
    logt, over, regret = R[task]["logtime"], R[task]["overload"], R[task]["regret"]
    if persons is not None:
        logt, over, regret = logt[persons], over[persons], regret[persons]
    return {
        "seconds": np.exp(logt.mean(axis=0)).tolist(),
        "overload": over.mean(axis=0).tolist(),
        "regret": regret.mean(axis=0).tolist(),
    }


def _forecasts(task_name: str, rec: dict) -> dict:
    """Effort, predicted entropy and expected looks of every model on every trial, from held out fits."""
    task, bases = _STATE["tasks"][task_name], _STATE["bases"][task_name]
    by_name = {m.name: m for m in _STATE["models"]}
    out: dict = {"effort": {}, "entropy": {}}
    for name in TIME_MODELS:
        e = []
        for j, trial in enumerate(task.trials):
            lam = rec[name]["theta_ho"][j]
            values = bases[name][j] if by_name[name].kind in ("race", "accept") else None
            e.append(effort(name, trial, task.shares, lam, values))
        out["effort"][name] = e
    for name in ENTROPY_MODELS:
        out["entropy"][name] = [
            entropy_bits(_predict(by_name[name], rec[name]["theta_ho"][j], bases[name][j]))
            for j in range(len(task.trials))
        ]
    out["looks"] = [
        sample_until_return_length(rec["SR"]["theta_ho"][j], bases["SR"][j]) for j in range(len(task.trials))
    ]
    out["regret"] = {
        name: [
            float(_predict(by_name[name], rec[name]["theta_ho"][j], bases[name][j]) @ (100.0 - trial.U))
            for j, trial in enumerate(task.trials)
        ]
        for name in REGRET_MODELS
    }
    return out


def replicate(b: int) -> dict:
    rec = run_replicate(_STATE["tasks"], _STATE["models"], _STATE["bases"], b)
    idx = resample_indices(_STATE["tasks"], b)
    for name in TASKS:
        rec[name]["people"] = _people(name, idx[name])
        rec[name]["forecast"] = _forecasts(name, rec[name])
    return rec


def _worker(bs: Sequence[int]) -> list[dict]:
    return [replicate(b) for b in bs]


# ----------------------------------------------------------------------------- summaries
def _completeness(rec: dict, task: str, model: str, bench: list[float]) -> float:
    return subset_completeness(rec[task], model, "ho", range(6), bench)


def choice_summary(point: dict, reps: list[dict]) -> dict:
    out: dict = {}
    names = [m.name for m in _STATE["models"]]
    for task in TASKS:
        bench = point[task]["Lstar"]
        K = {m: _completeness(point, task, m, bench) for m in names}
        KB = {m: np.array([_completeness(r, task, m, bench) for r in reps]) for m in names}
        block = {"held_out": {m: summarize(K[m], KB[m]) for m in names}, "process": {}, "against_best": {}}
        for a, c in PROCESS_PAIRS:
            block["process"][f"{a}-{c}"] = summarize(K[a] - K[c], KB[a] - KB[c])
        for proposed in ("SR", "SR_DbS"):
            for est in BEST_ESTABLISHED:
                block["against_best"][f"{proposed}-{est}"] = summarize(K[proposed] - K[est], KB[proposed] - KB[est])
        out[task] = block
    return out


def _nnls_time(effort_values: np.ndarray, seconds: np.ndarray) -> tuple[float, float]:
    """Nonnegative start up time and time per step, least squares on seconds."""
    A = np.column_stack([np.ones_like(effort_values), effort_values])
    coef, _ = nnls(A, seconds)
    return float(coef[0]), float(coef[1])


def fitted_time(effort_values: Sequence[float], seconds: Sequence[float]) -> dict[str, float]:
    """Start up time and time per step fitted to every trial, and the share of the variation in seconds explained."""
    e, s = np.asarray(effort_values, float), np.asarray(seconds, float)
    t0, tau = _nnls_time(e, s)
    resid = s - t0 - tau * e
    return {
        "start_seconds": t0,
        "step_ms": 1000.0 * tau,
        "r2": 1.0 - float(resid @ resid) / float(((s - s.mean()) ** 2).sum()),
    }


def held_out_time(effort_values: Sequence[float], seconds: Sequence[float]) -> tuple[list[float], list[float]]:
    """Predicted seconds of every trial from the constants fitted on the other trials, and log errors."""
    e, s = np.asarray(effort_values, float), np.asarray(seconds, float)
    pred, err = [], []
    for j in range(e.size):
        rest = np.arange(e.size) != j
        t0, tau = _nnls_time(e[rest], s[rest])
        p = t0 + tau * e[j]
        pred.append(float(p))
        err.append(float(abs(np.log(p / s[j]))) if p > 0 else float("inf"))
    return pred, err


def _constant_time(seconds: Sequence[float]) -> tuple[list[float], list[float]]:
    s = np.asarray(seconds, float)
    pred = [float(np.mean(np.delete(s, j))) for j in range(s.size)]
    return pred, [float(abs(np.log(p / x))) for p, x in zip(pred, s, strict=True)]


def _percent(log_error: float) -> float:
    return 100.0 * (np.exp(log_error) - 1.0)


def time_summary(point: dict, reps: list[dict]) -> dict:
    out: dict = {}
    for task in TASKS:
        block: dict = {"seconds": point[task]["people"]["seconds"], "per_step": {}, "held_out": {}, "error": {}}
        sec_b = np.array([r[task]["people"]["seconds"] for r in reps])
        for name in TIME_MODELS:
            e = np.array(point[task]["forecast"]["effort"][name])
            e_b = np.array([r[task]["forecast"]["effort"][name] for r in reps])
            per = np.asarray(point[task]["people"]["seconds"]) / e
            per_b = sec_b / e_b
            block["per_step"][name] = {
                "effort": e.tolist(),
                "seconds_per_step": per.tolist(),
                "percentile": np.percentile(per_b, [2.5, 97.5], axis=0).tolist(),
            }
            pred, err = held_out_time(e, point[task]["people"]["seconds"])
            errs_b = np.array([np.mean(held_out_time(eb, sb)[1]) for eb, sb in zip(e_b, sec_b, strict=True)])
            block["held_out"][name] = pred
            block["error"][name] = summarize(_percent(np.mean(err)), [_percent(x) for x in errs_b])
            block["error"][name]["log_point"] = float(np.mean(err))
            block["error"][name]["_log_reps"] = errs_b.tolist()
        pred, err = _constant_time(point[task]["people"]["seconds"])
        errs_b = np.array([np.mean(_constant_time(sb)[1]) for sb in sec_b])
        block["held_out"]["constant"] = pred
        block["error"]["constant"] = summarize(_percent(np.mean(err)), [_percent(x) for x in errs_b])
        block["error"]["constant"]["log_point"] = float(np.mean(err))
        block["error"]["constant"]["_log_reps"] = errs_b.tolist()
        block["error_minus_constant"] = {}
        for name in TIME_MODELS:
            d = block["error"][name]["log_point"] - block["error"]["constant"]["log_point"]
            db = np.array(block["error"][name]["_log_reps"]) - np.array(block["error"]["constant"]["_log_reps"])
            block["error_minus_constant"][name] = summarize(d, db)
        for entry in block["error"].values():
            entry.pop("_log_reps")
        out[task] = block
    return out


def _forecast_table(rec: dict, task: str) -> dict[str, list[float]]:
    task_obj = _STATE["tasks"][task]
    f = rec[task]["forecast"]
    table = {
        "options": [float(t.n) for t in task_obj.trials],
        "attributes": [float(used_attributes(t, task_obj.shares)) for t in task_obj.trials],
        "looks_SR": f["looks"],
    }
    table["cells"] = [a * b for a, b in zip(table["options"], table["attributes"], strict=True)]
    for name, v in f["entropy"].items():
        table[f"entropy_{name}"] = v
    for name in ("CL", "DbS_avg", "EBA", "SR"):
        table[f"effort_{name}"] = f["effort"][name]
    return table


def _calibrated_error(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error of a held out linear calibration y = a + b x, fitted on the other trials."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    errs = []
    for j in range(x.size):
        rest = np.arange(x.size) != j
        if np.ptp(x[rest]) == 0:
            pred = y[rest].mean()
        else:
            b, a = np.polyfit(x[rest], y[rest], 1)
            pred = a + b * x[j]
        errs.append(abs(pred - y[j]))
    return float(np.mean(errs))


def _constant_error(y: Sequence[float]) -> float:
    y = np.asarray(y, float)
    return float(np.mean([abs(np.delete(y, j).mean() - y[j]) for j in range(y.size)]))


def _scope_tables(rec: dict, scope: str) -> tuple[dict, dict, dict]:
    """Forecasts, observed outcomes and regret forecasts of the trials in a scope, files before laptops."""
    tasks = TASKS if scope == "both" else (scope,)
    F: dict[str, list[float]] = {}
    for t in tasks:
        for k, v in _forecast_table(rec, t).items():
            F.setdefault(k, []).extend(v)
    y = {name: [x for t in tasks for x in rec[t]["people"][name]] for name in ("overload", "regret")}
    R = {name: [x for t in tasks for x in rec[t]["forecast"]["regret"][name]] for name in REGRET_MODELS}
    return F, y, R


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return float(spearmanr(x, y).statistic) if np.ptp(x) > 0 and np.ptp(y) > 0 else 0.0


def _direct_error(forecast: Sequence[float], y: Sequence[float]) -> float:
    return float(np.mean(np.abs(np.asarray(forecast, float) - np.asarray(y, float))))


def premonition_summary(point: dict, reps: list[dict]) -> dict:
    """Forecasts of overload (feelings) and regret (choice quality), per task and over both tasks together.

    Overload forecasts are judged by their Spearman correlation with the overload index and by a held out linear
    calibration. Regret forecasts are each model's expected regret, the rubric shortfall of every option weighted
    by the model's held out choice probabilities, judged directly in regret points with no calibration.
    """
    out: dict = {}
    for scope in SCOPES:
        F, y, R = _scope_tables(point, scope)
        boot = [_scope_tables(r, scope) for r in reps]
        block: dict = {"overload": y["overload"], "regret": y["regret"], "forecasts": F, "regret_forecasts": R}
        yo, yo_b = y["overload"], [b[1]["overload"] for b in boot]
        base, base_b = _constant_error(yo), np.array([_constant_error(v) for v in yo_b])
        block["constant_error"] = summarize(base, base_b)
        block["by_forecast"] = {}
        for k in F:
            rho = _spearman(F[k], yo)
            rho_b = np.array([_spearman(b[0][k], v) for b, v in zip(boot, yo_b, strict=True)])
            err = _calibrated_error(F[k], yo)
            err_b = np.array([_calibrated_error(b[0][k], v) for b, v in zip(boot, yo_b, strict=True)])
            block["by_forecast"][k] = {
                "spearman": summarize(rho, rho_b),
                "error": summarize(err, err_b),
                "error_minus_constant": summarize(err - base, err_b - base_b),
            }
        yr, yr_b = y["regret"], [b[1]["regret"] for b in boot]
        rbase, rbase_b = _constant_error(yr), np.array([_constant_error(v) for v in yr_b])
        block["regret_constant_error"] = summarize(rbase, rbase_b)
        block["by_regret_forecast"] = {}
        for k in R:
            err = _direct_error(R[k], yr)
            err_b = np.array([_direct_error(b[2][k], v) for b, v in zip(boot, yr_b, strict=True)])
            rho = _spearman(R[k], yr)
            rho_b = np.array([_spearman(b[2][k], v) for b, v in zip(boot, yr_b, strict=True)])
            block["by_regret_forecast"][k] = {
                "spearman": summarize(rho, rho_b),
                "error": summarize(err, err_b),
                "error_minus_constant": summarize(err - rbase, err_b - rbase_b),
            }
        out[scope] = block
    return out


# ----------------------------------------------------------------------------- run
def run(csv: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    tasks = load_study(csv)
    ms = models()
    bases = precompute_bases(tasks, ms)
    R = load_responses(csv)
    responses = {}
    for task in TASKS:
        S = R[R["task"] == task].sort_values(["person", "trial"])
        n_p = S["person"].nunique()
        responses[task] = {
            "logtime": np.log(S["seconds"].to_numpy(float)).reshape(n_p, 6),
            "overload": S["overload"].to_numpy(float).reshape(n_p, 6),
            "regret": S["regret"].to_numpy(float).reshape(n_p, 6),
        }
        if n_p != len(tasks[task].panel):
            raise ValueError("the response table and the choice panel must hold the same participants")
    _STATE.update(tasks=tasks, models=ms, bases=bases, responses=responses)

    t0 = time.time()
    point = replicate(-1)
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
    summary = {
        "choices": choice_summary(point, reps),
        "time": time_summary(point, reps),
        "premonition": premonition_summary(point, reps),
        "trials": {t: list(tasks[t].keys) for t in TASKS},
        "provenance": provenance(
            {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
        ),
    }
    write_json(out / "summary.json", summary)
    (out / "tables.md").write_text(tables(summary))
    return summary


# ----------------------------------------------------------------------------- report
def _f(s: dict, digits: int = 1, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    lo, hi = s["percentile"]
    star = " *" if s.get("clear") else ""
    return f"{fmt.format(s['point'])} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def tables(summary: dict) -> str:
    L = ["# How people decide, development data (exploratory)", ""]
    C = summary["choices"]
    L += ["## 1. Choices, held out completeness (%)", "", "| Model | Files | Laptops |", "|---|---|---|"]
    for m in C["file"]["held_out"]:
        L.append(f"| {m} | {_f(C['file']['held_out'][m])} | {_f(C['laptop']['held_out'][m])} |")
    L += [
        "",
        "Same values, different process (points of completeness).",
        "",
        "| Contrast | Files | Laptops |",
        "|---|---|---|",
    ]
    for k in C["file"]["process"]:
        L.append(f"| {k} | {_f(C['file']['process'][k], sign=True)} | {_f(C['laptop']['process'][k], sign=True)} |")
    L += [
        "",
        "The search model against the established models (points).",
        "",
        "| Contrast | Files | Laptops |",
        "|---|---|---|",
    ]
    for k in C["file"]["against_best"]:
        L.append(
            f"| {k} | {_f(C['file']['against_best'][k], sign=True)} | {_f(C['laptop']['against_best'][k], sign=True)} |"
        )
    T = summary["time"]
    for task in TASKS:
        keys = summary["trials"][task]
        L += ["", f"## 2. Time, {task}", "", "Time law fitted to the typical seconds of all six trials.", ""]
        L += [
            "| Model | Effort, steps per trial | Start up time, s | Time per step, ms | Variation explained |",
            "|---|---|---|---|---|",
        ]
        L.append("| People, seconds | " + ", ".join(f"{s:.0f}" for s in T[task]["seconds"]) + " | | | |")
        for m, e in T[task]["per_step"].items():
            fit = fitted_time(e["effort"], T[task]["seconds"])
            effort = ", ".join(f"{n:.0f}" for n in e["effort"])
            L.append(f"| {m} | {effort} | {fit['start_seconds']:.1f} | {fit['step_ms']:.3g} | {fit['r2']:.2f} |")
        L += ["", "Held out predicted seconds, constants fitted on the other five trials.", ""]
        L += [
            "| Model | " + " | ".join(keys) + " | Typical error, % | Minus constant (log) |",
            "|---" * (len(keys) + 3) + "|",
        ]
        for m, p in T[task]["held_out"].items():
            extra = _f(T[task]["error_minus_constant"][m], 3, True) if m in T[task]["error_minus_constant"] else ""
            L.append(f"| {m} | " + " | ".join(f"{x:.0f}" for x in p) + f" | {_f(T[task]['error'][m])} | {extra} |")
    P = summary["premonition"]
    names = {"file": "files", "laptop": "laptops", "both": "both tasks together"}
    for scope in SCOPES:
        B = P[scope]
        L += ["", f"## 3. Premonition, {names[scope]}", ""]
        L.append(
            "Overload index by trial: "
            + ", ".join(f"{x:.2f}" for x in B["overload"])
            + f". Constant forecast error {_f(B['constant_error'], 3)}."
        )
        L += [
            "",
            "| Forecast | Spearman with overload | Held out error, scale points | Minus constant |",
            "|---|---|---|---|",
        ]
        for k, e in B["by_forecast"].items():
            cells = [_f(e["spearman"], 2, True), _f(e["error"], 3), _f(e["error_minus_constant"], 3, True)]
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L += [
            "",
            "Regret by trial: "
            + ", ".join(f"{x:.1f}" for x in B["regret"])
            + f". Constant forecast error {_f(B['regret_constant_error'], 2)} points.",
            "",
            "| Expected regret of | Forecasts by trial | Spearman with regret | Error, points | Minus constant |",
            "|---|---|---|---|---|",
        ]
        for k, e in B["by_regret_forecast"].items():
            fc = ", ".join(f"{x:.1f}" for x in B["regret_forecasts"][k])
            cells = [fc, _f(e["spearman"], 2, True), _f(e["error"], 2), _f(e["error_minus_constant"], 2, True)]
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += [
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose "
        "percentile and basic intervals both exclude zero. Development data, exploratory.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/process_dev"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.data, args.out, args.replicates, args.jobs)
    print((args.out / "tables.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
