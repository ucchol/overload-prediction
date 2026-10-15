"""Confirmatory test of two look satisficing on three open gaze datasets, ANALYSIS_PLAN.md Amendment 6.

On the snack data, where it was built, two look satisficing foresaw slow decisions far better than sample until return
and timed single decisions better, because a display without a really good option registers looks slowly. These three
datasets of choices among two or three foods were never analyzed here. Within a dataset every display has the same
number of items, so a model that times decisions by the number of items predicts the same time for every trial, while
the search models predict which trials take longer from the values alone. Every prediction comes from fits to other
people (ten folds of participants).

G1  Time, held out. log response time = log(t0 + tau effort), fitted on the training participants and applied to the
    held out ones. Effort is the number of items for the static models (a constant), and the expected number of looks
    for sample until return and for two look satisficing.
G2  Slow decisions. Within each participant, the rank correlation between the model's expected looks and the response
    time, averaged over participants and datasets.
G3  Choices. Two look satisficing (which chooses as sample until return) against CL, non inferiority within 2 percent
    of the held out likelihood of the chosen item.

Secondary. Each dataset on its own. Choices against DbS and MLBA. Three display features (the highest value, the mean
value and the gap between the two highest values) as forecasts of time, alone and against two look satisficing. The
time law of each search model fitted to everyone, which says how long a look takes. In Folke's data, forecasts of how
confident people felt.

Usage
    python -m overload.experiments.two_look_gaze --gaze data/gaze --out results/two_look_gaze --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import softmax

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..gaze import (
    DATASETS,
    GazeSet,
    chosen_log_cl,
    chosen_log_sr,
    fit_cl,
    fit_sr,
    load_gaze,
    sr_looks,
    two_look_looks,
)
from ..snack import scores as static_scores
from .search_confirmatory import fit_time_law, spearman
from .search_confirmatory import noninferiority as _noninferiority

SEED_OFFSET = 60000
FOLD_SEED = 20261014  # apart from the resampling seeds 60000 to 60999
FOLDS = 10
TIME_MODELS = ("static", "SR", "TLS")
STATIC = {"CL": "CL", "DbS": "DbS_avg", "MLBA": "MLBA_avg"}  # single attribute scores of Amendment 3
# MLBA's fixed kappa = 1 was set on snack ratings that span six units, so the values, which span ten, are put on that
# span before MLBA scores them. CL and DbS do not depend on the unit of the values.
MLBA_SCALE = 0.6
CHOICE_MODELS = (*STATIC, "SR")  # two look satisficing chooses as SR
SEARCH = ("SR", "TLS")
FEATURES = ("top_value", "mean_value", "top_gap")  # each expected to go with faster decisions
FORECASTS = ("looks_TLS", "looks_SR", *FEATURES)
CONFIDENCE_FORECASTS = ("looks_TLS", "looks_SR", "entropy_CL")
LN2 = np.log(2.0)
_STATE: dict = {}


def folds(n_participants: int) -> np.ndarray:
    """Fold of every participant, from a fixed permutation of the sorted participants."""
    perm = np.random.default_rng(FOLD_SEED).permutation(n_participants)
    fold = np.zeros(n_participants, dtype=int)
    fold[perm] = np.arange(n_participants) % FOLDS
    return fold


def draws(sets: dict[str, GazeSet], b: int) -> dict[str, np.ndarray | None]:
    if b < 0:
        return {name: None for name in DATASETS}
    rng = np.random.default_rng(SEED_OFFSET + b)
    out = {}
    for name in DATASETS:
        P = len(sets[name].participants)
        out[name] = rng.integers(0, P, size=P)
    return out


def _tie(x: np.ndarray) -> np.ndarray:
    """A forecast rounded to ten significant digits (Python's format ".10g") before it is ranked.

    Forecasts that are equal in exact arithmetic then tie, such as sample until return's looks or CL's entropy on two
    displays whose values differ by a constant, which floating point would otherwise rank in an arbitrary order.
    """
    return np.array([float(format(v, ".10g")) for v in np.asarray(x, dtype=float)])


def _within(person: np.ndarray, x: np.ndarray, y: np.ndarray, mult: np.ndarray) -> float:
    """Rank correlation within each participant (three trials or more, defined), averaged with draw weights."""
    x = _tie(x)
    rs, ws = [], []
    for p in np.flatnonzero(mult > 0):
        rows = np.flatnonzero(person == p)
        if rows.size >= 3:
            r = spearman(x[rows], y[rows])
            if np.isfinite(r):
                rs.append(r)
                ws.append(mult[p])
    rs, ws = np.array(rs), np.array(ws)
    return float(ws @ rs / ws.sum())


def _dataset(gs: GazeSet, d: np.ndarray | None) -> dict:
    P = len(gs.participants)
    mult = np.ones(P) if d is None else np.bincount(d, minlength=P).astype(float)
    w = mult[gs.participant]
    fold_of = folds(P)[gs.participant]
    T, n = gs.values.shape
    # every forecast is a symmetric function of a display's values, so it is computed from the values in ascending
    # order, which makes displays holding the same values tie exactly in the rank correlations
    srt = np.sort(gs.values, axis=1)
    S = {m: static_scores(gs.values * (MLBA_SCALE if m == "MLBA" else 1.0), family) for m, family in STATIC.items()}
    logq = {m: np.zeros(T) for m in CHOICE_MODELS}
    entropy_cl = np.zeros(T)
    looks = {m: np.zeros(T) for m in SEARCH}
    log_seconds = np.log(gs.seconds)
    abs_err = {m: np.zeros(T) for m in TIME_MODELS}
    lam: dict[str, list[float]] = {m: [] for m in CHOICE_MODELS}
    for f in range(FOLDS):
        test = fold_of == f
        train = ~test
        for m in STATIC:
            lm = fit_cl(S[m][train], gs.choice[train], w[train])
            lam[m].append(lm)
            logq[m][test] = chosen_log_cl(lm, S[m][test], gs.choice[test])
            if m == "CL":
                q = softmax(lm * srt[test], axis=1)
                entropy_cl[test] = -(q * np.log2(np.where(q > 0, q, 1.0))).sum(axis=1)
        ls = fit_sr(gs.values[train], gs.choice[train], w[train])
        lam["SR"].append(ls)
        logq["SR"][test] = chosen_log_sr(ls, gs.values[test], gs.choice[test])
        effort = {
            "static": np.full(T, float(n)),
            "SR": sr_looks(ls, srt),
            "TLS": two_look_looks(ls, srt),
        }
        for m in SEARCH:
            looks[m][test] = effort[m][test]
        for m in TIME_MODELS:
            t0, tau = fit_time_law(effort[m][train], log_seconds[train], w[train])
            abs_err[m][test] = np.abs(log_seconds[test] - np.log(t0 + tau * effort[m][test]))
    rec: dict = {"lambda": lam}
    rec["L"] = {m: float(w @ ((np.log(n) + logq[m]) / LN2) / w.sum()) for m in CHOICE_MODELS}
    rec["time"] = {m: float(w @ abs_err[m] / w.sum()) for m in TIME_MODELS}
    features = {
        "looks_TLS": looks["TLS"],
        "looks_SR": looks["SR"],
        "top_value": srt[:, -1],
        "mean_value": srt.mean(axis=1),
        "top_gap": srt[:, -1] - srt[:, -2],
    }
    rec["slow"] = {name: _within(gs.participant, features[name], gs.seconds, mult) for name in FORECASTS}
    # descriptive, the law of each search model fitted to everyone, and the looks it expects
    lam_all = fit_sr(gs.values, gs.choice, w)
    effort_all = {"SR": sr_looks(lam_all, srt), "TLS": two_look_looks(lam_all, srt)}
    rec["law"] = {m: [float(x) for x in fit_time_law(effort_all[m], log_seconds, w)] for m in SEARCH}
    rec["looks"] = {m: float(w @ looks[m] / w.sum()) for m in SEARCH}
    rec["typical_seconds"] = float(np.exp(w @ log_seconds / w.sum()))
    if gs.confidence is not None:
        conf_forecasts = {"looks_TLS": looks["TLS"], "looks_SR": looks["SR"], "entropy_CL": entropy_cl}
        ok = np.isfinite(gs.confidence)
        rec["confidence"] = {
            name: _within(gs.participant[ok], conf_forecasts[name][ok], gs.confidence[ok], mult)
            for name in CONFIDENCE_FORECASTS
        }
    return rec


def run_replicate(b: int) -> dict:
    sets = _STATE["sets"]
    d = draws(sets, b)
    rec: dict = {"b": b, "datasets": {name: _dataset(sets[name], d[name]) for name in DATASETS}}
    D = rec["datasets"]
    rec["pooled"] = {
        "time": {m: float(np.mean([D[x]["time"][m] for x in DATASETS])) for m in TIME_MODELS},
        "slow": {k: float(np.mean([D[x]["slow"][k] for x in DATASETS])) for k in FORECASTS},
        "L": {m: float(np.mean([D[x]["L"][m] for x in DATASETS])) for m in CHOICE_MODELS},
    }
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(b) for b in bs]


def _percent(dl: float) -> float:
    return 100.0 * (2.0**dl - 1.0)


def _percent_summary(e: dict) -> dict:
    return {
        "point": _percent(e["point"]),
        "percentile": [_percent(x) for x in e["percentile"]],
        "basic": [_percent(x) for x in e["basic"]],
        "p": e["p"],
        "clear": e["clear"],
    }


def _summary(point: dict, reps: list[dict], get) -> dict:
    return summarize(get(point), [get(r) for r in reps])


def _block(point: dict, reps: list[dict], get) -> dict:
    """G1 to G3 and the secondary measures for one dataset or for the pooled values."""
    out: dict = {"time": {}, "slow": {}, "choices": {}}
    for m in TIME_MODELS:
        out["time"][m] = _summary(point, reps, lambda r, m=m: get(r)["time"][m])
    for other in ("SR", "static"):
        out["time"][f"TLS_minus_{other}"] = _summary(
            point, reps, lambda r, o=other: get(r)["time"]["TLS"] - get(r)["time"][o]
        )
    for k in FORECASTS:
        out["slow"][k] = _summary(point, reps, lambda r, k=k: get(r)["slow"][k])
    out["slow"]["TLS_minus_SR"] = _summary(
        point, reps, lambda r: get(r)["slow"]["looks_TLS"] - get(r)["slow"]["looks_SR"]
    )
    for k in FEATURES:
        out["slow"][f"TLS_plus_{k}"] = _summary(
            point, reps, lambda r, k=k: get(r)["slow"]["looks_TLS"] + get(r)["slow"][k]
        )
    out["choices"]["times_chance"] = {m: 2.0 ** get(point)["L"][m] for m in CHOICE_MODELS}
    for m in STATIC:
        e = _summary(point, reps, lambda r, m=m: get(r)["L"]["SR"] - get(r)["L"][m])
        out["choices"][f"TLS_minus_{m}_percent"] = _percent_summary(e)
    return out


def summarize_all(point: dict, reps: list[dict]) -> dict:
    out = {"pooled": _block(point, reps, lambda r: r["pooled"])}
    for name in DATASETS:
        B = _block(point, reps, lambda r, name=name: r["datasets"][name])
        D = point["datasets"][name]
        B["lambda"] = D["lambda"]
        B["law"] = {
            m: {
                key: _summary(point, reps, lambda r, name=name, m=m, j=j: r["datasets"][name]["law"][m][j])
                for j, key in enumerate(("t0", "tau"))
            }
            for m in SEARCH
        }
        B["looks"] = {
            m: _summary(point, reps, lambda r, name=name, m=m: r["datasets"][name]["looks"][m]) for m in SEARCH
        }
        B["typical_seconds"] = _summary(point, reps, lambda r, name=name: r["datasets"][name]["typical_seconds"])
        out[name] = B
    P = out["pooled"]
    g1 = P["time"]
    P["G1"] = {
        "supported": bool(
            g1["TLS_minus_SR"]["clear"]
            and g1["TLS_minus_SR"]["point"] < 0
            and g1["TLS_minus_static"]["clear"]
            and g1["TLS_minus_static"]["point"] < 0
        )
    }
    s = P["slow"]
    P["G2"] = {
        "supported": bool(
            s["looks_TLS"]["clear"]
            and s["looks_TLS"]["point"] > 0
            and s["TLS_minus_SR"]["clear"]
            and s["TLS_minus_SR"]["point"] > 0
        )
    }
    P["G3"] = _noninferiority(P["choices"]["TLS_minus_CL_percent"])
    P["G3"]["supported"] = P["G3"]["noninferior"]
    conf = {
        k: _summary(point, reps, lambda r, k=k: r["datasets"]["folke2016"]["confidence"][k])
        for k in CONFIDENCE_FORECASTS
    }
    conf["TLS_minus_SR"] = _summary(
        point,
        reps,
        lambda r: (
            r["datasets"]["folke2016"]["confidence"]["looks_TLS"] - r["datasets"]["folke2016"]["confidence"]["looks_SR"]
        ),
    )
    out["folke2016"]["confidence"] = conf
    return out


def run(folder: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    _STATE.update(sets=load_gaze(folder))
    t0 = time.time()
    point = run_replicate(-1)
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
    summary = summarize_all(point, reps)
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "confirmatory, Amendment 6"}
    )
    write_json(out / "summary.json", summary)
    (out / "table_two_look_gaze.md").write_text(table(summary))
    return summary


def _f(e: dict, digits: int = 3, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    lo, hi = e["percentile"]
    star = " *" if e.get("clear") else ""
    return f"{fmt.format(e['point'])} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def table(s: dict) -> str:
    names = {"pooled": "Pooled (mean of the three datasets)", **{d: d for d in DATASETS}}
    L = ["# Two look satisficing on three gaze datasets (Amendment 6, confirmatory)", ""]
    P = s["pooled"]
    L += [
        "## Decisions",
        "",
        f"- G1, time held out. Two look satisficing minus sample until return "
        f"{_f(P['time']['TLS_minus_SR'], 4, True)}, minus the static models "
        f"{_f(P['time']['TLS_minus_static'], 4, True)}. Supported {P['G1']['supported']}.",
        f"- G2, slow decisions. Two look satisficing {_f(P['slow']['looks_TLS'], 3, True)}, minus sample until return "
        f"{_f(P['slow']['TLS_minus_SR'], 3, True)}. Supported {P['G2']['supported']}.",
        f"- G3, choices. Two look satisficing minus CL {_f(P['choices']['TLS_minus_CL_percent'], 1, True)} percent. "
        f"Non inferior within 2 percent {P['G3']['noninferior']}, better {P['G3']['better']}, "
        f"worse {P['G3']['worse']}. "
        f"Supported {P['G3']['supported']}.",
        "",
    ]
    for key in ("pooled", *DATASETS):
        B = s[key]
        L += [f"## {names[key]}", "", "| Measure | Value |", "|---|---|"]
        for m in TIME_MODELS:
            L.append(f"| Time error, {m} | {_f(B['time'][m])} |")
        L.append(f"| Time error, TLS minus SR | {_f(B['time']['TLS_minus_SR'], 4, True)} |")
        L.append(f"| Time error, TLS minus static | {_f(B['time']['TLS_minus_static'], 4, True)} |")
        for k in FORECASTS:
            L.append(f"| Rank correlation with time, {k} | {_f(B['slow'][k], 3, True)} |")
        L.append(f"| Rank correlation with time, TLS minus SR | {_f(B['slow']['TLS_minus_SR'], 3, True)} |")
        for k in FEATURES:
            L.append(f"| Rank correlation with time, TLS plus {k} | {_f(B['slow'][f'TLS_plus_{k}'], 3, True)} |")
        tc = B["choices"]["times_chance"]
        L.append(
            "| Choices, times chance, "
            + ", ".join(CHOICE_MODELS)
            + " | "
            + ", ".join(f"{tc[m]:.3f}" for m in CHOICE_MODELS)
            + " |"
        )
        for m in STATIC:
            L.append(f"| Choices, TLS minus {m}, percent | {_f(B['choices'][f'TLS_minus_{m}_percent'], 2, True)} |")
        if key != "pooled":
            for m in SEARCH:
                law = B["law"][m]
                L.append(
                    f"| Law fitted to everyone, {m}, start up seconds and seconds per look | "
                    f"{_f(law['t0'])}, {_f(law['tau'])} |"
                )
                L.append(f"| Expected looks, {m}, held out mean | {_f(B['looks'][m], 2)} |")
            L.append(f"| Typical response time, seconds (geometric mean) | {_f(B['typical_seconds'], 2)} |")
        L.append("")
    C = s["folke2016"]["confidence"]
    L += [
        "## Folke 2016, confidence (rank correlation within participant)",
        "",
        "| Forecast | Correlation |",
        "|---|---|",
    ]
    for k in (*CONFIDENCE_FORECASTS, "TLS_minus_SR"):
        L.append(f"| {k} | {_f(C[k], 3, True)} |")
    L += [
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile "
        "and basic intervals both exclude zero.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaze", type=Path, default=Path("data/gaze"))
    ap.add_argument("--out", type=Path, default=Path("results/two_look_gaze"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.gaze, args.out, args.replicates, args.jobs)
    print((args.out / "table_two_look_gaze.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
