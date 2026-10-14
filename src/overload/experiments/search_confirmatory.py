"""Confirmatory tests of the search model on the open snack data, ANALYSIS_PLAN.md Amendment 5.

The development data showed that a model that searches, sample until return, predicts choices as well as the static
models built on the same values, and that it alone is fast enough to be how people decided. Twelve displays could not
show more. The snack data of Thomas, Molter and Krajbich (2021) have thousands of random displays of 9 to 36 snacks,
with response times and eye movements, so the model's predictions about search, time and premonition can be tested
display by display. Every quantity below comes from fits to other set sizes (leave one set size out).

S1  Search grows sublinearly. The elasticity of the number of distinct items looked at with set size, observed and as
    predicted by the search model. Static models, as processes that evaluate every item, predict an elasticity of 1.
S2  Time, held out. log response time = log(t0 + tau effort), fitted on three set sizes and applied to the fourth.
    Effort is n for CL, n (n - 1) for DbS_avg and MLBA_avg, and the expected number of looks for the search model.
S3  Premonition of slow decisions. Within participant and set size, the rank correlation between a forecast and the
    response time, averaged over set sizes and participants. Forecasts, the search model's expected looks and CL's
    entropy.
S4  Premonition of missing the best. Held out log score of the event that a highest rated item is chosen, relative to
    a constant forecast.
S5  Choices. The search model predicts the chosen item of unseen set sizes at least as well as CL, the best static model
    on these data, to within a margin of 2 percent of the held out likelihood (non inferiority).

Secondary. Choices against DbS_avg and MLBA_avg, the elasticity of CL's sensitivity that the search model implies, the
level of search, the time each step may take, and rank correlations of other forecasts with time.

Usage
    python -m overload.experiments.search_confirmatory --snack data/snack --out results/search_confirmatory --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import softmax
from scipy.stats import rankdata

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..snack import SET_SIZES, SnackData, held_out, load_snack
from ..snack_search import LevelTrials, chosen_log_probability, expected_looks, fit_search, item_probabilities

SEED_OFFSET = 40000
STATIC = ("CL", "DbS_avg", "MLBA_avg")
MODELS = (*STATIC, "SR")
EQUIVALENCE = 0.1  # S1, margin on the elasticity scale
NONINFERIORITY = 2.0  # S5, margin in percent of the held out likelihood of the chosen item
FORECASTS = ("looks_SR", "entropy_CL", "entropy_DbS_avg", "entropy_MLBA_avg", "top_rating", "rating_sd", "top_count")
LN2 = np.log(2.0)

_STATE: dict = {}


# ----------------------------------------------------------------------------- data
def prepare(data: SnackData) -> dict:
    """Everything that does not depend on the bootstrap draw, per set size."""
    blocks = []
    for b in data.blocks:
        if b.rt is None:
            raise ValueError("the snack data must carry response times (Block.rt)")
        values = b.values
        top = values.max(axis=1)
        is_top = values == top[:, None]
        seconds = b.rt / 1000.0
        timed = np.isfinite(seconds) & (seconds > 0)
        blocks.append(
            {
                "n": b.n,
                "values": values,
                "levels": LevelTrials.from_values(values, b.choice),
                "choice": b.choice,
                "participant": b.participant,
                "distinct_seen": b.seen.sum(axis=1).astype(float),
                "is_top": is_top,
                "best": is_top[np.arange(b.choice.size), b.choice].astype(float),
                "log_seconds": np.where(timed, np.log(np.where(timed, seconds, 1.0)), np.nan),
                "seconds": seconds,
                "timed": timed,
                "features": {
                    "top_rating": top,
                    "rating_sd": values.std(axis=1),
                    "top_count": is_top.sum(axis=1).astype(float),
                },
            }
        )
    groups = []  # (block index, participant, trial rows with a response time)
    for k, blk in enumerate(blocks):
        for p in np.unique(blk["participant"]):
            rows = np.flatnonzero((blk["participant"] == p) & blk["timed"])
            if rows.size >= 3:
                groups.append((k, int(p), rows))
    return {"blocks": blocks, "groups": groups, "participants": len(data.participants)}


def draw(n_participants: int, b: int) -> np.ndarray | None:
    if b < 0:
        return None
    return np.random.default_rng(SEED_OFFSET + b).integers(0, n_participants, size=n_participants)


# ----------------------------------------------------------------------------- helpers
def entropy_bits(q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(q > 0, q * np.log2(q), 0.0).sum(axis=1)


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(w @ x / w.sum())


def elasticity(means: list[float]) -> float:
    """Least squares slope of log mean on log set size over the four set sizes."""
    return float(np.polyfit(np.log(SET_SIZES), np.log(means), 1)[0])


T0_MIN = 1e-9  # seconds, the smallest start up time of the time law
_LOG_RATIO_STEPS = np.arange(-12.0, 12.0 + 1e-9, 0.15)  # log of tau / t0 times the median effort


def fit_time_law(effort: np.ndarray, log_seconds: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Global minimum over t0 >= 1e-9 and tau >= 0 of the weighted squared error of log seconds against log(t0 + tau e).

    The law is profiled. For a fixed ratio r = tau / t0 the best log t0 is the weighted mean of y - log(1 + r e), so the
    error depends on r alone. It is minimized over a grid of r spanning twenty orders of magnitude around the inverse
    median effort, refined by bounded Brent search between the neighbors of the best grid value, and compared with the
    two boundaries, tau = 0 (time independent of effort) and t0 = 1e-9 (time proportional to effort).
    """
    keep = w > 0
    e, y, ww = effort[keep], log_seconds[keep], w[keep]
    total = float(ww.sum())

    def profile(log_r: float) -> tuple[float, float]:
        z = np.log1p(np.exp(log_r) * e)
        a = float(ww @ (y - z)) / total
        return a, float(ww @ (y - a - z) ** 2)

    a0 = float(ww @ y) / total
    candidates = [(float(ww @ (y - a0) ** 2), float(np.exp(a0)), 0.0)]  # tau = 0

    grid = _LOG_RATIO_STEPS - np.log(float(np.median(e)))
    values = np.array([profile(g)[1] for g in grid])
    k = int(np.argmin(values))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
    res = minimize_scalar(lambda g: profile(g)[1], bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    g = float(res.x) if res.fun <= values[k] else float(grid[k])
    a, sse = profile(g)
    if np.exp(a) >= T0_MIN:
        candidates.append((sse, float(np.exp(a)), float(np.exp(g + a))))

    def boundary(log_tau: float) -> float:  # t0 = T0_MIN
        return float(ww @ (y - np.log(T0_MIN + np.exp(log_tau) * e)) ** 2)

    g0 = float(ww @ (y - np.log(e))) / total
    res = minimize_scalar(boundary, bounds=(g0 - 10.0, g0 + 10.0), method="bounded", options={"xatol": 1e-12})
    candidates.append((float(res.fun), T0_MIN, float(np.exp(res.x))))
    _, t0, tau = min(candidates, key=lambda c: c[0])
    return t0, tau


def implied_cl_sensitivity(values: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """CL's sensitivity that best fits choice probabilities p (expected log likelihood, weighted trials)."""

    def score(lam: float) -> float:
        q = softmax(lam * values, axis=1)
        return float(w @ ((q - p) * values).sum(axis=1))

    if score(0.0) >= 0:
        return 0.0
    hi = 1.0
    while score(hi) < 0:
        hi *= 2.0
        if hi > 1e4:
            raise RuntimeError("no finite CL sensitivity reproduces these probabilities")
    return float(brentq(score, 0.0, hi, xtol=1e-12, rtol=4 * np.finfo(float).eps))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


# ----------------------------------------------------------------------------- one replicate
def run_replicate(b: int) -> dict:
    data: SnackData = _STATE["data"]
    prep = _STATE["prep"]
    blocks = prep["blocks"]
    K = len(blocks)
    d = draw(prep["participants"], b)
    w = data.weights(d)
    mult = np.ones(prep["participants"]) if d is None else np.bincount(d, minlength=prep["participants"]).astype(float)
    rec: dict = {"b": b}

    # held out probabilities of the static models, every trial of the held out set size
    q: dict[str, list[np.ndarray]] = {}
    lam: dict[str, list[float]] = {}
    for family in STATIC:
        ho = held_out(data, w, family, 0.0, "all", want_q=True)
        q[family], lam[family] = ho["q"], ho["lambda"]

    # the search model, one sensitivity per fold, applied to every set size (training sets need it for time)
    lam["SR"], q["SR"], looks_fold = [], [], []
    for k in range(K):
        lk = fit_search([(blocks[j]["levels"], w[j]) for j in range(K) if j != k])
        lam["SR"].append(lk)
        q["SR"].append(item_probabilities(lk, blocks[k]["levels"], blocks[k]["values"]))
        looks_fold.append([expected_looks(lk, blocks[j]["levels"]) for j in range(K)])
    looks_held = [looks_fold[k][k] for k in range(K)]
    rec["lambda"] = lam

    # choices, held out log score in bits above a uniform guess
    rec["L"] = {}
    for m in MODELS:
        by_size = []
        for k, blk in enumerate(blocks):
            if m == "SR":
                logq = chosen_log_probability(lam["SR"][k], blk["levels"])
            else:
                logq = np.log(q[m][k][np.arange(blk["choice"].size), blk["choice"]])
            by_size.append(wmean((np.log(blk["n"]) + logq) / LN2, w[k]))
        rec["L"][m] = {"by_size": by_size, "L": float(np.mean(by_size))}

    # S1, search
    observed = [wmean(blk["distinct_seen"], w[k]) for k, blk in enumerate(blocks)]
    predicted = [wmean(looks_held[k] - 1.0, w[k]) for k in range(K)]
    rec["S1"] = {
        "observed": observed,
        "predicted": predicted,
        "observed_elasticity": elasticity(observed),
        "predicted_elasticity": elasticity(predicted),
    }

    # S2, time law, held out
    rec["S2"] = {}
    for m in MODELS:
        errors, params = [], []
        for k in range(K):
            train = [j for j in range(K) if j != k]

            def effort(j: int, m: str = m, k: int = k) -> np.ndarray:
                n = blocks[j]["n"]
                if m == "SR":
                    return looks_fold[k][j]
                size = float(n) if m == "CL" else float(n * (n - 1))
                return np.full(blocks[j]["values"].shape[0], size)

            tm = [blocks[j]["timed"] for j in train]
            e = np.concatenate([effort(j)[t] for j, t in zip(train, tm, strict=True)])
            y = np.concatenate([blocks[j]["log_seconds"][t] for j, t in zip(train, tm, strict=True)])
            ww = np.concatenate([w[j][t] for j, t in zip(train, tm, strict=True)])
            t0, tau = fit_time_law(e, y, ww)
            th = blocks[k]["timed"]
            pred = np.log(t0 + tau * effort(k)[th])
            errors.append(wmean(np.abs(blocks[k]["log_seconds"][th] - pred), w[k][th]))
            params.append((t0, tau))
        rec["S2"][m] = {"by_size": errors, "error": float(np.mean(errors)), "params": params}

    # S3, premonition of slow decisions, and the other forecasts
    forecast = {}
    for k, blk in enumerate(blocks):
        f = {"looks_SR": looks_held[k]}
        for family in STATIC:
            f[f"entropy_{family}"] = entropy_bits(q[family][k])
        f.update(blk["features"])
        forecast[k] = f
    per_participant: dict[str, dict[int, list[float]]] = {name: {} for name in FORECASTS}
    for k, p, rows in prep["groups"]:
        if mult[p] == 0:
            continue
        y = blocks[k]["seconds"][rows]
        for name in FORECASTS:
            r = spearman(forecast[k][name][rows], y)
            if np.isfinite(r):
                per_participant[name].setdefault(p, []).append(r)
    rec["S3"] = {}
    for name in FORECASTS:
        ps = sorted(per_participant[name])
        r = np.array([np.mean(per_participant[name][p]) for p in ps])
        m = mult[ps]
        rec["S3"][name] = float(m @ r / m.sum())

    # S4, premonition of missing the best
    rec["S4"] = {}
    for m in (*MODELS, "constant"):
        scores = []
        for k, blk in enumerate(blocks):
            y = blk["best"]
            if m == "constant":
                train = [j for j in range(K) if j != k]
                base = sum(float(w[j] @ blocks[j]["best"]) for j in train) / sum(float(w[j].sum()) for j in train)
                prob = np.full(y.size, base)
            else:
                prob = (q[m][k] * blk["is_top"]).sum(axis=1)
            prob = np.clip(prob, 1e-15, 1 - 1e-15)
            scores.append(wmean((y * np.log(prob) + (1 - y) * np.log1p(-prob)) / LN2, w[k]))
        rec["S4"][m] = {"by_size": scores, "score": float(np.mean(scores))}

    # secondary, the elasticity of CL's sensitivity that the search model implies
    implied = [implied_cl_sensitivity(blk["values"], q["SR"][k], w[k]) for k, blk in enumerate(blocks)]
    rec["implied_cl_sensitivity"] = implied
    rec["implied_elasticity"] = elasticity(implied)

    # secondary, effort per set size and median response time (point estimates only)
    if b < 0:
        rec["median_seconds"] = [float(np.median(blk["seconds"][blk["timed"]])) for blk in blocks]
        rec["mean_effort"] = {
            "CL": [float(blk["n"]) for blk in blocks],
            "DbS_avg": [float(blk["n"] * (blk["n"] - 1)) for blk in blocks],
            "SR": [float(np.mean(looks_held[k][blocks[k]["timed"]])) for k in range(K)],
        }
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(b) for b in bs]


# ----------------------------------------------------------------------------- summaries
def _percent(dl: float) -> float:
    return 100.0 * (2.0**dl - 1.0)


def noninferiority(e: dict, margin: float = NONINFERIORITY) -> dict:
    """S5 decision from a summarized percent difference (search model minus CL).

    Non inferior when the lower bounds of both the percentile and the basic interval lie above minus the margin.
    Better when both intervals lie above zero, worse when both lie below it, which can happen within the margin.
    """
    lower = min(e["percentile"][0], e["basic"][0])
    return {
        "margin": margin,
        "noninferior": bool(lower > -margin),
        "better": bool(e["percentile"][0] > 0 and e["basic"][0] > 0),
        "worse": bool(e["percentile"][1] < 0 and e["basic"][1] < 0),
    }


def summarize_all(point: dict, reps: list[dict]) -> dict:
    out: dict = {}
    R = reps
    # S1
    obs = summarize(point["S1"]["observed_elasticity"], [r["S1"]["observed_elasticity"] for r in R])
    pred = summarize(point["S1"]["predicted_elasticity"], [r["S1"]["predicted_elasticity"] for r in R])
    below_one = summarize(1.0 - point["S1"]["observed_elasticity"], [1.0 - r["S1"]["observed_elasticity"] for r in R])
    diff = summarize(
        point["S1"]["predicted_elasticity"] - point["S1"]["observed_elasticity"],
        [r["S1"]["predicted_elasticity"] - r["S1"]["observed_elasticity"] for r in R],
    )
    within = bool(diff["percentile"][0] >= -EQUIVALENCE and diff["percentile"][1] <= EQUIVALENCE)
    out["S1"] = {
        "observed_elasticity": obs,
        "predicted_elasticity": pred,
        "one_minus_observed": below_one,
        "predicted_minus_observed": diff,
        "within_margin": within,
        "supported": bool(below_one["clear"] and below_one["point"] > 0 and within),
        "observed_by_size": point["S1"]["observed"],
        "predicted_by_size": point["S1"]["predicted"],
        "observed_over_predicted_by_size": [
            o / p for o, p in zip(point["S1"]["observed"], point["S1"]["predicted"], strict=True)
        ],
        "observed_by_size_interval": np.percentile([r["S1"]["observed"] for r in R], [2.5, 97.5], axis=0).tolist(),
        "predicted_by_size_interval": np.percentile([r["S1"]["predicted"] for r in R], [2.5, 97.5], axis=0).tolist(),
    }
    # S2
    out["S2"] = {"error": {}, "SR_minus": {}}
    for m in MODELS:
        out["S2"]["error"][m] = summarize(point["S2"][m]["error"], [r["S2"][m]["error"] for r in R])
        out["S2"]["error"][m]["by_size"] = point["S2"][m]["by_size"]
        out["S2"]["error"][m]["params"] = point["S2"][m]["params"]
    for m in STATIC:
        out["S2"]["SR_minus"][m] = summarize(
            point["S2"]["SR"]["error"] - point["S2"][m]["error"],
            [r["S2"]["SR"]["error"] - r["S2"][m]["error"] for r in R],
        )
    s2 = out["S2"]["SR_minus"]
    out["S2"]["supported"] = bool(
        s2["CL"]["clear"] and s2["CL"]["point"] < 0 and s2["DbS_avg"]["clear"] and s2["DbS_avg"]["point"] < 0
    )
    # S3
    out["S3"] = {"correlation": {}}
    for name in FORECASTS:
        out["S3"]["correlation"][name] = summarize(point["S3"][name], [r["S3"][name] for r in R])
    out["S3"]["SR_minus_CL"] = summarize(
        point["S3"]["looks_SR"] - point["S3"]["entropy_CL"],
        [r["S3"]["looks_SR"] - r["S3"]["entropy_CL"] for r in R],
    )
    sr = out["S3"]["correlation"]["looks_SR"]
    out["S3"]["supported"] = bool(sr["clear"] and sr["point"] > 0)
    # S4
    out["S4"] = {"score": {}, "minus_constant": {}}
    for m in (*MODELS, "constant"):
        out["S4"]["score"][m] = summarize(point["S4"][m]["score"], [r["S4"][m]["score"] for r in R])
    for m in MODELS:
        out["S4"]["minus_constant"][m] = summarize(
            point["S4"][m]["score"] - point["S4"]["constant"]["score"],
            [r["S4"][m]["score"] - r["S4"]["constant"]["score"] for r in R],
        )
    out["S4"]["SR_minus_CL"] = summarize(
        point["S4"]["SR"]["score"] - point["S4"]["CL"]["score"],
        [r["S4"]["SR"]["score"] - r["S4"]["CL"]["score"] for r in R],
    )
    # secondary, choices
    out["choices"] = {"times_chance": {}, "SR_minus_percent": {}}
    for m in MODELS:
        e = summarize(point["L"][m]["L"], [r["L"][m]["L"] for r in R])
        out["choices"]["times_chance"][m] = {
            "point": 2.0 ** e["point"],
            "percentile": [2.0**x for x in e["percentile"]],
            "by_size": [2.0**x for x in point["L"][m]["by_size"]],
        }
    for m in STATIC:
        dl = point["L"]["SR"]["L"] - point["L"][m]["L"]
        e = summarize(dl, [r["L"]["SR"]["L"] - r["L"][m]["L"] for r in R])
        out["choices"]["SR_minus_percent"][m] = {
            "point": _percent(e["point"]),
            "percentile": [_percent(x) for x in e["percentile"]],
            "basic": [_percent(x) for x in e["basic"]],
            "p": e["p"],
            "clear": e["clear"],
        }
    s5 = out["choices"]["SR_minus_percent"]["CL"]
    out["S5"] = {"SR_minus_CL_percent": s5, **noninferiority(s5)}
    out["S5"]["supported"] = out["S5"]["noninferior"]
    out["implied_elasticity"] = summarize(point["implied_elasticity"], [r["implied_elasticity"] for r in R])
    out["implied_cl_sensitivity"] = point["implied_cl_sensitivity"]
    out["lambda"] = point["lambda"]
    out["per_step_seconds"] = {
        m: [s / e for s, e in zip(point["median_seconds"], point["mean_effort"][m], strict=True)]
        for m in point["mean_effort"]
    }
    out["median_seconds"] = point["median_seconds"]
    out["mean_effort"] = point["mean_effort"]
    return out


# ----------------------------------------------------------------------------- run
def run(folder: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    data = load_snack(folder)
    _STATE.update(data=data, prep=prepare(data))
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
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "confirmatory, Amendment 5"}
    )
    write_json(out / "summary.json", summary)
    (out / "table_search_confirmatory.md").write_text(table(summary))
    return summary


# ----------------------------------------------------------------------------- report
def _f(e: dict, digits: int = 2, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    lo, hi = e["percentile"]
    star = " *" if e.get("clear") else ""
    return f"{fmt.format(e['point'])} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def table(s: dict) -> str:
    L = ["# Search model, confirmatory tests on the snack data (Amendment 5)", ""]
    S1 = s["S1"]
    L += [
        "## S1, search grows sublinearly",
        "",
        "| | 9 | 16 | 25 | 36 | Elasticity |",
        "|---|---|---|---|---|---|",
        "| Distinct items looked at, observed | "
        + " | ".join(f"{x:.2f}" for x in S1["observed_by_size"])
        + f" | {_f(S1['observed_elasticity'])} |",
        "| Predicted by sample until return | "
        + " | ".join(f"{x:.2f}" for x in S1["predicted_by_size"])
        + f" | {_f(S1['predicted_elasticity'])} |",
        "| Predicted by evaluating every item | 9 | 16 | 25 | 36 | 1 |",
        "| Observed over predicted by sample until return | "
        + " | ".join(f"{x:.2f}" for x in S1["observed_over_predicted_by_size"])
        + " | |",
        "",
        f"- Observed elasticity below one by {_f(S1['one_minus_observed'])}.",
        f"- Predicted minus observed elasticity {_f(S1['predicted_minus_observed'], sign=True)}, "
        f"within the margin of {EQUIVALENCE} {S1['within_margin']}. Supported {S1['supported']}.",
        "",
        "## S2, time held out (mean absolute error of log response time)",
        "",
        "| Model | Error | 9 | 16 | 25 | 36 | Sample until return minus model |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        e = s["S2"]["error"][m]
        diff = _f(s["S2"]["SR_minus"][m], 3, True) if m in s["S2"]["SR_minus"] else ""
        L.append(f"| {m} | {_f(e, 3)} | " + " | ".join(f"{x:.3f}" for x in e["by_size"]) + f" | {diff} |")
    L += ["", f"Supported {s['S2']['supported']}.", "", "## S3, premonition of slow decisions", ""]
    L += ["| Forecast | Mean within participant rank correlation with response time |", "|---|---|"]
    for name, e in s["S3"]["correlation"].items():
        L.append(f"| {name} | {_f(e, 3, True)} |")
    L += [
        "",
        f"- Sample until return minus CL entropy {_f(s['S3']['SR_minus_CL'], 3, True)}. "
        f"Supported {s['S3']['supported']}.",
        "",
        "## S4, premonition of missing the best (bits per trial)",
        "",
        "| Model | Log score | Minus constant |",
        "|---|---|---|",
    ]
    for m in (*MODELS, "constant"):
        mc = _f(s["S4"]["minus_constant"][m], 4, True) if m in s["S4"]["minus_constant"] else ""
        L.append(f"| {m} | {_f(s['S4']['score'][m], 4)} | {mc} |")
    s5 = s["S5"]["SR_minus_CL_percent"]
    L += [
        "",
        f"- Sample until return minus CL {_f(s['S4']['SR_minus_CL'], 4, True)}.",
        "",
        "## S5, choices, held out likelihood of the chosen item relative to chance",
        "",
        f"- Sample until return minus CL {_f(s5, 1, True)} percent "
        f"(basic {s5['basic'][0]:.1f} to {s5['basic'][1]:.1f}). "
        f"Non inferior within {s['S5']['margin']:.0f} percent {s['S5']['noninferior']}, better {s['S5']['better']}, "
        f"worse {s['S5']['worse']}. "
        f"Supported {s['S5']['supported']}.",
        "",
        "## Secondary",
        "",
        "Choices of every model, held out likelihood of the chosen item relative to chance.",
        "",
        "| Model | Times chance | 9 | 16 | 25 | 36 | Sample until return minus model, % |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in MODELS:
        e = s["choices"]["times_chance"][m]
        d = s["choices"]["SR_minus_percent"].get(m)
        dtext = _f(d, 1, True) if d else ""
        L.append(
            f"| {m} | {e['point']:.2f} [{e['percentile'][0]:.2f}, {e['percentile'][1]:.2f}] | "
            + " | ".join(f"{x:.2f}" for x in e["by_size"])
            + f" | {dtext} |"
        )
    L += [
        "",
        f"- Elasticity of CL's sensitivity implied by sample until return {_f(s['implied_elasticity'], 3, True)} "
        "(observed in Amendment 3, +0.18).",
        "- Time each step may take, median response time over mean effort, seconds, at 9, 16, 25 and 36 items. "
        + "; ".join(f"{m} " + ", ".join(f"{x:.3g}" for x in v) for m, v in s["per_step_seconds"].items())
        + ".",
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose "
        "percentile and basic intervals both exclude zero.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snack", type=Path, default=Path("data/snack"))
    ap.add_argument("--out", type=Path, default=Path("results/search_confirmatory"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.snack, args.out, args.replicates, args.jobs)
    print((args.out / "table_search_confirmatory.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
