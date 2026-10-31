"""Forecasting choice overload, the growth of missed best options with set size, for Amendment 8.

Choice overload means that more options make choices worse. With every option's value known, a choice is worse when
it misses the best option on display. A model forecasts, for a set size it was not fitted on, how often people will
miss the best option, and, trial by trial within a set size, which trials those misses fall on.

Models, each with one free sensitivity.
- CL. softmax(lambda v), sensitivity the same at every set size.
- CL with compensation. softmax(lambda f(n) v), f(n) = ((n - 1) / 7) ** 0.18, the rise of sensitivity with set size
  measured on the snack data (H1, Amendment 3). People who compensate choose more sharply in larger sets, so their
  misses grow less than CL's, or not at all.
- Sample until return (SR), the race on softmax(lambda v).

Panels are groups of trials that share a value scale and task conditions, each holding three or more set sizes. Every
forecast for a set size comes from fits to the panel's other set sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import mixed_logit
from .cdm import Displays, fit_softmax, fit_sr_displays, levels, softmax_masked, sr_levels, static_scores

COMPENSATION = 0.18  # elasticity of sensitivity with set size, H1 confirmatory (snack data)
MODELS = ("CL", "CL_comp", "SR")
EXPLORATORY_MODELS = (*MODELS, "DbS", "MLBA", "MXL")  # added after the freeze, exploratory only
REUTSKAJA_FILE = "fixations_data.dta"
REUTSKAJA_SHA256 = "5c7db9563370cff617b170858d517a9c3d18311728ed046e69f87b970a67f33a"  # the package zip


def compensation(n: np.ndarray) -> np.ndarray:
    return ((np.asarray(n, float) - 1.0) / 7.0) ** COMPENSATION


@dataclass(frozen=True)
class Panel:
    """Trials of one task condition. values are NaN past each trial's set size."""

    name: str
    values: np.ndarray  # trials by max set size
    n: np.ndarray
    choice: np.ndarray
    participant: np.ndarray  # index into the participants of the panel's group
    shift: float  # added to values for sample until return, so every value is a whole level from 1 to 35
    best_seen: np.ndarray | None = None  # Reutskaja only, the best option was fixated
    mlba_scale: float = 1.0  # puts the value scale on the six units on which MLBA's kappa = 1 was set

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(x) for x in np.unique(self.n))

    def miss(self) -> np.ndarray:
        top = np.nanmax(self.values, axis=1)
        return self.values[np.arange(self.n.size), self.choice] < top


def load_reutskaja(folder: str | Path) -> tuple[Panel, tuple[int, ...]]:
    """Every choice of the eye tracking experiment (sets of 4, 9 and 16 snacks rated from -5 to 5)."""
    df = pd.read_stata(Path(folder) / REUTSKAJA_FILE)
    df = df.sort_values(["subject", "trialid", "fixOrder"]).reset_index(drop=True)
    key = df[["subject", "trialid"]].to_numpy()
    first = np.r_[True, np.any(key[1:] != key[:-1], axis=1)]
    t = df.loc[first].reset_index(drop=True)
    n = t["numalt"].to_numpy(int)
    R = t[[f"rating{i}" for i in range(1, 17)]].to_numpy(float)
    inside = np.arange(16)[None, :] < n[:, None]
    if np.any(R[inside] < -5) or np.any(R[inside] > 5) or np.any(R[inside] != np.round(R[inside])):
        raise ValueError("ratings must be whole numbers from -5 to 5")
    values = np.where(inside, R, np.nan)
    choice = t["chosendestring"].to_numpy(int) - 1
    if np.any(choice < 0) or np.any(choice >= n):
        raise ValueError("a choice lies outside its set")
    ids = sorted({int(s) for s in t["subject"]})
    index = {s: i for i, s in enumerate(ids)}
    trial_of = np.cumsum(first) - 1
    top = np.nanmax(values, axis=1)
    roi = df["Roinum"].to_numpy(int) - 1
    on_best = values[trial_of, roi] == top[trial_of]
    best_seen = np.zeros(n.size, bool)
    np.logical_or.at(best_seen, trial_of, on_best)
    panel = Panel(
        name="reutskaja",
        values=values,
        n=n,
        choice=choice,
        participant=np.array([index[int(s)] for s in t["subject"]]),
        shift=6.0,
        mlba_scale=0.6,
        best_seen=best_seen,
    )
    return panel, tuple(ids)


def cdm_panels(data) -> dict[str, tuple[list[Panel], int]]:
    """The Caplin, Dean and Martin rounds of Amendment 7, by experiment and complexity, values in dollars."""
    out = {}
    for e, name in enumerate(("standard", "process")):
        people = np.unique(data.participant[data.experiment == e])
        local = np.full(len(data.participants), -1)
        local[people] = np.arange(people.size)
        panels = []
        for c in (3, 7):
            r = (data.experiment == e) & (data.complexity == c)
            panels.append(
                Panel(
                    name=f"cdm_{name}_{c}",
                    values=data.values[r],
                    n=data.n[r],
                    choice=data.choice[r],
                    participant=local[data.participant[r]],
                    shift=0.0,
                    mlba_scale=6.0 / 35.0,
                )
            )
        out[f"cdm_{name}"] = (panels, int(people.size))
    return out


# ----------------------------------------------------------------------------- forecasts
def miss_forecasts(
    panel: Panel, w: np.ndarray, models: tuple[str, ...] = MODELS, fitted: bool = False, loglik: dict | None = None
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    """Probability of missing the best option on every trial, per model. Held out leaves one set size out. Fitted
    (exploratory, added after Amendment 8) fits one sensitivity to every set size and predicts them all.

    MXL (exploratory) is CL with a log normal sensitivity across people, two parameters, predicted by its population
    average. When loglik is a dict it receives every model's log2 probability of the chosen option on every trial,
    from the same fits (exploratory choice accuracy)."""
    v, n, ch = panel.values, panel.n, panel.choice
    top = np.nanmax(v, axis=1, keepdims=True)
    best = np.isfinite(v) & (v == top)
    lv = levels(v + panel.shift, ch)
    disp = Displays.of(lv)
    top_level = (np.rint(top[:, 0] + panel.shift) - 1).astype(int)
    n_top = best.sum(axis=1)
    scores = {"CL": v, "CL_comp": v * compensation(n)[:, None]}
    if "DbS" in models:
        scores["DbS"] = static_scores(v, "DbS_avg")
        scores["MLBA"] = static_scores(v, "MLBA_avg", mlba_scale=panel.mlba_scale)
    out = {m: np.zeros(n.size) for m in models}
    lam: dict[str, list[float]] = {m: [] for m in models}
    folds = [(np.ones(n.size, bool), np.ones(n.size, bool))] if fitted else [(n == s, n != s) for s in panel.sizes]
    if loglik is not None:
        loglik.update({m: np.zeros(n.size) for m in models})
    log2 = np.log(2.0)
    for test, train in folds:
        for m in [x for x in models if x not in ("SR", "MXL")]:
            lm = fit_softmax(scores[m][train], ch[train], w[train])
            lam[m].append(lm)
            q = softmax_masked(lm * scores[m][test])
            out[m][test] = 1.0 - np.where(best[test], q, 0.0).sum(axis=1)
            if loglik is not None:
                loglik[m][test] = np.log(np.maximum(q[np.arange(q.shape[0]), ch[test]], 1e-300)) / log2
        if "MXL" in models:
            start = fit_softmax(v[train], ch[train], w[train])
            mu, sigma = mixed_logit.fit_mxl(v[train], ch[train], panel.participant[train], w[train], start=start)
            lam["MXL"].append([mu, sigma])
            q = mixed_logit.predict_mxl(mu, sigma, v[test])
            out["MXL"][test] = 1.0 - np.where(best[test], q, 0.0).sum(axis=1)
            if loglik is not None:
                loglik["MXL"][test] = np.log(np.maximum(q[np.arange(q.shape[0]), ch[test]], 1e-300)) / log2
        ls = fit_sr_displays(disp, np.where(train, w, 0.0))
        lam["SR"].append(ls)
        logp_u, _ = sr_levels(ls, disp.unique)
        logp = logp_u[disp.index]
        rows = np.flatnonzero(test)
        out["SR"][test] = 1.0 - n_top[rows] * np.exp(logp[rows, top_level[rows]])
        if loglik is not None:
            loglik["SR"][test] = np.maximum(logp[rows, disp.chosen[rows]], np.log(1e-300)) / log2
    return out, lam


def features(panel: Panel) -> dict[str, np.ndarray]:
    """Display features that forecast a miss, signed so that larger means a miss is more likely."""
    v = panel.values
    s = -np.sort(-np.where(np.isfinite(v), v, -np.inf), axis=1)
    top = s[:, 0]
    unit = 1.0  # one dollar or one rating point
    near = (np.isfinite(v) & (v >= top[:, None] - unit) & (v < top[:, None])).sum(axis=1)
    below = np.where(s < top[:, None], s, -np.inf).max(axis=1)  # the best value under the top one
    gap = np.where(np.isfinite(below), top - below, np.inf)  # no worse option, so no miss is possible
    return {"near_best": near.astype(float), "minus_gap": -gap}


def weighted_auc(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Probability that a trial with a miss outscores one without (ties count one half), trials weighted."""
    keep = w > 0
    s, y, w = score[keep], y[keep], w[keep]
    if not (y.any() and (~y).any()):
        return np.nan
    order = np.argsort(s, kind="mergesort")
    s, y, w = s[order], y[order], w[order]
    uniq, start = np.unique(s, return_index=True)
    end = np.r_[start[1:], s.size]
    wn = np.array([w[a:b][~y[a:b]].sum() for a, b in zip(start, end, strict=True)])
    wp = np.array([w[a:b][y[a:b]].sum() for a, b in zip(start, end, strict=True)])
    below = np.r_[0.0, np.cumsum(wn)[:-1]]
    return float((wp * (below + 0.5 * wn)).sum() / (wp.sum() * wn.sum()))


def slope_on_log2(sizes: tuple[int, ...], rates: list[float]) -> float:
    return float(np.polyfit(np.log2(np.asarray(sizes, float)), np.asarray(rates, float), 1)[0])


def panel_measures(panel: Panel, w: np.ndarray, models: tuple[str, ...] = MODELS, fitted: bool = False) -> dict:
    """Observed and forecast miss rates by set size, their growth, and trial by trial discrimination within sizes."""
    y = panel.miss()
    ll: dict | None = {} if "MXL" in models else None
    fc, lam = miss_forecasts(panel, w, models, fitted, ll)
    feats = features(panel)
    sizes = panel.sizes
    rate = {"observed": [float(w[panel.n == s] @ y[panel.n == s] / w[panel.n == s].sum()) for s in sizes]}
    for m in models:
        rate[m] = [float(w[panel.n == s] @ fc[m][panel.n == s] / w[panel.n == s].sum()) for s in sizes]
    growth = {k: slope_on_log2(sizes, r) for k, r in rate.items()}
    auc = {}
    # model forecasts are rounded to ten significant digits before ranking, so displays equal in exact arithmetic tie
    rounded = {m: np.array([float(format(x, ".10g")) for x in fc[m]]) for m in models}
    for name, sc in {**rounded, **feats}.items():
        vals, ws = [], []
        for s in sizes:
            r = panel.n == s
            a = weighted_auc(sc[r], y[r], w[r])
            if np.isfinite(a):
                vals.append(a)
                ws.append(w[r].sum())
        auc[name] = float(np.average(vals, weights=ws))
    rec = {"sizes": list(sizes), "rate": rate, "growth": growth, "auc": auc, "lambda": lam}
    if ll is not None:  # exploratory, mean log2 probability of the chosen option per trial, by set size and overall
        rec["loglik"] = {
            m: {
                "by_size": [float(w[panel.n == s] @ ll[m][panel.n == s] / w[panel.n == s].sum()) for s in sizes],
                "all": float(w @ ll[m] / w.sum()),
            }
            for m in models
        }
    if panel.best_seen is not None:
        unseen = []
        for s in sizes:
            r = (panel.n == s) & y
            unseen.append(float(w[r] @ (~panel.best_seen[r]) / w[r].sum()))
        rec["miss_best_unseen"] = unseen
    return rec
