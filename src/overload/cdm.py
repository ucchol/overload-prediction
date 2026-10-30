"""The search and satisficing data of Caplin, Dean and Martin (2011), for the confirmatory tests of Amendment 7.

People chose one of 10, 20 or 40 sums of money, each written out as a chain of 3 or 7 additions and subtractions, so
an option's value is known only once it has been worked out. A round began with a worthless option selected, people
could select any option at any time, and the last selection when they submitted was their choice. In Experiment 1
(standard choice) only the final choice paid. In Experiment 2 (choice process) the selection held at a random moment
paid, so every selection was meant as the best option found so far. The click file records every selection with its
time in seconds and screen position, and every option's value in screen order.

The data are those of the authors' replication package (openICPSR project 116100, version 1, CC BY 4.0), which needs
a free sign in, so they are not fetched by a script. ``scripts/prepare_cdm_data.sh`` checks the downloaded package and
copies the click file to ``data/cdm``.

Included rounds follow the authors' standard data set (their ``Tables.do``). Sets of 10 options with round codes 1 to
6, sets of 20 and 40 options with a round code, 3 or 7 operations at most, practice rounds and Experiments 3 and 4
left out, and only rounds that were submitted. The authors also dropped one participant as noncompliant without a
stated rule, so every participant is kept here.

The models are those of the other datasets, for sets of different sizes. Values are whole dollars, so sample until
return needs only the number of options worth each dollar amount (as for the snack ratings).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

PACKAGE_SHA256 = "fd91347335913529a408e15b97420434d2dc75d88c7a8f4a00c15bbe9a8257b3"
FILE = "ProcessedCP_All_Final.csv"
FILE_SHA256 = "86adf9e6331192a06765224b41cc4e10d263a85e624cee4da141e684524f7d9e"
SIZES: tuple[int, ...] = (10, 20, 40)
COMPLEXITIES: tuple[int, ...] = (3, 7)
EXPERIMENTS: tuple[str, ...] = ("standard", "process")  # Experiments 1 and 2, tracking 0 and 1
CELLS: tuple[tuple[int, int], ...] = tuple((n, c) for n in SIZES for c in COMPLEXITIES)
V_MAX = 35.0  # values were drawn below 35 dollars
LEVELS = np.arange(1.0, V_MAX + 1.0)  # whole dollars
MAX_N = max(SIZES)
_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 1e2, 21)])


@dataclass(frozen=True)
class CDMData:
    """Included rounds of one or both experiments, and every selection made in them."""

    experiment: np.ndarray  # rounds, 0 standard choice, 1 choice process
    n: np.ndarray  # rounds, options in the set
    complexity: np.ndarray  # rounds, operations per option
    participant: np.ndarray  # rounds, index into participants
    participants: tuple[int, ...]
    values: np.ndarray  # rounds by MAX_N, dollars in screen order, NaN past n
    choice: np.ndarray  # rounds, index of the final selection
    final_time: np.ndarray  # rounds, seconds at the final selection
    sel_round: np.ndarray  # selections, index of their round
    sel_value: np.ndarray  # dollars
    sel_time: np.ndarray  # seconds since the round began
    sel_position: np.ndarray  # index on screen
    sel_better: np.ndarray  # options in the set worth more than the selection
    sel_stop: np.ndarray  # True for the final selection of a round
    sel_first: np.ndarray  # True for the first selection of a round

    @property
    def rounds(self) -> int:
        return int(self.n.size)

    def mask(self) -> np.ndarray:
        """True where a round has an option, rounds by MAX_N."""
        return np.arange(MAX_N)[None, :] < self.n[:, None]


def load_cdm(folder: str | Path) -> CDMData:
    """Read the click file and keep the rounds Amendment 7 includes."""
    df = pd.read_csv(Path(folder) / FILE, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    keep = (
        (df["practice"] == 0)
        & df["tracking"].isin([0, 1])
        & df["rndmaxnumoper"].isin(COMPLEXITIES)
        & ~df["treatid"].isin([21, 24, 25])
        & (((df["numopts"] == 10) & (df["rndcode"] <= 6)) | (df["numopts"].isin([20, 40]) & df["rndcode"].notna()))
        & (df["submit"] == 1)
    )
    d = df.loc[keep].sort_values(["tracking", "userid", "rndnum", "slctnum"]).reset_index(drop=True)
    vals = d[[f"val{i}" for i in range(2, MAX_N + 2)]].to_numpy(float)  # position 1 is the worthless default
    n_row = d["numopts"].to_numpy(int)
    inside = np.arange(MAX_N)[None, :] < n_row[:, None]
    if np.any(vals[inside] < 1) or np.any(vals[inside] > V_MAX) or np.any(vals[inside] != np.round(vals[inside])):
        raise ValueError("every option must be worth a whole number of dollars from 1 to 35")
    vals = np.where(inside, vals, np.nan)
    pos = d["optord"].to_numpy(int) - 2
    listed = vals[np.arange(len(d)), np.clip(pos, 0, MAX_N - 1)]
    if np.any(pos < 0) or np.any(pos >= n_row) or not np.array_equal(listed, d["valnum"].to_numpy(float)):
        raise ValueError("a selection does not match the listed value at its position")
    key = d[["tracking", "userid", "rndnum"]].to_numpy()
    first = np.ones(len(d), bool)
    first[1:] = np.any(key[1:] != key[:-1], axis=1)
    round_of = np.cumsum(first) - 1
    last = np.r_[first[1:], True]
    if not np.array_equal(d["slctnum"].to_numpy()[last], d["numslct"].to_numpy()[last]):
        raise ValueError("the selections of a round are incomplete")
    ids = sorted({int(u) for u in d["userid"]})
    index = {u: i for i, u in enumerate(ids)}
    if int(round_of[-1]) + 1 != int(first.sum()):
        raise ValueError("rounds are not contiguous")
    rv = vals[first]
    sel_value = d["valnum"].to_numpy(float)
    better = (vals > sel_value[:, None]).sum(axis=1)
    return CDMData(
        experiment=d["tracking"].to_numpy(int)[first],
        n=n_row[first],
        complexity=d["rndmaxnumoper"].to_numpy(int)[first],
        participant=np.array([index[int(u)] for u in d["userid"].to_numpy()[first]]),
        participants=tuple(ids),
        values=rv,
        choice=pos[last],
        final_time=d["timeslct"].to_numpy(float)[last],
        sel_round=round_of,
        sel_value=sel_value,
        sel_time=d["timeslct"].to_numpy(float),
        sel_position=pos,
        sel_better=better,
        sel_stop=last,
        sel_first=first,
    )


# ----------------------------------------------------------------------------- static choice models
def static_scores(values: np.ndarray, family: str, mlba_scale: float = 6.0 / V_MAX) -> np.ndarray:
    """Scores per option of CL, DbS_avg and MLBA_avg for sets of any size (NaN past n), as in Amendment 3.

    MLBA's fixed kappa = 1 was set on ratings spanning six units, so the dollar values are first put on that span.
    """
    v = np.asarray(values, float)
    if family == "CL":
        return v
    ok = np.isfinite(v)
    n = ok.sum(axis=1, keepdims=True).astype(float)
    if family == "MLBA_avg":
        v = v * mlba_scale
    d = v[:, :, None] - v[:, None, :]
    pair = ok[:, :, None] & ok[:, None, :]
    if family == "DbS_avg":
        s = np.where(pair, np.sign(d), 0.0).sum(axis=2)
    elif family == "MLBA_avg":
        s = np.where(pair, np.exp(-np.abs(d)) * d, 0.0).sum(axis=2)
    else:
        raise ValueError(f"unknown family {family!r}")
    return np.where(ok, s / (n - 1.0), np.nan)


def softmax_masked(z: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(z), z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def chosen_log_softmax(lam: float, scores: np.ndarray, choice: np.ndarray) -> np.ndarray:
    z = np.where(np.isfinite(scores), lam * scores, -np.inf)
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    return z[np.arange(choice.size), choice] - lse


def fit_softmax(scores: np.ndarray, choice: np.ndarray, w: np.ndarray) -> float:
    """Weighted maximum likelihood sensitivity of a softmax over scores, lambda >= 0, by root finding."""
    keep = w > 0
    s, c, w = scores[keep], choice[keep], w[keep]
    chosen = s[np.arange(c.size), c]
    s0 = np.where(np.isfinite(s), s, 0.0)

    def grad(lam: float) -> float:
        q = softmax_masked(lam * s)
        return float(w @ (chosen - (q * s0).sum(axis=1)))

    if grad(0.0) <= 0.0:
        return 0.0
    hi = 1.0
    while grad(hi) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("choices are perfectly ordered by the scores")
    return float(brentq(grad, 0.0, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))


# ----------------------------------------------------------------------------- sample until return on dollar levels
@dataclass(frozen=True)
class Levels:
    counts: np.ndarray  # rounds by levels, options worth each dollar amount
    chosen: np.ndarray  # rounds, level of the chosen option

    def subset(self, rows: np.ndarray) -> Levels:
        return Levels(self.counts[rows], self.chosen[rows])


def levels(values: np.ndarray, choice: np.ndarray) -> Levels:
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    idx = np.where(ok, np.rint(v - LEVELS[0]), 0).astype(int)
    R = v.shape[0]
    counts = np.zeros((R, LEVELS.size))
    rows = np.repeat(np.arange(R), v.shape[1])[ok.ravel()]
    np.add.at(counts, (rows, idx.ravel()[ok.ravel()]), 1.0)
    return Levels(counts=counts, chosen=idx[np.arange(R), np.asarray(choice, int)])


@cache
def _laguerre() -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.laguerre.laggauss(MAX_N // 2 + 3)  # exact for degree up to MAX_N + 5


def sr_levels(lam: float, lv: Levels) -> tuple[np.ndarray, np.ndarray]:
    """log choice probability of one option on each level (rounds by levels) and expected looks per round."""
    x, wq = _laguerre()
    c = lv.counts
    present = c > 0
    top = np.where(present, LEVELS, -np.inf).max(axis=1, keepdims=True)
    z = np.where(present, lam * (LEVELS[None, :] - top), -np.inf)  # empty levels play no part
    logq = z - np.log((c * np.exp(z)).sum(axis=1, keepdims=True))
    logs = np.log1p(np.exp(logq)[:, :, None] * x[None, None, :])
    total = (c[:, :, None] * logs).sum(axis=1)
    looks = np.exp(total) @ wq
    inner = np.exp(total[:, None, :] - logs) @ (wq * x)
    logp = 2.0 * logq + np.log(inner)
    m = np.where(present, logp, -np.inf).max(axis=1, keepdims=True)
    norm = m + np.log((c * np.exp(np.where(present, logp - m, -np.inf))).sum(axis=1, keepdims=True))
    return logp - norm, looks


def sr_chosen_log(lam: float, lv: Levels) -> np.ndarray:
    logp, _ = sr_levels(lam, lv)
    return np.maximum(logp[np.arange(lv.chosen.size), lv.chosen], np.log(1e-300))


@dataclass(frozen=True)
class Displays:
    """The distinct displays of a set of rounds. The same option sets were shown to many people, so the likelihood
    of sample until return needs one evaluation per distinct display rather than per round."""

    unique: Levels  # one row per distinct display, chosen unused
    index: np.ndarray  # rounds, row of their display
    chosen: np.ndarray  # rounds, level of the chosen option

    @classmethod
    def of(cls, lv: Levels) -> Displays:
        u, inv = np.unique(lv.counts, axis=0, return_inverse=True)
        return cls(Levels(u, np.zeros(u.shape[0], int)), inv.ravel(), lv.chosen)

    def weights(self, w: np.ndarray) -> np.ndarray:
        """Summed round weights per distinct display and chosen level."""
        W = np.zeros(self.unique.counts.shape)
        np.add.at(W, (self.index, self.chosen), w)
        return W


def fit_sr_displays(displays: Displays, w: np.ndarray) -> float:
    """Weighted maximum likelihood sensitivity of sample until return, the grid and Brent search of Amendment 5.

    Identical to ``fit_sr`` (the same likelihood, summed by display), only faster.
    """
    W = displays.weights(w)
    used = W.sum(axis=1) > 0
    U, W = displays.unique.subset(np.flatnonzero(used)), W[used]
    floor = np.log(1e-300)

    def nll(lam: float) -> float:
        logp, _ = sr_levels(lam, U)
        return -float(np.sum(np.where(W > 0, W * np.maximum(logp, floor), 0.0)))

    vals = np.array([nll(g) for g in _GRID])
    k = int(np.argmin(vals))
    if k == _GRID.size - 1:
        raise RuntimeError("the search likelihood keeps rising at the largest sensitivity of the grid")
    lo, hi = _GRID[max(k - 1, 0)], _GRID[k + 1]
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(res.x) if res.fun <= vals[k] else float(_GRID[k])


def fit_sr(lv: Levels, w: np.ndarray) -> float:
    """Weighted maximum likelihood sensitivity of sample until return, the grid and Brent search of Amendment 5."""
    keep = np.flatnonzero(w > 0)
    sub, ww = lv.subset(keep), w[keep]

    def nll(lam: float) -> float:
        return -float(ww @ sr_chosen_log(lam, sub))

    vals = np.array([nll(g) for g in _GRID])
    k = int(np.argmin(vals))
    if k == _GRID.size - 1:
        raise RuntimeError("the search likelihood keeps rising at the largest sensitivity of the grid")
    lo, hi = _GRID[max(k - 1, 0)], _GRID[k + 1]
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(res.x) if res.fun <= vals[k] else float(_GRID[k])


# ----------------------------------------------------------------------------- stopping rules
def fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted logistic regression with an intercept and a standard normal prior on standardized slopes.

    Predictors are standardized with the weighted mean and standard deviation of the training rows, and the
    penalized likelihood (concave) is maximized by Newton's method. Returns the coefficients and the scaling.
    """
    X = np.atleast_2d(np.asarray(X, float).T).T
    keep = w > 0
    Xk, yk, wk = X[keep], y[keep].astype(float), w[keep]
    mu = wk @ Xk / wk.sum()
    sd = np.sqrt(wk @ (Xk - mu) ** 2 / wk.sum())
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(Xk)), (Xk - mu) / sd])
    beta = np.zeros(Z.shape[1])
    if np.all(yk == yk[0]):
        # every training outcome is the same, so the unpenalized intercept has no finite maximum. The fit takes its
        # limit, the slopes stay at the prior mode and the rule predicts that outcome with probability one.
        beta[0] = np.inf if yk[0] else -np.inf
        return beta, mu, sd
    pen = np.r_[0.0, np.ones(Z.shape[1] - 1)]

    def objective(b: np.ndarray) -> float:
        eta = Z @ b
        return float(wk @ (yk * eta - np.logaddexp(0.0, eta)) - 0.5 * pen @ b**2)

    f = objective(beta)
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
        g = Z.T @ (wk * (yk - p)) - pen * beta
        H = (Z * (wk * p * (1 - p))[:, None]).T @ Z + np.diag(pen)
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            new = beta + t * step
            fn = objective(new)
            if fn >= f - 1e-12 or t < 1e-10:
                break
            t *= 0.5
        if abs(fn - f) < 1e-12 * max(1.0, abs(f)) and np.max(np.abs(new - beta)) < 1e-10:
            beta, f = new, fn
            break
        beta, f = new, fn
    return beta, mu, sd


def logistic_log2(beta: np.ndarray, mu: np.ndarray, sd: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log2 probability of each observed outcome, floored at 1e-300."""
    X = np.atleast_2d(np.asarray(X, float).T).T
    eta = beta[0] + ((X - mu) / sd) @ beta[1:]
    logp1 = -np.logaddexp(0.0, -eta)
    logp0 = -np.logaddexp(0.0, eta)
    lp = np.where(y, logp1, logp0)
    return np.maximum(lp, np.log(1e-300)) / np.log(2.0)
