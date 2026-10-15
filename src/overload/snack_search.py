"""The search model on the snack data, for the confirmatory tests of Amendment 5.

Sample until return (``overload.search``) chooses the first item that attention visits twice, with attention drawn to
item i with probability q_i = softmax(lambda v)_i. On the snack data every value is a liking rating from -3 to 3, so
the items of a trial fall on seven rating levels and the model needs only the number of items on each level.

With level values v_l, counts c_l and q_l the attention probability of one item on level l,

    P(an item on level l is chosen) = q_l^2 * int t e^(-t) prod_m (1 + q_m t)^(c_m) / (1 + q_l t) dt,
    E[looks]                        = int e^(-t) prod_m (1 + q_m t)^(c_m) dt,

integrals from 0 to infinity. Both integrands are e^(-t) times a polynomial of degree at most n, so Gauss Laguerre
quadrature with n // 2 + 3 nodes is exact up to rounding.

The sensitivity is fitted by maximum likelihood on pooled, weighted trials. The likelihood need not be concave, so the
maximum is located on a grid and refined between the neighbors of the best grid value, as on the development data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.optimize import minimize_scalar

LEVELS = np.arange(-3.0, 4.0)  # the seven liking ratings
_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 1e2, 21)])


@cache
def _laguerre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights exact for e^(-t) times any polynomial of degree up to n + 3."""
    x, w = np.polynomial.laguerre.laggauss(n // 2 + 3)
    return x, w


@dataclass(frozen=True)
class LevelTrials:
    """Trials reduced to rating levels. ``counts`` is trials by levels, ``chosen`` the chosen item's level."""

    counts: np.ndarray
    chosen: np.ndarray
    n: int

    @classmethod
    def from_values(cls, values: np.ndarray, choice: np.ndarray) -> LevelTrials:
        values = np.asarray(values, float)
        level = np.rint(values - LEVELS[0]).astype(int)
        if level.min() < 0 or level.max() >= LEVELS.size or not np.allclose(values, LEVELS[level]):
            raise ValueError("every value must be an integer rating from -3 to 3")
        T, n = values.shape
        counts = np.zeros((T, LEVELS.size))
        np.add.at(counts, (np.repeat(np.arange(T), n), level.ravel()), 1.0)
        chosen = level[np.arange(T), np.asarray(choice, int)]
        return cls(counts=counts, chosen=chosen, n=n)

    def subset(self, rows: np.ndarray) -> LevelTrials:
        return LevelTrials(self.counts[rows], self.chosen[rows], self.n)


def _log_attention(lam: float, counts: np.ndarray) -> np.ndarray:
    """log q per item on each level, trials by levels. Empty levels get the value they would have."""
    present = counts > 0
    top = np.where(present, LEVELS, -np.inf).max(axis=1, keepdims=True)
    z = lam * (LEVELS[None, :] - top)
    return z - np.log((counts * np.exp(z)).sum(axis=1, keepdims=True))


def _log_levels(lam: float, trials: LevelTrials) -> tuple[np.ndarray, np.ndarray]:
    """log choice probability of one item on each level (trials by levels) and expected looks per trial."""
    x, wq = _laguerre(trials.n)
    logq = _log_attention(lam, trials.counts)
    logs = np.log1p(np.exp(logq)[:, :, None] * x[None, None, :])  # trials by levels by nodes
    total = (trials.counts[:, :, None] * logs).sum(axis=1)  # trials by nodes
    looks = np.exp(total) @ wq
    inner = np.exp(total[:, None, :] - logs) @ (wq * x)  # trials by levels
    logp = 2.0 * logq + np.log(inner)
    present = trials.counts > 0
    m = np.where(present, logp, -np.inf).max(axis=1, keepdims=True)
    norm = m + np.log((trials.counts * np.exp(np.where(present, logp - m, -np.inf))).sum(axis=1, keepdims=True))
    return logp - norm, looks


def level_probabilities(lam: float, trials: LevelTrials) -> tuple[np.ndarray, np.ndarray]:
    """Choice probability of one item on each level (trials by levels) and expected looks per trial."""
    logp, looks = _log_levels(lam, trials)
    return np.exp(logp), looks


def chosen_log_probability(lam: float, trials: LevelTrials) -> np.ndarray:
    logp, _ = _log_levels(lam, trials)
    return logp[np.arange(trials.chosen.size), trials.chosen]


def item_probabilities(lam: float, trials: LevelTrials, values: np.ndarray) -> np.ndarray:
    """Choice probability of every displayed item, trials by items."""
    p, _ = level_probabilities(lam, trials)
    level = np.rint(np.asarray(values, float) - LEVELS[0]).astype(int)
    return np.take_along_axis(p, level, axis=1)


def expected_looks(lam: float, trials: LevelTrials) -> np.ndarray:
    return level_probabilities(lam, trials)[1]


def two_look_levels(lam: float, trials: LevelTrials, v_top: float = float(LEVELS[-1])) -> tuple[np.ndarray, np.ndarray]:
    """Expected looks and expected distinct items looked at under two look satisficing, per trial.

    Attention per item on level l is q_l = exp(lam v_l / 2) / Z and a look registers with probability
    h_l = exp(lam (v_l - v_top) / 2), so items register at rate rho_l = q_l h_l and P = sum_l c_l rho_l. With
    a_l = q_l + P - rho_l, and t = x / P or t = x / a_l,

        E[looks]    = (1 / P) int e^(-x) prod_m (1 + rho_m x / P)^(c_m) dx,
        E[distinct] = sum_l c_l q_l / a_l int e^(-x) prod_m (1 + rho_m x / a_l)^(c_m - [m = l]) dx,

    both exact under Gauss Laguerre quadrature. Choices are those of sample until return at lam
    (``item_probabilities``).
    """
    x, wq = _laguerre(trials.n)
    c = trials.counts
    present = c > 0
    top = np.where(present, LEVELS, -np.inf).max(axis=1, keepdims=True)
    za = 0.5 * lam * (LEVELS[None, :] - top)
    q = np.exp(za) / (c * np.exp(za)).sum(axis=1, keepdims=True)  # per item on each level
    rho = q * np.exp(0.5 * lam * (LEVELS[None, :] - v_top))
    P = (c * rho).sum(axis=1)  # trials
    lg = np.log1p(rho[:, :, None] * (x[None, None, :] / P[:, None, None]))  # trials by levels by nodes
    looks = np.exp((c[:, :, None] * lg).sum(axis=1)) @ wq / P
    a = q + P[:, None] - rho  # trials by levels
    scaled = rho[:, None, :, None] * (x[None, None, None, :] / a[:, :, None, None])  # trials, l, m, nodes
    lg2 = np.log1p(scaled)
    total = (c[:, None, :, None] * lg2).sum(axis=2)  # trials, l, nodes
    own = np.take_along_axis(lg2, np.arange(LEVELS.size)[None, :, None, None], axis=2)[:, :, 0, :]
    integral = np.exp(total - own) @ wq  # trials, l
    distinct = np.where(present, c * q / a * integral, 0.0).sum(axis=1)
    return looks, distinct


def _nll(lam: float, parts: Sequence[tuple[LevelTrials, np.ndarray]]) -> float:
    return -float(sum(w @ chosen_log_probability(lam, t) for t, w in parts))


def _positive(parts: Sequence[tuple[LevelTrials, np.ndarray]]) -> list[tuple[LevelTrials, np.ndarray]]:
    """Only trials with positive weight enter a fit."""
    out = []
    for t, w in parts:
        w = np.asarray(w, float)
        keep = np.flatnonzero(w > 0)
        if keep.size:
            out.append((t.subset(keep), w[keep]))
    return out


def fit_search(parts: Sequence[tuple[LevelTrials, np.ndarray]]) -> float:
    """Weighted maximum likelihood sensitivity of sample until return, lambda >= 0.

    ``parts`` pairs the trials of a set size with their weights. The grid spans four orders of magnitude around the
    scale of one rating step, five points per decade, and the best grid value is refined by bounded Brent search
    between its neighbors.
    """
    parts = _positive(parts)
    if not parts:
        return 0.0
    vals = np.array([_nll(g, parts) for g in _GRID])
    k = int(np.argmin(vals))
    if k == _GRID.size - 1:
        raise RuntimeError("the search likelihood keeps rising at the largest sensitivity of the grid")
    lo, hi = _GRID[max(k - 1, 0)], _GRID[k + 1]
    res = minimize_scalar(lambda g: _nll(g, parts), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(res.x) if res.fun <= vals[k] else float(_GRID[k])
