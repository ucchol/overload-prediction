"""Maximum likelihood of the single parameter, and the three ways a model is evaluated.

For a softmax model with fixed scores v_t in the fitted trials, the negative log likelihood

    NLL(lambda) = sum_t [ N_t logsumexp(lambda v_t) - lambda c_t . v_t ]

is convex in lambda, with derivative sum_t [ N_t E_q[v_t] - c_t . v_t ], which increases in lambda.
The estimate is zero when that derivative is nonnegative at zero, and otherwise the derivative's root,
bracketed by doubling and found by Brent's method. The lapse of the screening model is handled the same
way on [0, 1], since -sum c log((1 - eps) P + eps / n) is convex in eps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, softmax

from .data import Task
from .models import Model
from .search import accept_in_reading_order, sample_until_return

_XTOL = 1e-14
_RTOL = 4 * np.finfo(float).eps
_LAMBDA_MAX = 1e9


class SeparationError(RuntimeError):
    """The likelihood keeps rising as the sensitivity grows, so no finite estimate exists."""


# ----------------------------------------------------------------------------- softmax
def softmax_nll(lam: float, scores: list[np.ndarray], counts: list[np.ndarray]) -> float:
    return float(sum(c.sum() * logsumexp(lam * v) - lam * float(c @ v) for v, c in zip(scores, counts, strict=True)))


def _softmax_gradient(lam: float, scores: list[np.ndarray], counts: list[np.ndarray]) -> float:
    g = 0.0
    for v, c in zip(scores, counts, strict=True):
        q = softmax(lam * v)
        g += c.sum() * float(q @ v) - float(c @ v)
    return g


def fit_softmax(scores: list[np.ndarray], counts: list[np.ndarray]) -> float:
    """Maximum likelihood sensitivity, lambda >= 0, for fixed option scores."""
    active = [(v, c) for v, c in zip(scores, counts, strict=True) if c.sum() > 0 and np.ptp(v) > 0]
    if not active:
        return 0.0
    vs, cs = [a[0] for a in active], [a[1] for a in active]
    if _softmax_gradient(0.0, vs, cs) >= 0.0:
        return 0.0
    if all(float(c[v < v.max()].sum()) == 0.0 for v, c in zip(vs, cs, strict=True)):
        raise SeparationError("every choice falls on a highest scoring option, so the likelihood has no maximum")
    scale = max(float(np.ptp(v)) for v in vs)
    hi = 1.0 / scale
    while _softmax_gradient(hi, vs, cs) < 0.0:
        hi *= 2.0
        if hi > _LAMBDA_MAX:
            raise SeparationError("choices are perfectly ordered by the scores")
    return float(brentq(_softmax_gradient, 0.0, hi, args=(vs, cs), xtol=_XTOL, rtol=_RTOL, maxiter=500))


def predict_softmax(lam: float, scores: np.ndarray) -> np.ndarray:
    return softmax(lam * scores)


# ----------------------------------------------------------------------------- lapse
def _lapse_gradient(eps: float, bases: list[np.ndarray], counts: list[np.ndarray]) -> float:
    g = 0.0
    for p, c in zip(bases, counts, strict=True):
        u = 1.0 / p.size
        mix = (1.0 - eps) * p + eps * u
        chosen = c > 0
        g -= float(np.sum(c[chosen] * (u - p[chosen]) / mix[chosen]))
    return g


def fit_lapse(bases: list[np.ndarray], counts: list[np.ndarray]) -> float:
    """Maximum likelihood lapse, 0 <= eps <= 1, toward a uniform guess."""
    if all(c.sum() == 0 for c in counts):
        return 1.0
    zero_on_chosen = any(np.any((c > 0) & (p <= 0.0)) for p, c in zip(bases, counts, strict=True))
    if not zero_on_chosen and _lapse_gradient(0.0, bases, counts) >= 0.0:
        return 0.0
    if _lapse_gradient(1.0, bases, counts) <= 0.0:
        return 1.0
    lo = 1e-200 if zero_on_chosen else 0.0
    return float(brentq(_lapse_gradient, lo, 1.0, args=(bases, counts), xtol=_XTOL, rtol=_RTOL, maxiter=500))


def predict_lapse(eps: float, base: np.ndarray) -> np.ndarray:
    return (1.0 - eps) * base + eps / base.size


# ----------------------------------------------------------------------------- search models
SEARCH_RULES = {"race": sample_until_return, "accept": accept_in_reading_order}
_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 121)])


def search_nll(kind: str, lam: float, scores: list[np.ndarray], counts: list[np.ndarray]) -> float:
    rule = SEARCH_RULES[kind]
    total = 0.0
    for v, c in zip(scores, counts, strict=True):
        p = rule(lam, v)
        chosen = c > 0
        total -= float(c[chosen] @ np.log(np.maximum(p[chosen], 1e-300)))
    return total


def fit_search(kind: str, scores: list[np.ndarray], counts: list[np.ndarray]) -> float:
    """Maximum likelihood sensitivity of a search model, lambda >= 0.

    The likelihood of a search model is not guaranteed to be concave in lambda, so the estimate is located
    on a grid of 122 values spanning six orders of magnitude (scaled by the widest score range) and refined
    by a bounded scalar search between the neighbors of the best grid value.
    """
    active = [(v, c) for v, c in zip(scores, counts, strict=True) if c.sum() > 0 and np.ptp(v) > 0]
    if not active:
        return 0.0
    vs, cs = [a[0] for a in active], [a[1] for a in active]
    scale = max(float(np.ptp(v)) for v in vs)
    grid = _GRID / scale
    vals = np.array([search_nll(kind, g, vs, cs) for g in grid])
    k = int(np.argmin(vals))
    if k == grid.size - 1:
        raise SeparationError("the search model likelihood keeps rising at the largest sensitivity")
    lo, hi = grid[max(k - 1, 0)], grid[k + 1]
    res = minimize_scalar(
        lambda lam: search_nll(kind, lam, vs, cs),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 500},
    )
    return float(res.x) if res.fun <= vals[k] else float(grid[k])


# ----------------------------------------------------------------------------- evaluation
@dataclass
class Evaluation:
    """Predictions of one model for one task.

    ``fitted[t]`` comes from the fit to all trials, ``held_out[t]`` from the fit to every other
    trial and ``alone[t]`` from the fit to trial t by itself. ``params`` records each fit's estimate.
    """

    model: str
    fitted: list[np.ndarray]
    held_out: list[np.ndarray]
    alone: list[np.ndarray]
    params: dict[str, object]


def _fit(model: Model, bases: list[np.ndarray], counts: list[np.ndarray]) -> float:
    if model.kind == "softmax":
        return fit_softmax(bases, counts)
    if model.kind == "lapse":
        return fit_lapse(bases, counts)
    if model.kind in SEARCH_RULES:
        return fit_search(model.kind, bases, counts)
    raise ValueError(f"unknown model kind {model.kind!r}")


def _predict(model: Model, theta: float, base: np.ndarray) -> np.ndarray:
    if model.kind == "softmax":
        return predict_softmax(theta, base)
    if model.kind == "lapse":
        return predict_lapse(theta, base)
    return SEARCH_RULES[model.kind](theta, base)


def evaluate(
    model: Model,
    task: Task,
    counts: list[np.ndarray],
    bases: list[np.ndarray] | None = None,
    alone: bool = True,
) -> Evaluation:
    """Fitted, held out and (optionally) alone predictions of ``model`` for ``task``."""
    bases = model.bases(task) if bases is None else bases
    T = len(bases)
    pooled = _fit(model, bases, counts)
    fitted = [_predict(model, pooled, b) for b in bases]
    held, theta_ho = [], []
    for t in range(T):
        others = [u for u in range(T) if u != t]
        th = _fit(model, [bases[u] for u in others], [counts[u] for u in others])
        theta_ho.append(th)
        held.append(_predict(model, th, bases[t]))
    al, theta_al = [], []
    if alone:
        for t in range(T):
            th = _fit(model, [bases[t]], [counts[t]])
            theta_al.append(th)
            al.append(_predict(model, th, bases[t]))
    return Evaluation(
        model=model.name,
        fitted=fitted,
        held_out=held,
        alone=al,
        params={"pooled": pooled, "held_out": theta_ho, "alone": theta_al},
    )
