"""The open snack choice data of Thomas, Molter and Krajbich (2021), the confirmatory data of H1.

Specified in ANALYSIS_PLAN.md, Amendment 3. Every set size forms one block of trials with a fixed number of
displayed items. Participants saw every display once, so trials are pooled, and a bootstrap replicate weights
each trial by the number of times its participant was drawn.

Models score the items of a trial from their liking ratings and turn scores into probabilities with one fitted
sensitivity, over all displayed items or over the looked-at items only. The fit is the same maximum likelihood
as ``overload.fit.fit_softmax``, vectorized over the trials of a block and weighted. The root of the score function
is found by Newton's method safeguarded by bisection, which reaches the same root as Brent's method in fewer steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .fit import SeparationError
from .models import KAPPA0, set_size_factor

SET_SIZES: tuple[int, ...] = (9, 16, 25, 36)
COMMIT = "f6ee87fdca53f6b28c251689067927ffea25be37"
FAMILIES: tuple[str, ...] = ("CL", "DbS_avg", "MLBA_avg")
VERSIONS: tuple[str, ...] = ("all", "seen")
BETA_BOUNDS = (-3.0, 3.0)

_XTOL = 1e-14
_RTOL = 4 * np.finfo(float).eps
_LAMBDA_MAX = 1e9


# ----------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Block:
    """The included trials of one set size."""

    n: int
    values: np.ndarray  # trials by items, liking ratings
    choice: np.ndarray  # chosen item index per trial
    seen: np.ndarray  # trials by items, True when the item was looked at in the trial
    participant: np.ndarray  # index into SnackData.participants per trial
    rt: np.ndarray | None = None  # response time per trial in milliseconds (Amendment 5), unused by the H1 tests

    @property
    def trials(self) -> int:
        return int(self.choice.size)

    @property
    def chosen_seen(self) -> np.ndarray:
        return self.seen[np.arange(self.trials), self.choice]


@dataclass(frozen=True)
class SnackData:
    blocks: tuple[Block, ...]
    participants: tuple[int, ...]  # identifiers, sorted
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def weights(self, draw: np.ndarray | None = None) -> list[np.ndarray]:
        """Trial weights per block, one for the original sample, the draw count of its participant otherwise."""
        if draw is None:
            return [np.ones(b.trials) for b in self.blocks]
        mult = np.bincount(draw, minlength=len(self.participants)).astype(float)
        return [mult[b.participant] for b in self.blocks]


def load_snack(folder: str | Path) -> SnackData:
    """Read the four aggregate files and keep the trials Amendment 3 includes."""
    folder = Path(folder)
    frames = {n: pd.read_csv(folder / f"setsize-{n}_desc-data.csv") for n in SET_SIZES}
    ids = sorted({int(s) for df in frames.values() for s in df["subject"]})
    index = {s: i for i, s in enumerate(ids)}
    blocks = []
    for n in SET_SIZES:
        df = frames[n]
        values = df[[f"item_value_{i}" for i in range(n)]].to_numpy(float)
        choice = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
        gaze = df[[f"cumulative_gaze_{i}" for i in range(n)]].to_numpy(float)
        ok = (
            np.isfinite(choice)
            & (choice >= 0)
            & (choice <= n - 1)
            & (choice == np.round(choice))
            & np.isfinite(values).all(axis=1)
        )
        blocks.append(
            Block(
                n=n,
                values=values[ok],
                choice=choice[ok].astype(int),
                seen=np.nan_to_num(gaze[ok], nan=0.0) > 0.0,
                participant=np.array([index[int(s)] for s in df["subject"].to_numpy()[ok]], dtype=int),
                rt=pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)[ok] if "rt" in df else None,
            )
        )
    return SnackData(blocks=tuple(blocks), participants=tuple(ids))


# ----------------------------------------------------------------------------- scores
def scores(values: np.ndarray, family: str) -> np.ndarray:
    """Item scores per trial from the ratings, the single attribute version of Section 3 of the plan."""
    if family == "CL":
        return values.astype(float)
    n = values.shape[1]
    d = values[:, :, None] - values[:, None, :]  # d[t, i, j] = v_i - v_j
    if family == "DbS_avg":
        return np.sign(d).sum(axis=2) / (n - 1)
    if family == "MLBA_avg":
        return (np.exp(-KAPPA0 * np.abs(d)) * d).sum(axis=2) / (n - 1)
    raise ValueError(f"unknown family {family!r}")


def included_rows(data: SnackData, version: str) -> list[np.ndarray]:
    """Trials that enter an analysis of a version, every trial, or those whose chosen item was looked at."""
    if version == "all":
        return [np.arange(b.trials) for b in data.blocks]
    return [np.flatnonzero(b.chosen_seen) for b in data.blocks]


# ----------------------------------------------------------------------------- probabilities and fit
def probabilities(lam: float, S: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Softmax of lam * S over the available items of every trial, zero for unavailable items."""
    z = lam * S
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compress(S: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Distinct scores of every trial and how many available items carry each, padded with zero counts.

    Items with equal scores are interchangeable in a softmax, so a trial reduces to its score levels. With
    ratings on seven levels, a trial has at most seven distinct scores under every model here.
    """
    T, n = S.shape
    avail = np.ones((T, n), dtype=bool) if mask is None else mask
    order = np.argsort(S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, order, axis=1)
    As = np.take_along_axis(avail, order, axis=1)
    first = np.ones((T, n), dtype=bool)
    first[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
    level = np.cumsum(first, axis=1) - 1
    L = int(level.max()) + 1 if T else 1
    rows = np.repeat(np.arange(T), n)
    counts = np.zeros((T, L))
    np.add.at(counts, (rows, level.ravel()), As.ravel().astype(float))
    levels = np.zeros((T, L))
    levels[rows, level.ravel()] = Ss.ravel()
    return levels, counts


@dataclass(frozen=True, eq=False)
class Base:
    """Unscaled scores, choices and availability of a set of trials, with their compressed form."""

    S: np.ndarray
    choice: np.ndarray
    mask: np.ndarray | None  # None means every item is available

    @cached_property
    def chosen(self) -> np.ndarray:
        return self.S[np.arange(self.choice.size), self.choice]

    @cached_property
    def compressed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score levels, available counts per level, and a log mask that is minus infinity for empty levels,
        each stored levels by trials so that sums over levels run along the first axis."""
        levels, counts = compress(self.S, self.mask)
        logmask = np.where(counts > 0, 0.0, -np.inf)
        return (
            np.ascontiguousarray(levels.T),
            np.ascontiguousarray(counts.T),
            np.ascontiguousarray(logmask.T),
        )

    @cached_property
    def fit_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """The informative trials (available scores not all equal) in compressed form, their chosen scores,
        whether each choice has the top available score, the trial index and the largest score range."""
        levels, counts, logmask = self.compressed
        present = counts > 0
        smax = np.where(present, levels, -np.inf).max(axis=0)
        smin = np.where(present, levels, np.inf).min(axis=0)
        i = np.flatnonzero(smax > smin)
        chosen = self.chosen[i]
        span = float((smax[i] - smin[i]).max()) if i.size else 0.0
        return (
            np.ascontiguousarray(levels[:, i]),
            np.ascontiguousarray(counts[:, i]),
            np.ascontiguousarray(logmask[:, i]),
            chosen,
            chosen >= smax[i],
            i,
            span,
        )


@dataclass(frozen=True, eq=False)
class Part:
    """Trials entering a fit, their weights and the set size factor that multiplies their scores."""

    base: Base
    w: np.ndarray
    c: float = 1.0


def make_part(S: np.ndarray, choice: np.ndarray, mask: np.ndarray | None, w: np.ndarray, c: float = 1.0) -> Part:
    return Part(Base(np.asarray(S, float), np.asarray(choice, int), mask), np.asarray(w, float), float(c))


def _grad_hess(lam: float, items: Sequence[tuple]) -> tuple[float, float]:
    """First and second derivative of the weighted negative log likelihood in lambda."""
    g = h = 0.0
    for levels, counts, logmask, chosen, w, c in items:
        z = (lam * c) * levels + logmask
        z -= z.max(axis=0)
        e = counts * np.exp(z)
        den = e.sum(axis=0)
        es = e * levels
        m1 = es.sum(axis=0) / den
        m2 = (es * levels).sum(axis=0) / den
        g += c * float(w @ (m1 - chosen))
        h += c * c * float(w @ (m2 - m1 * m1))
    return g, h


def fit_sensitivity(parts: Sequence[Part], start: float | None = None) -> float:
    """Weighted maximum likelihood sensitivity, lambda >= 0, as in ``overload.fit.fit_softmax``."""
    items, tops, spans = [], [], []
    for p in parts:
        levels, counts, logmask, chosen, top, index, span = p.base.fit_arrays
        w = p.w[index]
        if index.size and (w > 0).any():
            items.append((levels, counts, logmask, chosen, w, p.c))
            tops.append(bool(np.all(top[w > 0])))
            spans.append(span * p.c)
    if not items:
        return 0.0
    g, h = _grad_hess(0.0, items)
    if g >= 0.0:
        return 0.0
    if all(tops):
        raise SeparationError("every choice falls on a highest scoring item, so the likelihood has no maximum")
    # Newton's method from the left of the root, where the score function is concave and the steps rise toward
    # the root. A start to the right of the root only sets the upper bracket.
    lo, hi = 0.0, math.inf
    lam = -g / h
    if start is not None and start > 0.0:
        gs, hs = _grad_hess(start, items)
        if gs < 0.0:
            lo, lam = start, max(start - gs / hs, start)
        else:
            hi = start
            lam = min(lam, 0.5 * start)
    for _ in range(500):
        g, h = _grad_hess(lam, items)
        if g == 0.0:
            return lam
        if g < 0.0:
            lo = lam
        else:
            hi = lam
        new = lam - g / h if h > 0.0 else math.inf
        if abs(new - lam) <= _XTOL + _RTOL * abs(new):
            return new
        if not lo < new < hi:
            new = 2.0 * lam if math.isinf(hi) else 0.5 * (lo + hi)
            if hi - lo <= _XTOL + _RTOL * abs(new):
                return new
        if new > _LAMBDA_MAX:
            raise SeparationError("choices are perfectly ordered by the scores")
        lam = new
    raise RuntimeError("the sensitivity fit did not converge")


def _chosen_log_probability(lam: float, part: Part) -> np.ndarray:
    """Natural log of the probability of the chosen item in every trial of a part."""
    levels, counts, logmask = part.base.compressed
    a = lam * part.c
    z = a * levels + logmask
    zmax = z.max(axis=0)
    log_den = zmax + np.log((counts * np.exp(z - zmax)).sum(axis=0))
    return a * part.base.chosen - log_den


def weighted_loglik(lam: float, parts: Sequence[Part]) -> float:
    """Weighted log likelihood in nats."""
    return float(sum(p.w @ _chosen_log_probability(lam, p) for p in parts))


def log_scores(lam: float, part: Part) -> np.ndarray:
    """log2(n q_chosen) per trial, bits above a uniform guess over the displayed items."""
    return (np.log(part.base.S.shape[1]) + _chosen_log_probability(lam, part)) / np.log(2.0)


# ----------------------------------------------------------------------------- model evaluation
def _base(data: SnackData, k: int, family: str, version: str, rows: np.ndarray) -> Base:
    key = (k, family, version, rows.tobytes())
    if key not in data._cache:
        b = data.blocks[k]
        sk = ("scores", k, family)
        if sk not in data._cache:
            data._cache[sk] = scores(b.values, family)
        mask = b.seen[rows] if version == "seen" else None
        data._cache[key] = Base(data._cache[sk][rows], b.choice[rows], mask)
    return data._cache[key]


def _parts(
    data: SnackData,
    weights: Sequence[np.ndarray],
    family: str,
    beta: float,
    version: str,
    blocks: Sequence[int],
    rows: Sequence[np.ndarray] | None = None,
) -> list[Part]:
    rows = included_rows(data, version) if rows is None else rows
    return [
        Part(_base(data, k, family, version, rows[k]), weights[k][rows[k]], set_size_factor(data.blocks[k].n, beta))
        for k in blocks
    ]


def held_out(
    data: SnackData,
    weights: Sequence[np.ndarray],
    family: str,
    beta: float,
    version: str,
    rows: Sequence[np.ndarray] | None = None,
    want_q: bool = False,
) -> dict:
    """Leave one set size out. Per set size the fitted sensitivity, the weighted mean held-out log score and, when
    asked, the held-out probabilities of every included trial."""
    K = len(data.blocks)
    rows = included_rows(data, version) if rows is None else rows
    lams, means, probs = [], [], []
    start = None
    for k in range(K):
        others = [j for j in range(K) if j != k]
        lam = fit_sensitivity(_parts(data, weights, family, beta, version, others, rows), start)
        start = lam if lam > 0 else None
        (part,) = _parts(data, weights, family, beta, version, [k], rows)
        ls = log_scores(lam, part)
        lams.append(lam)
        means.append(float(part.w @ ls / part.w.sum()))
        if want_q:
            probs.append(probabilities(lam * part.c, part.base.S, part.base.mask))
    return {"lambda": lams, "by_size": means, "L": float(np.mean(means)), "q": probs}


def size_alone(
    data: SnackData, weights: Sequence[np.ndarray], version: str, rows: Sequence[np.ndarray] | None = None
) -> list[float]:
    """CL's sensitivity fitted to each set size by itself."""
    rows = included_rows(data, version) if rows is None else rows
    return [fit_sensitivity(_parts(data, weights, "CL", 0.0, version, [k], rows)) for k in range(len(data.blocks))]


def elasticity(
    data: SnackData, weights: Sequence[np.ndarray], version: str, rows: Sequence[np.ndarray]
) -> tuple[float, float]:
    """CL's sensitivity elasticity, (beta, lambda) maximizing the likelihood with beta in BETA_BOUNDS.

    The profile likelihood in beta (lambda refitted at each beta) is maximized by bounded Brent search.
    """
    K = range(len(data.blocks))
    last = {"lam": None}

    def negative_profile(beta: float) -> float:
        parts = _parts(data, weights, "CL", beta, version, K, rows)
        lam = fit_sensitivity(parts, last["lam"])
        last["lam"] = lam if lam > 0 else None
        return -weighted_loglik(lam, parts)

    res = minimize_scalar(negative_profile, bounds=BETA_BOUNDS, method="bounded", options={"xatol": 1e-9})
    beta = float(res.x)
    lam = fit_sensitivity(_parts(data, weights, "CL", beta, version, K, rows))
    return beta, lam
