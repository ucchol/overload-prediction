"""Choice models, each with exactly one fitted parameter.

Softmax models turn option scores into probabilities with a fitted sensitivity. Their scores depend
only on the displayed options and the fixed rubric shares, so they are computed once per trial and
reused across folds and bootstrap replicates. The screening model, elimination by aspects, yields a
fixed base distribution per trial and fits a lapse toward a uniform guess.

Scores follow Section 3 of ANALYSIS_PLAN.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np

from .data import Task, Trial

KAPPA0 = 1.0
BETAS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)


# ----------------------------------------------------------------------------- scores
def score_qr(trial: Trial, r: np.ndarray) -> np.ndarray:
    """Quantal response reads the rubric scores directly."""
    return trial.U.astype(float)


def score_cl(trial: Trial, r: np.ndarray) -> np.ndarray:
    """Conditional logit, the rubric weighted sum of an option's own attribute levels."""
    return trial.X @ r


def _dbs_sum(trial: Trial, r: np.ndarray) -> np.ndarray:
    X = trial.X
    wins = np.sign(X[:, None, :] - X[None, :, :]).sum(axis=1)
    return wins @ r


def _mlba_sum(trial: Trial, r: np.ndarray, kappa: float = KAPPA0) -> np.ndarray:
    X = trial.X
    d = X[:, None, :] - X[None, :, :]
    w = np.exp(-kappa * np.abs(d))
    return (w * d).sum(axis=1) @ r


def score_dbs_sum(trial: Trial, r: np.ndarray) -> np.ndarray:
    """Decision by sampling, ordinal comparisons summed over every rival."""
    return _dbs_sum(trial, r)


def score_dbs_avg(trial: Trial, r: np.ndarray) -> np.ndarray:
    """Decision by sampling, ordinal comparisons averaged over rivals."""
    return _dbs_sum(trial, r) / (trial.n - 1)


def score_mlba_sum(trial: Trial, r: np.ndarray) -> np.ndarray:
    """MLBA drift with distance weighted comparisons summed over every rival."""
    return _mlba_sum(trial, r)


def score_mlba_avg(trial: Trial, r: np.ndarray) -> np.ndarray:
    """MLBA drift with distance weighted comparisons averaged over rivals."""
    return _mlba_sum(trial, r) / (trial.n - 1)


def set_size_factor(n: int, beta: float) -> float:
    """Multiplier ((n - 1) / 7) ** beta, which equals one in an eight option set."""
    return ((n - 1) / 7.0) ** beta


def scaled(scorer: Callable[[Trial, np.ndarray], np.ndarray], beta: float):
    """A scorer whose scores are multiplied by the set size factor."""

    def score(trial: Trial, r: np.ndarray) -> np.ndarray:
        return set_size_factor(trial.n, beta) * scorer(trial, r)

    score.__name__ = f"{scorer.__name__}_beta{beta:+.1f}"
    return score


# ----------------------------------------------------------------------------- screening
def eba_probabilities(X: np.ndarray, r: np.ndarray, aspect: str = "median") -> np.ndarray:
    """Elimination by aspects with rubric weights, as specified in Section 3 of the plan.

    Each attribute with a positive share defines one aspect, fixed for the trial. With ``aspect``
    "median" an option has the aspect when its level is at or above the median of the displayed set,
    with "best" when its level equals the best level in the set. From the remaining set, an aspect is
    drawn with probability proportional to its share among the aspects that some but not all remaining
    options have, options without it are eliminated, and the process repeats. When no aspect splits
    the remaining set, the choice is uniform over it.
    """
    n, _ = X.shape
    keep = np.flatnonzero(r > 0)
    if aspect == "median":
        has = X[:, keep] >= np.median(X[:, keep], axis=0)
    elif aspect == "best":
        has = X[:, keep] >= X[:, keep].max(axis=0)
    else:
        raise ValueError(f"unknown aspect rule {aspect!r}")
    weights = r[keep]
    masks = [frozenset(np.flatnonzero(has[:, a]).tolist()) for a in range(len(keep))]

    @cache
    def choose(remaining: frozenset[int]) -> tuple[float, ...]:
        size = len(remaining)
        splitting = [a for a in range(len(keep)) if 0 < len(remaining & masks[a]) < size]
        p = np.zeros(n)
        if not splitting:
            for i in remaining:
                p[i] = 1.0 / size
            return tuple(p)
        total = weights[splitting].sum()
        for a in splitting:
            p += (weights[a] / total) * np.asarray(choose(remaining & masks[a]))
        return tuple(p)

    return np.asarray(choose(frozenset(range(n))))


# ----------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class Model:
    """A model of the plan: a name, its kind and how it reads a trial.

    For kind "softmax", ``base`` returns option scores and the fitted parameter is a sensitivity.
    For kind "lapse", ``base`` returns a probability distribution and the fitted parameter is a lapse.
    """

    name: str
    kind: str
    base: Callable[[Trial, np.ndarray], np.ndarray]

    def bases(self, task: Task) -> list[np.ndarray]:
        return [np.asarray(self.base(t, task.shares), dtype=float) for t in task.trials]


def _eba(aspect: str) -> Callable[[Trial, np.ndarray], np.ndarray]:
    def base(trial: Trial, r: np.ndarray) -> np.ndarray:
        return eba_probabilities(trial.X, r, aspect)

    base.__name__ = f"eba_{aspect}"
    return base


def baseline_models() -> list[Model]:
    """Models of result R0."""
    return [
        Model("QR", "softmax", score_qr),
        Model("CL", "softmax", score_cl),
        Model("DbS_avg", "softmax", score_dbs_avg),
        Model("MLBA_avg", "softmax", score_mlba_avg),
        Model("DbS_sum", "softmax", score_dbs_sum),
        Model("MLBA_sum", "softmax", score_mlba_sum),
    ]


def set_size_models() -> list[Model]:
    """Models of test A1a, the beta grid for CL and the averaged comparison scores."""
    out = []
    for label, scorer in (("CL", score_cl), ("DbS_avg", score_dbs_avg), ("MLBA_avg", score_mlba_avg)):
        for beta in BETAS:
            out.append(Model(f"{label}|beta={beta:+.1f}", "softmax", scaled(scorer, beta)))
    return out


def screening_models() -> list[Model]:
    """Models of test A1b, elimination by aspects with the primary and the secondary aspect rule."""
    return [Model("EBA", "lapse", _eba("median")), Model("EBA_best", "lapse", _eba("best"))]
