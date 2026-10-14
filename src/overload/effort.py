"""How much work each model does to reach a decision, in elementary steps.

Every choice model is also an algorithm, and an algorithm takes time. Following Payne, Bettman and Johnson
(1993), a model's effort is the number of elementary steps it must carry out on a trial. Here one step is
reading one attribute value of one option and using it once, either adding it into the option's value or
comparing it with the same attribute of another option. A model may only use the attributes it values, so m
counts the attributes that differ between the options of a trial and carry a positive rubric share.

- QR and CL value every option on its own, n m steps (a weighted sum per option).
- DbS and MLBA compare every option with every rival on every attribute, n (n - 1) m steps, whether the
  comparisons are then averaged or summed.
- Elimination by aspects checks one aspect on every option still in play, removes those without it and
  repeats. Its effort is the expected number of option and aspect checks until the choice is made.
- Sample until return looks at options one at a time and reads the m attributes of each option it looks
  at, E[looks] m steps.
- Accept in reading order reads the m attributes of each option it inspects, E[inspections] m steps.

Effort is a count, so it needs a clock to become seconds. ``overload.experiments.decision_time`` converts it
with two constants shared by every model, a start up time and a time per step.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.special import expit

from .data import Trial
from .search import sample_until_return_length


def used_attributes(trial: Trial, r: np.ndarray) -> int:
    """Attributes that differ between the trial's options and carry a positive rubric share."""
    return int(((np.ptp(trial.X, axis=0) > 0) & (r > 0)).sum())


def eba_checks(X: np.ndarray, r: np.ndarray, aspect: str = "median") -> float:
    """Expected number of option and aspect checks of elimination by aspects (``models.eba_probabilities``).

    Aspects are the valued attributes that differ between the options, drawn in proportion to their rubric
    shares, and each draw is checked on every option still in play. A draw that every remaining option
    passes, or none does, eliminates nothing and is followed by another, so a stage with remaining set S
    costs |S| / s checks on average, where s is the share of the rubric carried by aspects that split S.
    When no aspect splits S and more than one option remains, every aspect is checked once to find that
    out, |S| A checks for A aspects, and the choice among S is random.
    """

    n, _ = X.shape
    keep = np.flatnonzero((r > 0) & (np.ptp(X, axis=0) > 0))  # attributes shown that differ and are valued
    if keep.size == 0:
        return 0.0
    cut = np.median(X[:, keep], axis=0) if aspect == "median" else X[:, keep].max(axis=0)
    has = X[:, keep] >= cut
    weights = r[keep]
    masks = [frozenset(np.flatnonzero(has[:, a]).tolist()) for a in range(len(keep))]

    @cache
    def checks(remaining: frozenset[int]) -> float:
        size = len(remaining)
        if size == 1:
            return 0.0
        splitting = [a for a in range(len(keep)) if 0 < len(remaining & masks[a]) < size]
        if not splitting:
            return float(size * len(keep))
        total = weights[splitting].sum()
        stage = size * weights.sum() / total
        return stage + sum((weights[a] / total) * checks(remaining & masks[a]) for a in splitting)

    return float(checks(frozenset(range(n))))


def accept_inspections(lam: float, v: np.ndarray) -> float:
    """Expected inspections of accept in reading order, passes repeating until an option is accepted."""
    v = np.asarray(v, dtype=float)
    a = expit(lam * (v - v.max()))
    reach = np.concatenate([[1.0], np.cumprod(1.0 - a)[:-1]])
    return float(reach.sum() / (1.0 - np.prod(1.0 - a)))


def effort(
    model: str, trial: Trial, r: np.ndarray, lam: float | None = None, values: np.ndarray | None = None
) -> float:
    """Elementary steps of ``model`` on ``trial``. Search models need their sensitivity and values."""
    n, m = trial.n, used_attributes(trial, r)
    if model in ("QR", "CL"):
        return float(n * m)
    if model in ("DbS_avg", "DbS_sum", "MLBA_avg", "MLBA_sum"):
        return float(n * (n - 1) * m)
    if model == "EBA":
        return eba_checks(trial.X, r)
    if model.startswith("SR"):
        return sample_until_return_length(lam, values) * m
    if model == "Accept_read":
        return accept_inspections(lam, values) * m
    raise ValueError(f"no effort rule for {model!r}")
