"""Measures of Section 5 of ANALYSIS_PLAN.md.

Log scores are in bits per choice above a uniform guess. Completeness divides a model's summed log
score over a set of trials by the summed attainable log score, which uses the Miller (1955) correction
so that the true choice probabilities score about 100 percent on samples of the observed size.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_LN2 = np.log(2.0)
_FLOOR = 1e-300


def shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        raise ValueError("a trial without choices has no shares")
    return counts / total


def log_score(counts: np.ndarray, q: np.ndarray) -> float:
    """Bits per choice above a uniform guess, (1/N) sum_i n_i log2(n q_i)."""
    total = counts.sum()
    return float(counts @ np.log(np.maximum(q, _FLOOR)) / total + np.log(q.size)) / _LN2


def entropy_bits(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def attainable(counts: np.ndarray, corrected: bool = True) -> float:
    """Log score of the observed shares, Miller corrected by default."""
    total = counts.sum()
    p = counts / total
    value = np.log2(p.size) - entropy_bits(p)
    if corrected:
        m = int((counts > 0).sum())
        value -= (m - 1) / (2.0 * total * _LN2)
    return float(value)


def completeness(log_scores: Sequence[float], attainables: Sequence[float]) -> float:
    """100 times the summed log score over the summed attainable log score."""
    return 100.0 * float(np.sum(log_scores)) / float(np.sum(attainables))


def overlap(p: np.ndarray, q: np.ndarray) -> float:
    """Share of choices on which observed and predicted distributions agree, sum_i min(p_i, q_i)."""
    return float(np.minimum(p, q).sum())


def effective_number(p: np.ndarray) -> float:
    """Number of equally likely options with the same entropy, exp of the entropy in nats."""
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def sampled_effective_number(q: np.ndarray, n_choices: int, draws: int = 2000, seed: int = 0) -> float:
    """Mean effective number of the shares of ``n_choices`` choices drawn from ``q``.

    The comparison value for observed shares, since a finite sample of choices looks more concentrated
    than the distribution it came from.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(n_choices, q / q.sum(), size=draws) / n_choices
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(samples > 0, np.log(samples), 0.0)
    return float(np.exp(-(samples * logs).sum(axis=1)).mean())
