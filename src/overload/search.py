"""Search models, in which a choice is the end of a search rather than the readout of every value.

The established models (QR, CL, DbS, MLBA) say how attractive each option is and turn every value into a
choice probability at once. Nothing in them says how long a decision takes, or how hard it feels. A search
model adds the missing half. Attention moves from option to option, and the decision ends when a stopping
rule is met, so the same mechanism yields the choice, the number of steps taken and how contested the
decision was.

Sample until return (the proposed model). Attention is drawn to attractive options. At each step the next
option looked at is drawn with probability proportional to exp(lambda v), so attractive options are looked
at more often, and any option can be looked at again. The decision ends when attention returns to an option
already looked at, and that option is chosen. In the language of evidence, the first option to collect two
looks wins a race. The one free parameter is lambda, the sensitivity of attention to value, exactly as in
CL, and v are the same rubric based values.

Everything follows in closed form. Let q = softmax(lambda v). Embedding the looks in a Poisson process of
unit rate, option i collects looks at rate q_i, and

    P(i chosen)       = q_i^2 * sum_k (k + 1)! e_k(q without i),
    E[looks taken]    = sum_k k! e_k(q),

where e_k is the k-th elementary symmetric polynomial. Every look before the last falls on a new option, so
the number of distinct options looked at is the number of looks minus one.

What it implies. When one option stands out, attention keeps coming back to it and the race ends after two
or three looks. When many options are about equally attractive, looks spread over them and the race runs
long, which is where the model expects slow decisions and overload. Choice probabilities are sharper than
CL's for poor options (close to q_i squared) and similar for the best ones.

Accept in reading order (a rejected alternative, kept for the record). Options are inspected in the order
listed, each accepted with probability 1 / (1 + exp(-lambda (v_i - v_max))), and passes repeat until one is
accepted. It implies strong primacy, and it predicted unseen trials far worse than CL (STORY.md).
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy.special import expit, softmax


def scaled_symmetric(q: np.ndarray) -> np.ndarray:
    """f_k = k! e_k(q) for k = 0 .. n, where e_k is the k-th elementary symmetric polynomial.

    Built by the recursion f_k <- f_k + x k f_(k-1) over the entries x of q. With q summing to one every f_k
    lies between 0 and 1, so no factorial is ever formed.
    """
    n = q.size
    k = np.arange(1, n + 1, dtype=float)
    f = np.zeros(n + 1)
    f[0] = 1.0
    for x in q:
        f[1:] = f[1:] + x * k * f[:-1]
    return f


def scaled_symmetric_without(q: np.ndarray) -> np.ndarray:
    """Row i holds f_0 .. f_(n-1) of q with entry i left out."""
    n = q.size
    k = np.arange(1, n, dtype=float)
    F = np.zeros((n, n))
    F[:, 0] = 1.0
    for j, x in enumerate(q):
        upd = F.copy()
        upd[:, 1:] = F[:, 1:] + x * k * F[:, :-1]
        upd[j] = F[j]  # option j does not enter its own row
        F = upd
    return F


def race_probabilities_exact(q: np.ndarray) -> np.ndarray:
    """Probability that each option is the first to collect two looks, by the exact polynomial sums.

    P(i) = q_i^2 sum_k (k + 1)! e_k(q without i) = q_i^2 sum_k (k + 1) f_k(q without i). The reference
    for ``race_probabilities``, which computes the same sums faster by quadrature.
    """
    q = np.asarray(q, dtype=float) / np.sum(q)
    F = scaled_symmetric_without(q)
    p = q**2 * (F * np.arange(1, q.size + 1, dtype=float)).sum(axis=1)
    return p / p.sum()


def race_length_exact(q: np.ndarray) -> float:
    """Expected number of looks until some option has been looked at twice, sum_k k! e_k(q), exactly."""
    q = np.asarray(q, dtype=float) / np.sum(q)
    return float(scaled_symmetric(q).sum())


@cache
def _laguerre(m: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.laguerre.laggauss(m)


_QUADRATURE_MAX_OPTIONS = 64


def _race(q: np.ndarray) -> tuple[np.ndarray, float]:
    """Race probabilities and expected length by Gauss-Laguerre quadrature.

    Both are integrals over the Poisson time t of e^-t times a polynomial of degree at most n + 1,
    P(i) = q_i^2 int t e^-t prod_(j != i) (1 + q_j t) dt and E[looks] = int e^-t prod_j (1 + q_j t) dt,
    so a rule with n / 2 + 8 or more nodes is exact up to rounding. Every term is positive, so nothing
    cancels, and the results agree with the exact sums to about 1e-14. Above 64 options the exact sums
    are used, since quadrature rules with many nodes are themselves inaccurate.
    """
    q = np.asarray(q, dtype=float) / np.sum(q)
    if q.size > _QUADRATURE_MAX_OPTIONS:
        return race_probabilities_exact(q), race_length_exact(q)
    x, w = _laguerre(max(24, q.size // 2 + 8))
    logs = np.log1p(np.outer(x, q))  # nodes by options
    total = logs.sum(axis=1)
    p = q**2 * ((w * x)[:, None] * np.exp(total[:, None] - logs)).sum(axis=0)
    return p / p.sum(), float((w * np.exp(total)).sum())


def race_probabilities(q: np.ndarray) -> np.ndarray:
    """Probability that each option is the first to collect two looks, looks drawn with probabilities q."""
    return _race(q)[0]


def race_length(q: np.ndarray) -> float:
    """Expected number of looks until some option has been looked at twice."""
    return _race(q)[1]


def sample_until_return(lam: float, v: np.ndarray) -> np.ndarray:
    """Choice probabilities of the proposed model for values v and sensitivity lam."""
    return race_probabilities(softmax(lam * np.asarray(v, dtype=float)))


def sample_until_return_length(lam: float, v: np.ndarray) -> float:
    """Expected number of looks of the proposed model."""
    return race_length(softmax(lam * np.asarray(v, dtype=float)))


def accept_in_reading_order(lam: float, v: np.ndarray) -> np.ndarray:
    """Choice probabilities of the rejected alternative, inspection in listed order with repeated passes."""
    v = np.asarray(v, dtype=float)
    a = expit(lam * (v - v.max()))
    reach = np.concatenate([[1.0], np.cumprod(1.0 - a)[:-1]])
    p = a * reach
    return p / p.sum()
