"""Two look satisficing (TLS) on the snack data (VERIFY_BRIEF_TWO_LOOK.md), level form.

Looks fall on item i with probability q_i, with replacement; a look at i registers with probability h_i;
the first item with two registered looks is chosen. With rho_i = q_i h_i and P = sum rho:
  E[looks]    = E_SR(rho / P) / P                       (Wald: each registered look costs 1/P looks)
  E[distinct] = sum_i q_i int e^{-(q_i + P - rho_i) t} prod_{j != i} (1 + rho_j t) dt
                (item i is looked at iff its first look comes before the race of the others stops)
TLS: q = softmax(lam v / 2), h = exp(lam (v - 3) / 2), so rho / P = softmax(lam v) and TLS chooses like SR.
Uniform attention (sensitivity): q = 1 / n, h = exp(lam (v - 3)).
Integrals of polynomials (degree < n <= 36) by Gauss-Laguerre with 24 nodes, exact to degree 47.
Reuses ../independent_search/search_core.py (data, SR fit, time law, grouped Spearman, summaries).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_search"))
import search_core as srch  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, sc, sys.modules[__name__])

LEV = srch.LEV
V_TOP = 3.0
_T, LOG_W = srch._T, srch.LOG_W


def _log_counts(C):
    with np.errstate(divide="ignore"):
        return np.log(C)


def tls_log_qh(lam: float, C: np.ndarray):
    """TLS attention q = softmax(lam v / 2) over items and registration h = exp(lam (v - 3) / 2), per level."""
    logq = srch.sr_logq(lam / 2.0, C)
    logh = np.broadcast_to(lam * (LEV - V_TOP) / 2.0, C.shape)
    return logq, logh


def uniform_log_qh(lam: float, C: np.ndarray, n: np.ndarray):
    """Uniform attention q = 1 / n and registration h = exp(lam (v - 3)), per level."""
    logq = np.where(C > 0, -np.log(n)[:, None], -np.inf)
    logh = np.broadcast_to(lam * (LEV - V_TOP), C.shape)
    return logq, logh


def two_look_expectations(logq: np.ndarray, logh: np.ndarray, C: np.ndarray, need_distinct: bool = True):
    """E[looks] and E[distinct items looked at] per trial (level form, exact quadrature)."""
    logrho = np.where(C > 0, logq + logh, -np.inf)
    logP = logsumexp(logrho + _log_counts(C), axis=1)
    p = np.exp(logrho - logP[:, None])  # rho / P, zero for absent levels
    logF = np.einsum("tl,tlm->tm", C, np.log1p(p[:, :, None] * _T[None, None, :]))
    E_looks = np.exp(logsumexp(LOG_W[None, :] + logF, axis=1) - logP)
    if not need_distinct:
        return E_looks, None
    q, rho, P = np.exp(logq), np.exp(logrho), np.exp(logP)
    a = q + P[:, None] - rho
    distinct = np.zeros(len(C))
    for l in range(7):
        idx = np.flatnonzero(C[:, l] > 0)
        if len(idx) == 0:
            continue
        al = a[idx, l]
        L = np.log1p(rho[idx][:, None, :] * _T[None, :, None] / al[:, None, None])  # (T, M, 7)
        logG = np.einsum("tk,tmk->tm", C[idx], L) - L[:, :, l]  # prod over the other items
        logI = logsumexp(LOG_W[None, :] + logG, axis=1) - np.log(al)
        distinct[idx] += C[idx, l] * np.exp(logq[idx, l] + logI)
    return E_looks, distinct


# ============================================================================ independent exact routes (checks)
def _fk(x: np.ndarray) -> np.ndarray:
    """f_k = k! e_k(x), k = 0..len(x)."""
    n = len(x)
    f = np.zeros(n + 1)
    f[0] = 1.0
    for j in range(n):
        f[1:] = f[1:] + np.arange(1, n + 1) * x[j] * f[:-1]
    return f


def exact_item_level(q: np.ndarray, h: np.ndarray):
    """Item-level exact E[looks], E[distinct] and choice probabilities through elementary symmetric polynomials.

    int_0^inf e^{-a t} prod_j (1 + r_j t) dt = sum_k k! e_k(r) / a^{k+1};
    P(choose i) = rho_i^2 int t e^{-P t} prod_{j != i}(1 + rho_j t) dt = rho_i^2 sum_k (k+1)! e_k(rho_-i) / P^{k+2}.
    """
    q, h = np.asarray(q, float), np.asarray(h, float)
    rho = q * h
    P = rho.sum()
    E_looks = _fk(rho / P).sum() / P
    distinct, choice = 0.0, np.zeros(len(q))
    for i in range(len(q)):
        others = np.delete(rho, i)
        a = q[i] + P - rho[i]
        distinct += q[i] * _fk(others / a).sum() / a
        g = _fk(others / P)
        choice[i] = (rho[i] / P) ** 2 * np.sum(np.arange(1, len(q) + 1) * g)
    return E_looks, distinct, choice


def simulate_two_look(q: np.ndarray, h: np.ndarray, rng: np.random.Generator, max_steps: int = 1_000_000):
    """Simulate the process on trials (rows): choice, number of looks, distinct items looked at, looks per item."""
    T, n = q.shape
    cum = np.cumsum(q, axis=1)
    reg = np.zeros((T, n), int)
    looks_per_item = np.zeros((T, n))
    choice = np.full(T, -1)
    nlooks = np.zeros(T, int)
    live = np.arange(T)
    step = 0
    while live.size and step < max_steps:
        step += 1
        u = rng.random(live.size)
        item = np.minimum((u[:, None] > cum[live]).sum(axis=1), n - 1)
        looks_per_item[live, item] += 1
        r = rng.random(live.size) < h[live, item]
        reg[live[r], item[r]] += 1
        done = r & (reg[live, item] >= 2)
        choice[live[done]] = item[done]
        nlooks[live[done]] = step
        live = live[~done]
    if live.size:
        raise RuntimeError("simulation did not finish")
    return choice, nlooks, (looks_per_item > 0).sum(axis=1), looks_per_item
