"""Search models, global lambda fits, effort, time laws and forecasts (VERIFY_BRIEF_PROCESS.md).

Builds on the R0 implementation in ../independent/core.py (stimuli, rubric shares, values, counts,
softmax and EBA fits, measures). Nothing here prints or stores identifiers or counts.

Sample until return (SR). q = softmax(lambda v), looks i.i.d. from q, the first option looked at twice
is chosen. Exact computation by Gauss-Laguerre quadrature of
    P(i) = q_i^2 int_0^inf t e^-t prod_{j != i} (1 + q_j t) dt,  E[looks] = int_0^inf e^-t prod_j (1 + q_j t) dt.
Both integrands are polynomials of degree n, so M nodes with 2M - 1 >= n integrate them exactly; every
term is positive, so the sums are accurate. They are done in log space. sr_dp below is an independent
exact version through k! e_k(q), used in the checks.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import brentq, nnls
from scipy.special import logsumexp
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent"))
import core  # noqa: E402  (the R0 implementation)


def assert_local(*modules):
    """Independence guard: every module used must be loaded from the verify folder."""
    root = os.path.realpath(VERIFY) + os.sep
    for mod in modules:
        if not os.path.realpath(mod.__file__).startswith(root):
            raise ImportError(f"module {mod.__name__} loaded from outside the verify folder")


assert_local(core, sys.modules[__name__])

GL_M = 24  # exact for n <= 47
_T, _W = laggauss(GL_M)
LOG_T, LOG_W = np.log(_T), np.log(_W)
NEG = -1e200  # finite stand-in for log 0 in grid tables
GRID = np.concatenate([[0.0], np.logspace(-4, 4, 481)])  # 60 points per decade
SEARCH_MODELS = ("SR", "SR_DbS", "SR_MLBA", "Accept_read")
SEARCH_VALUES = {"SR": "CL|beta=+0.0", "SR_DbS": "DbS_avg|beta=+0.0", "SR_MLBA": "MLBA_avg|beta=+0.0", "Accept_read": "CL|beta=+0.0"}


# ============================================================================ sample until return
def sr_batch(lam: float, V: np.ndarray, mask: np.ndarray, need_grad: bool = True):
    """log P(i), d log P(i)/d lambda and E[looks] for trials stacked as rows (padded options masked)."""
    Vm = np.where(mask, V, 0.0)
    z = np.where(mask, lam * V, -np.inf)
    logq = z - logsumexp(z, axis=1, keepdims=True)
    q = np.exp(logq)
    qt = q[:, :, None] * _T[None, None, :]
    l1p = np.log1p(qt)  # (T, I, M)
    logF = l1p.sum(axis=1)  # (T, M)
    logE = logsumexp(LOG_W[None] + logF, axis=1)
    logT = (LOG_W + LOG_T)[None, None, :] + logF[:, None, :] - l1p
    tmax = logT.max(axis=2, keepdims=True)
    e = np.exp(logT - tmax)
    s = e.sum(axis=2)
    logP = np.where(mask, 2 * logq + np.log(s) + tmax[..., 0], -np.inf)
    if not need_grad:
        return logP, None, np.exp(logE)
    vbar = (q * Vm).sum(axis=1, keepdims=True)
    dq = q * (Vm - vbar)
    frac = _T[None, None, :] / (1.0 + qt)  # t / (1 + q t)
    D = (dq[:, :, None] * frac).sum(axis=1)  # d log F / d lambda, (T, M)
    dlogT = D[:, None, :] - dq[:, :, None] * frac
    dlogP = np.where(mask, 2 * (Vm - vbar) + (e * dlogT).sum(axis=2) / s, 0.0)
    return logP, dlogP, np.exp(logE)


def sr_dp(q: np.ndarray):
    """Exact SR probabilities and E[looks] through f_k = k! e_k(q) (independent of the quadrature)."""
    q = np.asarray(q, float)
    n = len(q)
    f = np.zeros(n + 1)
    f[0] = 1.0
    for j in range(n):
        f[1:] = f[1:] + np.arange(1, n + 1) * q[j] * f[:-1]
    P = np.zeros(n)
    for i in range(n):
        g = np.zeros(n)
        g[0] = 1.0
        for j in range(n):
            if j != i:
                g[1:] = g[1:] + np.arange(1, n) * q[j] * g[:-1]
        P[i] = q[i] ** 2 * np.sum(np.arange(1, n + 1) * g)  # (k+1)! e_k = (k+1) g_k
    return P, f.sum()


# ============================================================================ accept in reading order
def acc_batch(lam: float, V: np.ndarray, mask: np.ndarray, need_grad: bool = True):
    """log P(i), d log P(i)/d lambda and expected inspections for accept in reading order."""
    vmax = np.where(mask, V, -np.inf).max(axis=1, keepdims=True)
    d = np.where(mask, V - vmax, 0.0)  # <= 0
    x = lam * d
    log_a = -np.logaddexp(0.0, -x)
    log_1ma = np.where(mask, -np.logaddexp(0.0, x), 0.0)  # padded options sit at the end and count as 1
    cum = np.cumsum(log_1ma, axis=1) - log_1ma  # sum over j < i
    logR = log_1ma.sum(axis=1, keepdims=True)
    log1mR = np.log(-np.expm1(logR))
    logP = np.where(mask, log_a + cum - log1mR, -np.inf)
    insp = np.exp(logsumexp(np.where(mask, cum, -np.inf), axis=1) - log1mR[:, 0])
    if not need_grad:
        return logP, None, insp
    a = np.exp(log_a)
    ad = np.where(mask, a * d, 0.0)
    cum_ad = np.cumsum(ad, axis=1) - ad
    R = np.exp(logR)
    dlogP = np.where(mask, (1 - a) * d - cum_ad - R * ad.sum(axis=1, keepdims=True) / (1 - R), 0.0)
    return logP, dlogP, insp


BATCH = {"SR": sr_batch, "SR_DbS": sr_batch, "SR_MLBA": sr_batch, "Accept_read": acc_batch}


# ============================================================================ global fits
class SearchModel:
    """One search model on one task: padded values, and log P on the lambda grid (choice free)."""

    def __init__(self, task: dict, name: str):
        self.name = name
        self.fn = BATCH[name]
        m = core.SOFTMAX_MODELS.index(SEARCH_VALUES[name])
        self.V = task["V"][m]
        self.mask = task["valid"]
        self.nvec = task["nvec"]
        table = np.zeros((len(GRID),) + self.V.shape)
        for g, lam in enumerate(GRID):
            table[g] = self.fn(lam, self.V, self.mask, need_grad=False)[0]
        self.grid_logP = np.where(np.isfinite(table), table, NEG)  # (G, C, I)

    def loglik(self, lam, cnt, trials):
        logP, _, _ = self.fn(lam, self.V[trials], self.mask[trials], need_grad=False)
        return float(np.where(cnt[trials] > 0, cnt[trials] * np.where(np.isfinite(logP), logP, NEG), 0.0).sum())

    def score(self, lam, cnt, trials):
        _, dlogP, _ = self.fn(lam, self.V[trials], self.mask[trials])
        return float((cnt[trials] * dlogP).sum())

    def fit(self, cnt: np.ndarray, trials: np.ndarray) -> tuple[float, dict]:
        """Global ML of lambda >= 0 on the pooled counts of ``trials``.

        The log likelihood is evaluated on a fixed grid (0 and 1e-4 to 1e4, 60 points per decade).
        Every local maximum of the grid values is refined by Brent's method on the analytic score
        within its two neighbours, and the best refined point is kept. If the grid maximum sits at
        the top of the grid, the search continues by doubling until the likelihood falls.
        """
        LLg = np.einsum("gci,ci->g", self.grid_logP[:, trials], cnt[trials])
        lams = GRID
        if int(np.argmax(LLg)) == len(GRID) - 1:  # extend upwards
            ext, vals = [GRID[-1]], [LLg[-1]]
            while len(ext) < 60:
                ext.append(ext[-1] * 2.0)
                vals.append(self.loglik(ext[-1], cnt, trials))
                if vals[-1] < vals[-2]:
                    break
            else:
                raise RuntimeError(f"{self.name}: likelihood still rising at lambda {ext[-1]:.3g}")
            lams = np.concatenate([GRID, ext[1:]])
            LLg = np.concatenate([LLg, vals[1:]])
        G = len(lams)
        cands = [g for g in range(G - 1) if (g == 0 or LLg[g] >= LLg[g - 1]) and LLg[g] >= LLg[g + 1]]
        best_lam, best_ll, refined_best = None, -np.inf, True
        for g in cands:
            refined = True
            if g == 0 and self.score(0.0, cnt, trials) <= 0:
                lam = 0.0
            else:
                lo = lams[g - 1] if g > 0 else 0.0
                hi = lams[g + 1]
                s_lo, s_hi = self.score(lo, cnt, trials), self.score(hi, cnt, trials)
                if s_lo > 0 and s_hi < 0:
                    lam = brentq(lambda x: self.score(x, cnt, trials), lo, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=200)
                else:  # no sign change between the neighbours: keep the grid point and flag it
                    lam, refined = lams[g], False
            ll = self.loglik(lam, cnt, trials)
            if ll > best_ll:
                best_lam, best_ll, refined_best = lam, ll, refined
        return best_lam, {"loglik": best_ll, "grid_max": float(LLg.max()), "n_local": len(cands), "refined": refined_best}

    def predict(self, lam: float, c: int):
        """Probabilities and E[looks] (SR family) or expected inspections (Accept_read) for trial c."""
        logP, _, extra = self.fn(lam, self.V[c:c + 1], self.mask[c:c + 1], need_grad=False)
        return np.where(self.mask[c], np.exp(logP[0]), 0.0), float(extra[0])


def search_heldout(models: dict, cnt: np.ndarray) -> dict:
    """Leave-one-trial-out fits of every search model; held-out probabilities and expected looks.

    Also returns per fold diagnostics: number of grid local maxima and whether the optimum was refined.
    """
    C = cnt.shape[0]
    out = {}
    for name, sm in models.items():
        lam = np.zeros(C)
        Q = np.zeros(sm.V.shape)
        extra = np.zeros(C)
        n_local = np.zeros(C, int)
        refined = np.zeros(C, bool)
        for h in range(C):
            lam[h], info = sm.fit(cnt, np.array([c for c in range(C) if c != h]))
            n_local[h], refined[h] = info["n_local"], info["refined"]
            Q[h], extra[h] = sm.predict(lam[h], h)
        out[name] = dict(lam=lam, Q=Q, extra=extra, n_local=n_local, refined=refined)
    return out


# ============================================================================ effort
def differing_shared(task: dict, c: int) -> np.ndarray:
    """Attributes that differ between the trial's options and have a positive rubric share."""
    X = task["trials"][c]["X"]
    return np.where((task["r"] > 0) & (np.ptp(X, axis=0) > 0))[0]


def eba_checks(H: np.ndarray, w: np.ndarray) -> float:
    """Expected elementary checks of EBA on an aspect matrix H (options x aspects), weights w."""
    A, W = H.shape[1], w.sum()
    memo: dict = {}

    def checks(S: tuple) -> float:
        if len(S) == 1:
            return 0.0
        if S in memo:
            return memo[S]
        held = H[list(S)].sum(axis=0)
        split = (held > 0) & (held < len(S))
        if not split.any():
            val = len(S) * A
        else:
            Ws = w[split].sum()
            val = len(S) * W / Ws + sum((w[a] / Ws) * checks(tuple(i for i in S if H[i, a])) for a in np.where(split)[0])
        memo[S] = val
        return val

    return checks(tuple(range(H.shape[0])))


def fixed_efforts(task: dict) -> dict:
    """Efforts that do not depend on the fits: m, QR and CL n m, DbS and MLBA n(n-1)m, EBA checks."""
    C = len(task["trials"])
    m = np.zeros(C)
    eba = np.zeros(C)
    for c, t in enumerate(task["trials"]):
        ks = differing_shared(task, c)
        m[c] = len(ks)
        Xk = t["X"][:, ks]
        eba[c] = eba_checks(Xk >= np.median(Xk, axis=0), task["r"][ks]) if len(ks) else 0.0
    n = task["nvec"]
    return {"m": m, "n": n, "QR": n * m, "CL": n * m, "DbS_avg": n * (n - 1) * m, "MLBA_avg": n * (n - 1) * m, "EBA": eba}


# ============================================================================ time law and forecasts
def time_law(effort: np.ndarray, seconds: np.ndarray):
    """Leave-one-trial-out NNLS of seconds on [1, effort]; predictions and log errors."""
    C = len(seconds)
    pred = np.zeros(C)
    for j in range(C):
        keep = np.arange(C) != j
        coef, _ = nnls(np.column_stack([np.ones(C - 1), effort[keep]]), seconds[keep])
        pred[j] = coef[0] + coef[1] * effort[j]
    return pred, np.abs(np.log(pred / seconds))


def constant_law(seconds: np.ndarray):
    C = len(seconds)
    pred = np.array([seconds[np.arange(C) != j].mean() for j in range(C)])
    return pred, np.abs(np.log(pred / seconds))


def typical_error(logerr) -> float:
    return float(100 * (np.exp(np.mean(logerr)) - 1))


def spearman0(f: np.ndarray, target: np.ndarray) -> float:
    """Spearman correlation (average ranks for ties), 0 when the forecast is constant."""
    return 0.0 if np.ptp(f) == 0 else float(spearmanr(f, target).correlation)


def constant_error(target: np.ndarray) -> float:
    """Mean over trials of |mean of the other trials' targets - target|."""
    C = len(target)
    return float(np.mean([abs(target[np.arange(C) != j].mean() - target[j]) for j in range(C)]))


def forecast_stats(f: np.ndarray, target: np.ndarray) -> dict:
    """Spearman correlation, held-out linear calibration error, constant error and their difference."""
    C = len(target)
    err = np.zeros(C)
    for j in range(C):
        keep = np.arange(C) != j
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(C - 1), f[keep]]), target[keep], rcond=None)
        err[j] = abs(coef[0] + coef[1] * f[j] - target[j])
    ce = constant_error(target)
    return {"spearman": spearman0(f, target), "error": float(err.mean()), "constant_error": ce, "error_minus_constant": float(err.mean() - ce)}


def regret_stats(f: np.ndarray, target: np.ndarray) -> dict:
    """Direct regret forecast: mean absolute error (no calibration), constant error, difference, Spearman."""
    err = float(np.mean(np.abs(f - target)))
    ce = constant_error(target)
    return {"error": err, "constant_error": ce, "error_minus_constant": err - ce, "spearman": spearman0(f, target)}


def entropy_bits(q: np.ndarray) -> float:
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())
