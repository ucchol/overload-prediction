"""Two look satisficing on three open gaze datasets (VERIFY_BRIEF_GAZE.md, Amendment 6): data and models.

Displays have two or three items, so every quantity is computed exactly at the item level:
  E_SR(p) = sum_k k! e_k(p)                          (expected looks of sample until return)
  P_SR(i) = p_i^2 sum_{k=0}^{n-1} (k+1)! e_k(p_-i)    (race probabilities)
  TLS: rho = softmax(lam v / 2) exp(lam (v - 10) / 2), P = sum rho, E_TLS = E_SR(rho / P) / P,
       and rho / P = softmax(lam v), so TLS chooses as SR.
  Static models choose with softmax(lam s): CL s = v; DbS_avg s_i = sum_{j != i} sign(v_i - v_j) / (n - 1);
  MLBA_avg s_i = sum_{j != i} exp(-|d|) d / (n - 1) with d = 0.6 (v_i - v_j) (kappa = 1).
Reuses ../independent_search/search_core.py (time law, grouped Spearman, summaries).
Nothing here prints or stores identifiers, participant counts or trial counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_search"))
import search_core as srch  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, sc, sys.modules[__name__])

GAZE_DIR = os.path.realpath(os.path.join(VERIFY, "gaze"))
DATASETS = ("krajbich2010", "krajbich2011", "folke2016")
STATIC = ("CL", "DbS", "MLBA")
V_TOP = 10.0
MLBA_SCALE = 0.6
LOG_FLOOR = float(np.log(1e-300))
SR_GRID = np.concatenate([[0.0], np.logspace(-2, 2, 21)])  # Amendment 5 grid
SR_XATOL = 1e-10
DENSE_GRID = np.concatenate([[0.0], np.logspace(-4, 3, 561)])  # global-maximum diagnostic only
N_FOLDS = 10
FOLD_SEED = 20261014
LN2 = np.log(2.0)


# ============================================================================ data
def load_dataset(data_dir: str, name: str, allow_real: bool = False) -> dict:
    """Included trials of one dataset, participants, folds, display features and (Folke) confidence."""
    if os.path.realpath(data_dir) == GAZE_DIR and not allow_real:
        raise PermissionError("the real gaze data are analysed only after the coordinator's go-ahead (--real)")
    df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
    vcols = sorted((c for c in df.columns if c.startswith("item_value_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    n = len(vcols)
    if n < 2 or vcols != [f"item_value_{i}" for i in range(n)]:
        raise ValueError("item value columns must be item_value_0 .. item_value_{n-1}")
    V = df[vcols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ch = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
    rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
    conf_all = None
    if name == "folke2016":
        orig = pd.read_csv(os.path.join(data_dir, "folke2016_original.csv"), usecols=["Conf", "Choice_RT"])
        if len(orig) != len(df):
            raise ValueError("folke2016_original.csv and folke2016.csv have different numbers of rows")
        rt_orig = pd.to_numeric(orig["Choice_RT"], errors="coerce").to_numpy(float)
        if not np.all(np.isclose(rt, rt_orig, rtol=1e-9, atol=0.0, equal_nan=True)):
            raise ValueError("response times of folke2016_original.csv and folke2016.csv differ on some row")
        conf_all = pd.to_numeric(orig["Conf"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore"):
        keep = (np.isfinite(V).all(axis=1) & np.isfinite(ch) & (ch == np.floor(ch)) & (ch >= 0) & (ch <= n - 1)
                & np.isfinite(rt) & (rt > 0))
    V = V[keep]
    if (V > V_TOP).any():
        raise ValueError(f"{name}: an included item value lies above {V_TOP:g}; the analysis stops (Amendment 6 item 1)")
    subj = df["subject"].to_numpy()[keep]
    ids = np.unique(subj)  # sorted participants
    pidx = np.searchsorted(ids, subj)
    P = len(ids)
    perm = np.random.default_rng(FOLD_SEED).permutation(P)  # fresh generator per dataset; position j holds participant perm[j]
    fold_of = np.empty(P, int)
    fold_of[perm] = np.arange(P) % N_FOLDS
    Vs = np.sort(V, axis=1)
    return dict(name=name, n=n, V=V, choice=ch[keep].astype(int), rt_ms=rt[keep], y=np.log(rt[keep] / 1000.0),
                pidx=pidx, P=P, fold=fold_of[pidx], fold_of=fold_of,
                top_value=Vs[:, -1], mean_value=V.mean(axis=1), top_gap=Vs[:, -1] - Vs[:, -2],
                conf=None if conf_all is None else conf_all[keep])


# ============================================================================ scores of the static models
def dbs_scores(V: np.ndarray) -> np.ndarray:
    n = V.shape[1]
    return np.sign(V[:, :, None] - V[:, None, :]).sum(axis=2) / (n - 1)


def mlba_scores(V: np.ndarray, scale: float = MLBA_SCALE) -> np.ndarray:
    n = V.shape[1]
    d = scale * (V[:, :, None] - V[:, None, :])  # the item itself has d = 0 and adds nothing
    return (np.exp(-np.abs(d)) * d).sum(axis=2) / (n - 1)


def static_scores(V: np.ndarray) -> dict:
    return {"CL": V, "DbS": dbs_scores(V), "MLBA": mlba_scores(V)}


# ============================================================================ models
def log_softmax(lam: float, S: np.ndarray) -> np.ndarray:
    z = lam * S
    return z - logsumexp(z, axis=1, keepdims=True)


def esym(p: np.ndarray) -> np.ndarray:
    """f_k = k! e_k(p) of each row, k = 0..n (T, n + 1)."""
    T, n = p.shape
    f = np.zeros((T, n + 1))
    f[:, 0] = 1.0
    k = np.arange(1, n + 1)[None, :]
    for j in range(n):
        f[:, 1:] = f[:, 1:] + k * p[:, j:j + 1] * f[:, :-1]
    return f


def sr_looks_of(p: np.ndarray) -> np.ndarray:
    """E_SR(p) = sum_k k! e_k(p)."""
    return esym(p).sum(axis=1)


def _drop(x: np.ndarray, col: np.ndarray) -> np.ndarray:
    T, n = x.shape
    mask = np.ones((T, n), bool)
    mask[np.arange(T), col] = False
    return x[mask].reshape(T, n - 1)


def sr_log_probs(lam: float, V: np.ndarray) -> np.ndarray:
    """log race probabilities of every item (T, n), in log space."""
    logp = log_softmax(lam, V)
    p = np.exp(logp)
    T, n = V.shape
    k1 = np.arange(1, n + 1)[None, :]
    out = np.empty((T, n))
    for i in range(n):
        g = esym(_drop(p, np.full(T, i)))
        out[:, i] = 2 * logp[:, i] + np.log((g * k1).sum(axis=1))
    return out


def sr_log_chosen(lam: float, V: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """log P_SR(chosen item), floored at log(1e-300)."""
    logp = log_softmax(lam, V)
    p = np.exp(logp)
    g = esym(_drop(p, choice))
    lp = 2 * logp[np.arange(len(choice)), choice] + np.log((g * np.arange(1, V.shape[1] + 1)[None, :]).sum(axis=1))
    return np.maximum(lp, LOG_FLOOR)


def softmax_log_chosen(lam: float, S: np.ndarray, choice: np.ndarray) -> np.ndarray:
    return np.maximum(log_softmax(lam, S)[np.arange(len(choice)), choice], LOG_FLOOR)


def sr_expected_looks(lam: float, V: np.ndarray) -> np.ndarray:
    return sr_looks_of(np.exp(log_softmax(lam, V)))


def tls_expected_looks(lam: float, V: np.ndarray) -> np.ndarray:
    logrho = log_softmax(lam / 2.0, V) + lam * (V - V_TOP) / 2.0
    logP = logsumexp(logrho, axis=1)
    return sr_looks_of(np.exp(logrho - logP[:, None])) * np.exp(-logP)


def entropy_bits(logq: np.ndarray) -> np.ndarray:
    q = np.exp(logq)
    return -(np.where(q > 0, q * logq, 0.0)).sum(axis=1) / LN2


def round10(x: np.ndarray) -> np.ndarray:
    """float(format(x, ".10g")) of every element (Amendment 6 item 6): forecasts are rounded to ten significant
    digits before they are ranked, so that forecasts equal in exact arithmetic tie. Never applied to response
    times or confidence. Each distinct value is formatted once."""
    x = np.asarray(x, float)
    u, inv = np.unique(x, return_inverse=True)
    r = np.array([float(format(v, ".10g")) for v in u.tolist()])
    return r[np.asarray(inv).reshape(-1)].reshape(x.shape)


# ============================================================================ fits
def fit_softmax(S: np.ndarray, choice: np.ndarray, w: np.ndarray, init: float | None = None, rtol: float = 1e-13) -> float:
    """Weighted ML of lambda >= 0 for softmax(lambda s): safeguarded Newton root finding on the score of the
    concave likelihood. lambda = 0 when the score at zero is not positive."""
    idx = np.flatnonzero(w > 0)
    S, choice, w = S[idx], choice[idx], w[idx]
    sc_ = S[np.arange(len(choice)), choice]

    def score(lam):
        q = np.exp(log_softmax(lam, S))
        m = (q * S).sum(axis=1)
        var = (q * (S - m[:, None]) ** 2).sum(axis=1)
        return float(w @ (sc_ - m)), -float(w @ var)

    g0, _ = score(0.0)
    spread = float(w @ (S.max(axis=1) - S.min(axis=1)))
    if spread <= 0 or g0 <= 1e-13 * spread:
        return 0.0
    lo, hi = 0.0, (1.5 * init if init else 1.0)
    while score(hi)[0] > 0:
        lo, hi = hi, 2 * hi
        if hi > 1e8:
            raise RuntimeError("lambda diverges")
    x = init if (init and lo < init < hi) else 0.5 * (lo + hi)
    for _ in range(300):
        g, h = score(x)
        if g > 0:
            lo = x
        elif g < 0:
            hi = x
        step = -g / h if h < 0 else np.inf
        if np.isfinite(step) and abs(step) <= rtol * abs(x):
            return x + step
        xn = x + step
        if not (lo < xn < hi):
            xn = 0.5 * (lo + hi)
        if hi - lo <= 4 * np.finfo(float).eps * max(abs(xn), 1e-300):
            return xn
        x = xn
    raise RuntimeError("softmax fit did not converge")


def fit_cl(V, choice, w, init=None):
    return fit_softmax(V, choice, w, init)


class SRFit:
    """SR's lambda by the procedure of Amendment 6 item 3: the grid of zero and 21 values spaced evenly in log
    between 0.01 and 100, the best grid value refined by bounded Brent search (absolute tolerance 1e-10)
    between its neighbours. The log likelihood on the grid and on a dense diagnostic grid is tabulated per
    trial once (weight free)."""

    def __init__(self, V: np.ndarray, choice: np.ndarray):
        self.V, self.choice = V, choice
        self.tab = np.array([sr_log_chosen(x, V, choice) for x in SR_GRID])
        self.tab_dense = np.array([sr_log_chosen(x, V, choice) for x in DENSE_GRID])

    def loglik(self, lam: float, w: np.ndarray) -> float:
        idx = np.flatnonzero(w > 0)
        return float(w[idx] @ sr_log_chosen(lam, self.V[idx], self.choice[idx]))

    def fit(self, w: np.ndarray):
        LLg = self.tab @ w
        g = int(np.argmax(LLg))
        lo, hi = SR_GRID[max(g - 1, 0)], SR_GRID[min(g + 1, len(SR_GRID) - 1)]
        res = minimize_scalar(lambda x: -self.loglik(x, w), bounds=(lo, hi), method="bounded",
                              options={"xatol": SR_XATOL, "maxiter": 2000})
        lam, ll = float(res.x), -float(res.fun)
        dense_excess = float((self.tab_dense @ w).max() - ll)  # > 0 would mean a better maximum elsewhere
        return lam, {"ll": ll, "dense_excess": dense_excess, "grid_index": g}
