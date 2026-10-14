"""Sample-until-return (SR) search model on the snack data (VERIFY_BRIEF_SEARCH.md, Amendment 5).

Reuses the independent snack implementation (../independent_snack/snack_core.py) for trial inclusion,
the level-count representation, the static models (CL, DbS_avg, MLBA_avg) and the summaries.
Nothing here prints or stores identifiers, participant counts or trial counts.

Level form. Ratings are integers -3..3 and items sharing a rating share their look probability q, so
every SR quantity depends on a trial only through the counts C_l of displayed items at the 7 levels:
    prod_j (1 + q_j t) = prod_l (1 + q_l t)^{C_l}.
P(item at level l) = q_l^2 int t e^-t prod_l' (1 + q_l' t)^{C_l'} / (1 + q_l t) dt and
E[looks] = int e^-t prod_l (1 + q_l t)^{C_l} dt are integrals of polynomials of degree at most n,
done exactly by Gauss-Laguerre quadrature with 24 nodes (exact to degree 47; displays have <= 36 items).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_snack"))
import snack_core as sc  # noqa: E402


def assert_local(*modules):
    """Independence guard: every module used must be loaded from the verify folder."""
    root = os.path.realpath(VERIFY) + os.sep
    for mod in modules:
        if not os.path.realpath(mod.__file__).startswith(root):
            raise ImportError(f"module {mod.__name__} loaded from outside the verify folder")


assert_local(sc, sys.modules[__name__])

REAL_DATA = os.path.realpath(os.path.join(VERIFY, "snack"))
SET_SIZES = sc.SET_SIZES
NS = np.array(SET_SIZES, float)
LOGN = np.log(NS)
LEV = np.arange(-3.0, 4.0)
STATIC = ("CL", "DbS_avg", "MLBA_avg")
MODELS = STATIC + ("SR",)
GL_M = 24
_T, _W = laggauss(GL_M)
LOG_T, LOG_W = np.log(_T), np.log(_W)
GRID = np.concatenate([[0.0], np.logspace(-4, 2, 361)])  # 60 points per decade
T0_MIN = 1e-9
P_CLIP = 1e-15


# ============================================================================ data
def load_search_data(data_dir: str, allow_real: bool = False) -> dict:
    """Trials of the four set sizes with the inclusion rule of Amendment 3 and response times."""
    if os.path.realpath(data_dir) == REAL_DATA and not allow_real:
        raise PermissionError("the real snack data are analysed only after the coordinator's go-ahead (--real)")
    subjects, choices, Vs, Ls, rts = [], [], [], [], []
    for n in SET_SIZES:
        df = pd.read_csv(os.path.join(data_dir, f"setsize-{n}_desc-data.csv"))
        V = df[[f"item_value_{i}" for i in range(n)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        G = df[[f"cumulative_gaze_{i}" for i in range(n)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        G = np.where(np.isfinite(G), G, 0.0)  # a missing gaze value counts as zero
        ch = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
        rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
        with np.errstate(invalid="ignore"):
            keep = np.isfinite(ch) & (ch == np.floor(ch)) & (ch >= 0) & (ch <= n - 1) & np.isfinite(V).all(axis=1)
        subjects.append(df["subject"].to_numpy()[keep])
        choices.append(ch[keep].astype(int))
        Vs.append(V[keep])
        Ls.append(G[keep] > 0)
        rts.append(rt[keep])
    data = sc.build_data(subjects, choices, Vs, Ls)
    return add_search_fields(data, np.concatenate(rts), np.concatenate([L.sum(axis=1) for L in Ls]).astype(float))


def add_search_fields(data: dict, rt_ms: np.ndarray, n_looked: np.ndarray) -> dict:
    T = len(data["choice"])
    C = data["C_all"]
    data["rt_ms"] = rt_ms
    with np.errstate(invalid="ignore"):
        data["timed"] = np.isfinite(rt_ms) & (rt_ms > 0)
    data["log_rt"] = np.where(data["timed"], np.log(np.where(data["timed"], rt_ms, 1000.0) / 1000.0), np.nan)  # seconds
    data["n_looked"] = n_looked
    top = 6 - np.argmax((C > 0)[:, ::-1], axis=1)  # highest rating level present
    data["top_level"] = top
    data["top_rating"] = LEV[top]
    data["top_count"] = C[np.arange(T), top]
    mean = (C * LEV).sum(axis=1) / data["n"]
    data["rating_sd"] = np.sqrt((C * (LEV[None, :] - mean[:, None]) ** 2).sum(axis=1) / data["n"])  # ddof 0
    data["y_top"] = (data["lc"] == top).astype(float)
    return data


# ============================================================================ SR, level form
def _log_counts(C):
    with np.errstate(divide="ignore"):
        return np.log(C)


def sr_logq(lam: float, C: np.ndarray) -> np.ndarray:
    """log look probability of one item at each rating level (T, 7); -inf for absent levels."""
    z = lam * LEV
    logZ = logsumexp(z[None, :] + _log_counts(C), axis=1, keepdims=True)
    return np.where(C > 0, z[None, :] - logZ, -np.inf)


def _sr_parts(lam, C):
    logq = sr_logq(lam, C)
    q = np.exp(logq)
    qt = q[:, :, None] * _T[None, None, :]
    l1p = np.log1p(qt)  # (T, 7, M)
    logF = np.einsum("tl,tlm->tm", C, l1p)
    return logq, q, qt, l1p, logF


def sr_expected_looks(lam: float, C: np.ndarray) -> np.ndarray:
    logF = _sr_parts(lam, C)[4]
    return np.exp(logsumexp(LOG_W[None, :] + logF, axis=1))


def sr_probs(lam: float, C: np.ndarray):
    """Normalised probability that a given item of each level is chosen (T, 7), and E[looks] (T,)."""
    logq, q, qt, l1p, logF = _sr_parts(lam, C)
    logI = logsumexp((LOG_W + LOG_T)[None, None, :] + logF[:, None, :] - l1p, axis=2)
    logP = np.where(C > 0, 2 * logq + logI, -np.inf)
    lognorm = logsumexp(logP + _log_counts(C), axis=1, keepdims=True)
    P = np.where(C > 0, np.exp(logP - lognorm), 0.0)
    return P, np.exp(logsumexp(LOG_W[None, :] + logF, axis=1))


def sr_chosen(lam: float, C: np.ndarray, lc: np.ndarray, need_grad: bool = True):
    """log P(chosen item) and its derivative in lambda. The exact probabilities sum to one, so no
    normalisation is applied here (the quadrature is exact; tests show sums within 1e-13)."""
    logq, q, qt, l1p, logF = _sr_parts(lam, C)
    r = np.arange(len(lc))
    a = (LOG_W + LOG_T)[None, :] + logF - l1p[r, lc]
    amax = a.max(axis=1, keepdims=True)
    e = np.exp(a - amax)
    s = e.sum(axis=1)
    logP = 2 * logq[r, lc] + np.log(s) + amax[:, 0]
    if not need_grad:
        return logP, None
    vbar = (C * q * LEV[None, :]).sum(axis=1)
    dq = q * (LEV[None, :] - vbar[:, None])
    frac = _T[None, None, :] / (1.0 + qt)
    D = np.einsum("tl,tlm->tm", C * dq, frac)  # d log F / d lambda
    dterm = D - dq[r, lc][:, None] * frac[r, lc]
    dlogP = 2 * (LEV[lc] - vbar) + (e * dterm).sum(axis=1) / s
    return logP, dlogP


def sr_exact(q: np.ndarray):
    """Exact SR item probabilities and E[looks] through k! e_k(q) (independent of the quadrature)."""
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
        P[i] = q[i] ** 2 * np.sum(np.arange(1, n + 1) * g)
    return P, f.sum()


class SRFitter:
    """Global maximum likelihood of SR's lambda >= 0 on weighted trials.

    log P(chosen) and its derivative are tabulated once for every trial on a lambda grid (0 and 1e-4
    to 1e2, 60 points per decade); they do not depend on the weights. For given weights the grid log
    likelihood is a matrix-vector product. Every grid local maximum is refined by Brent's method on the
    exact score between its neighbours (the tabulated scores give the bracket ends), and the best is
    kept. If the maximum sits at the top of the grid, the search continues by doubling.
    """

    def __init__(self, C: np.ndarray, lc: np.ndarray):
        self.C, self.lc = C, lc
        self.tab_ll = np.empty((len(GRID), len(lc)))
        self.tab_s = np.empty((len(GRID), len(lc)))
        for g, lam in enumerate(GRID):
            self.tab_ll[g], self.tab_s[g] = sr_chosen(lam, C, lc)

    def loglik(self, lam, w):
        idx = np.flatnonzero(w > 0)
        return float(w[idx] @ sr_chosen(lam, self.C[idx], self.lc[idx], need_grad=False)[0])

    def score(self, lam, w):
        idx = np.flatnonzero(w > 0)
        return float(w[idx] @ sr_chosen(lam, self.C[idx], self.lc[idx])[1])

    def fit(self, w: np.ndarray):
        LLg = self.tab_ll @ w
        Sg = self.tab_s @ w
        lams = GRID
        if int(np.argmax(LLg)) == len(GRID) - 1:
            ext, vals, sc_ = [GRID[-1]], [LLg[-1]], [Sg[-1]]
            while len(ext) < 60:
                ext.append(ext[-1] * 2.0)
                vals.append(self.loglik(ext[-1], w))
                sc_.append(self.score(ext[-1], w))
                if vals[-1] < vals[-2]:
                    break
            else:
                raise RuntimeError("SR likelihood still rising at the top of the extended grid")
            lams = np.concatenate([GRID, ext[1:]])
            LLg = np.concatenate([LLg, vals[1:]])
            Sg = np.concatenate([Sg, sc_[1:]])
        G = len(lams)
        cands = [g for g in range(G - 1) if (g == 0 or LLg[g] >= LLg[g - 1]) and LLg[g] >= LLg[g + 1]]
        results = []
        for g in cands:
            if g == 0 and Sg[0] <= 0:
                results.append((0.0, True))
                continue
            lo, hi = (lams[g - 1], g - 1) if g > 0 else (0.0, 0), (lams[g + 1], g + 1)
            (a, ia), (b, ib) = lo, hi
            if Sg[ia] > 0 and Sg[ib] < 0:
                cache = {a: Sg[ia], b: Sg[ib]}
                lam = brentq(lambda x: cache[x] if x in cache else self.score(x, w), a, b,
                             xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=200)
                results.append((lam, True))
            else:
                results.append((lams[g], False))
        if len(results) == 1:
            lam, refined = results[0]
        else:
            lls = [self.loglik(lam, w) for lam, _ in results]
            lam, refined = results[int(np.argmax(lls))]
        return lam, {"n_local": len(cands), "refined": refined}


# ============================================================================ static models
def static_level_probs(P: "sc.Problem", lam: float) -> np.ndarray:
    """Per-item softmax probability at each level (T, 7), zero for absent levels."""
    e = np.where(P.C > 0, np.exp(lam * P.D), 0.0)
    return e / (P.C * e).sum(axis=1, keepdims=True)


def entropy_bits(C: np.ndarray, q: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where((C > 0) & (q > 0), C * q * np.log2(q), 0.0)
    return -t.sum(axis=1)


def soft_problem(P: "sc.Problem", Ptarget: np.ndarray) -> "sc.Problem":
    """CL problem whose 'chosen score' is the target expectation, so fit_lambda maximises
    sum_t w_t sum_items Ptarget log softmax(lambda v): its score is sum w (E_target[v] - E_q[v])."""
    Q = P.subset(np.arange(len(P.trials)))
    Q.dc = (Q.C * Ptarget * LEV[None, :]).sum(axis=1) - Q.Smax
    return Q


# ============================================================================ time law
def fit_time_law(e: np.ndarray, y: np.ndarray, w: np.ndarray):
    """min over t0 >= T0_MIN and tau >= 0 of sum w (y - log(t0 + tau e))^2, globally.

    Profile in u = log(t0 / tau): for fixed u the best log tau is a weighted mean, so the objective is
    a function of u alone. Candidates are the refined local minima of the profile on a grid over the
    feasible u (t0 >= T0_MIN), the boundary tau = 0, and the boundary t0 = T0_MIN. Efforts are
    compressed to their distinct values. Returns (t0, tau, objective).
    """
    keep = w > 0
    e, y, w = e[keep], y[keep], w[keep]
    ue, inv = np.unique(e, return_inverse=True)
    Wu = np.bincount(inv, weights=w)
    Yu = np.bincount(inv, weights=w * y)
    Qu = np.bincount(inv, weights=w * y * y)
    W, Ysum, Qsum = Wu.sum(), Yu.sum(), Qu.sum()
    loge = np.log(ue)

    def prof(u):
        u = np.atleast_1d(np.asarray(u, float))
        L = np.logaddexp(u[:, None], loge[None, :])  # log(s + e)
        c = (Ysum - L @ Wu) / W  # best log tau
        a = L + c[:, None]
        obj = Qsum - 2 * (a @ Yu) + (a * a) @ Wu
        return obj, c

    ybar = Ysum / W
    cands = [(max(Qsum - Ysum * ybar, 0.0), float(np.exp(ybar)), 0.0)]  # tau = 0
    log_t0min = np.log(T0_MIN)

    def h(u):  # log t0 at the profile optimum, minus log T0_MIN; increasing in u
        return float(u + prof(u)[1][0] - log_t0min)

    u_hi = loge.max() + 40.0
    u_lo = -1000.0
    if h(u_lo) < 0:
        u_lo = brentq(h, u_lo, u_hi, xtol=1e-13)
    grid = np.unique(np.concatenate([np.linspace(u_lo, u_hi, 60),
                                     np.linspace(loge.min() - 5.0, loge.max() + 5.0, 120)]))
    grid = grid[(grid >= u_lo) & (grid <= u_hi)]
    Gv, _ = prof(grid)
    for i in range(len(grid)):
        if (i == 0 or Gv[i] <= Gv[i - 1]) and (i == len(grid) - 1 or Gv[i] <= Gv[i + 1]):
            a = grid[max(i - 1, 0)]
            b = grid[min(i + 1, len(grid) - 1)]
            res = minimize_scalar(lambda u: float(prof(u)[0][0]), bounds=(a, b), method="bounded",
                                  options={"xatol": 1e-12, "maxiter": 500})
            u = res.x if res.fun <= Gv[i] else grid[i]
            obj, c = prof(u)
            cands.append((float(obj[0]), float(np.exp(u + c[0])), float(np.exp(c[0]))))
    # boundary t0 = T0_MIN, tau >= tau at u_lo
    v_lo = float(prof(u_lo)[1][0])

    def fb(v):
        a = np.logaddexp(log_t0min, v + loge)
        return float(Qsum - 2 * (a @ Yu) + (a * a) @ Wu)

    vgrid = v_lo + np.linspace(0.0, 40.0, 161)
    fv = np.array([fb(v) for v in vgrid])
    i = int(np.argmin(fv))
    res = minimize_scalar(fb, bounds=(vgrid[max(i - 1, 0)], vgrid[min(i + 1, len(vgrid) - 1)]), method="bounded",
                          options={"xatol": 1e-12, "maxiter": 500})
    v = res.x if res.fun <= fv[i] else vgrid[i]
    cands.append((fb(v), T0_MIN, float(np.exp(v))))
    best = min(cands, key=lambda z: z[0])
    return best[1], best[2], best[0]


def time_law_predict(t0: float, tau: float, e: np.ndarray) -> np.ndarray:
    """log(t0 + tau e)."""
    if tau == 0:
        return np.full(len(e), np.log(t0))
    return np.logaddexp(np.log(t0), np.log(tau) + np.log(e))


def time_law_objective(t0, tau, e, y, w):
    return float(w @ (y - np.log(t0 + tau * e)) ** 2)


# ============================================================================ within-participant rank correlations
class Groups:
    """Participant x set size groups of timed trials with at least three timed trials."""

    def __init__(self, data: dict):
        timed = np.flatnonzero(data["timed"])
        key = data["pidx"][timed] * 4 + data["size_idx"][timed]
        ukey, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        ok = cnt[inv] >= 3
        self.trials = timed[ok]  # trial indices entering S3
        ukey2, self.gid = np.unique(key[ok], return_inverse=True)
        self.ng = len(ukey2)
        self.gpart = (ukey2 // 4).astype(int)  # participant index of each group
        self.P = data["P"]
        self.rt_rank, self.rt_const = self.rank(data["rt_ms"][self.trials])

    def rank(self, values: np.ndarray):
        """Average ranks within groups (ties share the mean rank) and a flag for constant groups."""
        gid = self.gid
        order = np.lexsort((values, gid))
        gs, vs = gid[order], values[order]
        new_group = np.r_[True, gs[1:] != gs[:-1]]
        starts = np.flatnonzero(new_group)
        pos = np.arange(len(order)) - starts[np.cumsum(new_group) - 1] + 1.0
        new_run = new_group | np.r_[True, vs[1:] != vs[:-1]]
        run = np.cumsum(new_run) - 1
        avg = np.bincount(run, weights=pos) / np.bincount(run)
        ranks = np.empty(len(order))
        ranks[order] = avg[run]
        runs_per_group = np.bincount(gs[new_run], minlength=self.ng)
        return ranks, runs_per_group == 1

    def correlations(self, values: np.ndarray) -> np.ndarray:
        """Spearman correlation per group with the response time; nan when either variable is constant."""
        xr, xconst = self.rank(values)
        yr = self.rt_rank
        cnt = np.bincount(self.gid, minlength=self.ng)
        dx = xr - (np.bincount(self.gid, weights=xr, minlength=self.ng) / cnt)[self.gid]
        dy = yr - (np.bincount(self.gid, weights=yr, minlength=self.ng) / cnt)[self.gid]
        sxy = np.bincount(self.gid, weights=dx * dy, minlength=self.ng)
        sxx = np.bincount(self.gid, weights=dx * dx, minlength=self.ng)
        syy = np.bincount(self.gid, weights=dy * dy, minlength=self.ng)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = sxy / np.sqrt(sxx * syy)
        return np.where(xconst | self.rt_const, np.nan, r)

    def participant_mean(self, rgroup: np.ndarray, wp: np.ndarray) -> float:
        """Mean over each participant's set sizes, then the weighted mean over participants."""
        ok = np.isfinite(rgroup)
        s = np.bincount(self.gpart[ok], weights=rgroup[ok], minlength=self.P)
        k = np.bincount(self.gpart[ok], minlength=self.P)
        has = (k > 0) & (wp > 0)
        return float((wp[has] * s[has] / k[has]).sum() / wp[has].sum())


# ============================================================================ summaries
def wmean(x, w):
    return float(w @ x / w.sum())


def summary_S(t, v):
    return sc.summary_diff(t, v)


def summary_percent(d, v):
    """100 (2^dL - 1) with every interval end transformed from dL; p and clear from dL."""
    s = sc.summary_diff(d, v)
    tr = lambda x: float(100.0 * (2.0 ** x - 1.0))  # noqa: E731
    return {"point": tr(s["point"]), "percentile": [tr(x) for x in s["percentile"]],
            "basic": [tr(x) for x in s["basic"]], "p": s["p"], "clear": s["clear"]}
