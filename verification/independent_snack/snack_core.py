"""Independent reimplementation of the confirmatory H1 tests on the snack data (Amendment 3).

Written from ANALYSIS_PLAN.md (Sections 3 to 5, Amendments 2 and 3) and VERIFY_BRIEF_SNACK.md only.
Nothing in this module prints or stores participant counts or trial counts.

Representation. Ratings are integers from -3 to 3, and every score family (CL, DbS_avg, MLBA_avg)
gives items of equal rating within a trial equal scores. A trial's softmax therefore depends only on
the score at each of the 7 rating levels and on how many items of each level are in the support
(all displayed items, or the looked-at items). The item-level code in checks_snack.py confirms this.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
DATA = os.path.join(VERIFY, "snack")
SET_SIZES = (9, 16, 25, 36)
LEVELS = np.arange(-3.0, 4.0)
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
FAMILIES = ("CL", "DbS_avg", "MLBA_avg")
VERSIONS = ("all", "seen")
ESAMP_DRAWS = 2000
BETA_BOUNDS = (-3.0, 3.0)
LN2 = np.log(2.0)


def cell_name(family: str, beta: float, version: str) -> str:
    return f"{family}|beta={float(beta) + 0.0:+.1f}|{version}"


# ============================================================================ data
def load_data(data_dir: str = DATA) -> dict:
    subjects, choices, Vs, Ls = [], [], [], []
    for k, n in enumerate(SET_SIZES):
        df = pd.read_csv(os.path.join(data_dir, f"setsize-{n}_desc-data.csv"))
        V = df[[f"item_value_{i}" for i in range(n)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        G = df[[f"cumulative_gaze_{i}" for i in range(n)]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        G = np.where(np.isfinite(G), G, 0.0)  # a missing gaze value counts as zero
        ch = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
        keep = np.isfinite(ch) & (ch == np.floor(ch)) & (ch >= 0) & (ch <= n - 1) & np.isfinite(V).all(axis=1)
        subjects.append(df["subject"].to_numpy()[keep])
        choices.append(ch[keep].astype(int))
        Vs.append(V[keep])
        Ls.append(G[keep] > 0)
    return build_data(subjects, choices, Vs, Ls)


def build_data(subjects: list, choices: list, Vs: list, Ls: list) -> dict:
    """Trial table from per set size arrays (ratings V, looked-at masks L, chosen indices)."""
    sizes = []
    for k, ch in enumerate(choices):
        if not np.isin(Vs[k], LEVELS).all():
            raise ValueError("ratings outside the integers -3..3")
        sizes.append(np.full(len(ch), k))
    subject = np.concatenate(subjects)
    ids = np.unique(subject)  # sorted distinct identifiers over the four files
    size_idx = np.concatenate(sizes)
    if [V.shape[1] for V in Vs] != list(SET_SIZES):
        raise ValueError("displays must have 9, 16, 25 and 36 items in that order")
    n = np.array(SET_SIZES, float)[size_idx]
    choice = np.concatenate(choices)
    T = len(choice)
    C_all = np.zeros((T, 7))
    C_seen = np.zeros((T, 7))
    lc = np.zeros(T, int)
    seen_chosen = np.zeros(T, bool)
    frac_looked = np.zeros(T)
    start = 0
    for k in range(4):
        V, L, ch = Vs[k], Ls[k], choices[k]
        rows = slice(start, start + len(ch))
        for li, lev in enumerate(LEVELS):
            C_all[rows, li] = (V == lev).sum(axis=1)
            C_seen[rows, li] = ((V == lev) & L).sum(axis=1)
        r = np.arange(len(ch))
        lc[rows] = (V[r, ch] + 3).astype(int)
        seen_chosen[rows] = L[r, ch]
        frac_looked[rows] = L.mean(axis=1)
        start += len(ch)
    return dict(
        size_idx=size_idx,
        n=n,
        pidx=np.searchsorted(ids, subject),
        P=len(ids),
        choice=choice,
        C_all=C_all,
        C_seen=C_seen,
        lc=lc,
        seen_chosen=seen_chosen,
        frac_looked=frac_looked,
        V_by_size=Vs,
        looked_by_size=Ls,
    )


def level_scores(family: str, C_all: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Score of an item at each rating level, per trial (T, 7), from every displayed item."""
    d = LEVELS[:, None] - LEVELS[None, :]  # d[l, l'] = level_l - level_l'
    if family == "CL":
        return np.tile(LEVELS, (len(n), 1))
    if family == "DbS_avg":
        K = np.sign(d)
    elif family == "MLBA_avg":
        K = np.exp(-np.abs(d)) * d  # kappa0 = 1
    else:
        raise ValueError(family)
    return (C_all @ K.T) / (n - 1)[:, None]  # the item itself has d = 0 and adds nothing


class Problem:
    """One score family with one support, restricted to a set of trials (indices into the data)."""

    def __init__(self, data: dict, family: str, support: str, trials: np.ndarray):
        S = level_scores(family, data["C_all"], data["n"])[trials]
        C = (data["C_all"] if support == "all" else data["C_seen"])[trials]
        lc = data["lc"][trials]
        if not (C[np.arange(len(lc)), lc] > 0).all():
            raise ValueError("chosen item outside the support")
        supp = C > 0
        self.family, self.support = family, support
        self.trials = np.asarray(trials)
        self.C = C
        self.Smax = np.where(supp, S, -np.inf).max(axis=1)
        self.Smin = np.where(supp, S, np.inf).min(axis=1)
        self.D = np.where(supp, S - self.Smax[:, None], 0.0)  # <= 0, finite everywhere
        self.dc = S[np.arange(len(lc)), lc] - self.Smax  # chosen score relative to the top
        self.size_idx = data["size_idx"][trials]
        self.n = data["n"][trials]
        self._finalize()

    def _finalize(self):
        self.logA = np.log((self.n - 1) / 7.0)
        self.levels_used = [l for l in range(7) if self.C[:, l].any()]
        self.Dl = [np.ascontiguousarray(self.D[:, l]) for l in self.levels_used]
        self.Cl = [np.ascontiguousarray(self.C[:, l]) for l in self.levels_used]
        self.mean0 = (self.C * self.D).sum(axis=1) / self.C.sum(axis=1)  # uniform mean of D over the support

    def subset(self, idx: np.ndarray) -> "Problem":
        """The same problem on a subset of its trials (positions within this problem)."""
        sub = object.__new__(Problem)
        sub.family, sub.support = self.family, self.support
        for name in ("trials", "C", "Smax", "Smin", "D", "dc", "size_idx", "n"):
            setattr(sub, name, getattr(self, name)[idx])
        sub._finalize()
        return sub


def moments(P: Problem, x: np.ndarray, need_var: bool = True):
    """Softmax over the support with sensitivity x (K, T) >= 0 applied to the level scores.

    Returns the mean m1 and variance of D = S - Smax, and log Z - x Smax, all (K, T).
    Every exponent x * D is <= 0 and the top level contributes at least one item, so nothing overflows.
    """
    tot = np.zeros_like(x)
    s1 = np.zeros_like(x)
    s2 = np.zeros_like(x) if need_var else None
    e = np.empty_like(x)
    ed = np.empty_like(x)
    for d, c in zip(P.Dl, P.Cl):
        np.multiply(x, d, out=e)
        np.exp(e, out=e)
        e *= c
        tot += e
        np.multiply(e, d, out=ed)
        s1 += ed
        if need_var:
            ed *= d
            s2 += ed
    s1 /= tot
    var = np.maximum(s2 / tot - s1 * s1, 0.0) if need_var else None
    return s1, var, np.log(tot)


# ============================================================================ fitting
def fit_lambda(P: Problem, mult: np.ndarray, u: np.ndarray, init=None, rtol=1e-13, maxit=300) -> np.ndarray:
    """Weighted ML of lambda >= 0 for K problems at once.

    Problem k maximises sum_t u[k, t] (lam mult[k, t] s_chosen - log sum_supp exp(lam mult[k, t] s)).
    lambda = 0 when the score at zero is not positive. Otherwise Newton steps on the (decreasing) score,
    safeguarded by a bracket [lo, hi] that is tightened at every step (bisection when a step leaves it,
    doubling while no upper end is known). Stops when the step is below rtol relative to lambda or the
    bracket has collapsed to machine precision.
    """
    K = mult.shape[0]
    um = u * mult

    def score(lam):
        m1, var, _ = moments(P, lam[:, None] * mult)
        return (um * (P.dc[None] - m1)).sum(axis=1), -(um * mult * var).sum(axis=1)

    g0 = (um * (P.dc - P.mean0)[None]).sum(axis=1)  # score at lambda = 0 (uniform over the support)
    spread = (P.Smax - P.Smin)[None] * mult
    mag = (u * spread).sum(axis=1)
    zero = (g0 <= 1e-13 * mag) | (mag <= 0)
    smax = np.where(u > 0, spread, 0).max(axis=1)
    start = 1.0 / np.where(smax > 0, smax, 1.0)
    if init is not None:
        start = np.where(init > 0, init, start)
    lo = np.zeros(K)
    hi = np.full(K, np.inf)
    x = np.where(zero, 0.0, start)
    for _ in range(maxit):
        g, h = score(x)
        lo = np.where((g > 0) & ~zero, x, lo)
        hi = np.where((g < 0) & ~zero, x, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            step = -g / h
        small = np.isfinite(step) & (np.abs(step) <= rtol * np.abs(x))  # converged: accept the last step
        xn = x + np.where(np.isfinite(step), step, 0.0)
        bad = ~small & (~np.isfinite(step) | (xn <= lo) | (xn >= hi))
        fallback = np.where(np.isfinite(hi), 0.5 * (lo + hi), 2.0 * np.maximum(x, lo))
        xn = np.where(bad, fallback, xn)
        xn = np.where(zero, 0.0, xn)
        done = small | (hi - lo <= 4 * np.finfo(float).eps * np.abs(xn)) | zero
        x = xn
        if done.all():
            return x
    raise RuntimeError("lambda fit did not converge")


def loglik(P: Problem, lam: np.ndarray, mult: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = lam[:, None] * mult
    _, _, logtot = moments(P, x, need_var=False)
    return (u * (x * P.dc[None] - logtot)).sum(axis=1)


def log2_scores(P: Problem, lam: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """log2(n q_chosen) per (problem, trial), n the number of displayed items."""
    x = lam[:, None] * mult
    _, _, logtot = moments(P, x, need_var=False)
    return (x * P.dc[None] - logtot) / LN2 + np.log2(P.n)[None]


def joint_newton(P: Problem, u: np.ndarray, beta: float, lam: float, tol=1e-12, maxit=60):
    """Newton ascent on (lambda, beta) from a nearby start with backtracking on the log likelihood.

    Returns (beta, lambda) at an interior maximum, or None when it does not converge cleanly, in
    which case the caller falls back to the profile search.
    """
    lo_b, hi_b = BETA_BOUNDS

    def ll(lm, bt):
        return float(loglik(P, np.array([lm]), np.exp(bt * P.logA)[None], u[None])[0])

    cur = ll(lam, beta)
    for _ in range(maxit):
        m = np.exp(beta * P.logA)
        a = m * P.logA
        m1, var, _ = moments(P, (lam * m)[None])
        l1, l2 = P.dc - m1[0], -var[0]
        g = np.array([np.sum(u * l1 * m), np.sum(u * l1 * lam * a)])
        H = np.array([
            [np.sum(u * l2 * m * m), np.sum(u * (l2 * m * lam * a + l1 * a))],
            [0.0, np.sum(u * (l2 * (lam * a) ** 2 + l1 * lam * a * P.logA))],
        ])
        H[1, 0] = H[0, 1]
        if not (H[0, 0] < 0 and np.linalg.det(H) > 0):
            return None
        step = -np.linalg.solve(H, g)
        t = 1.0
        while True:
            lam_n, beta_n = lam + t * step[0], beta + t * step[1]
            if lam_n > 0 and lo_b < beta_n < hi_b:
                new = ll(lam_n, beta_n)
                if new >= cur - 1e-12 * abs(cur):
                    break
            t *= 0.5
            if t < 1e-8:
                return None
        small = abs(t * step[1]) <= tol and abs(t * step[0]) <= tol * lam_n
        lam, beta, cur = lam_n, beta_n, new
        if small:
            return beta, lam
    return None


# ---------------------------------------------------------------------------- joint lambda and beta
def profile(P: Problem, beta: float, u: np.ndarray, init=None):
    """lambda*(beta) and the derivative of the profile log likelihood in beta (envelope theorem)."""
    mult = np.exp(beta * P.logA)[None]
    lam = fit_lambda(P, mult, u[None], init=None if init is None else np.array([init]))[0]
    m1, _, _ = moments(P, lam * mult)
    dbeta = float((u * lam * mult[0] * P.logA * (P.dc - m1[0])).sum())
    return lam, dbeta


def fit_joint(P: Problem, u: np.ndarray, guess=None, xtol=1e-13, start=None):
    """Joint ML of lambda >= 0 and beta in BETA_BOUNDS for CL scaled by ((n - 1)/7)^beta.

    With ``start`` = (beta, lambda) near the answer (bootstrap replicates), Newton ascent is tried
    first and accepted only if the profile derivative at its answer is zero to precision. Otherwise
    the profile derivative is found by bracketing (around ``guess`` if given, else on a grid over
    the bounds) and Brent's method. Returns beta, lambda, and the profile derivative at the answer.
    """
    lo_b, hi_b = BETA_BOUNDS
    if start is not None:
        res = joint_newton(P, u, start[0], start[1])
        if res is not None:
            beta, lam = res
            lam_p, d = profile(P, beta, u, init=lam)
            scale = np.sum(u) * max(1.0, lam_p)
            if abs(d) <= 1e-9 * scale and abs(lam_p - lam) <= 1e-10 * lam:
                return beta, lam_p, d
        guess = start[0]
    state = {"lam": None}

    def g(beta):
        lam, d = profile(P, beta, u, init=state["lam"])
        state["lam"] = lam
        return d

    if guess is None:
        grid = np.linspace(lo_b, hi_b, 61)
        vals = np.array([g(b) for b in grid])
        sign_changes = np.where((vals[:-1] > 0) & (vals[1:] <= 0))[0]
        if len(sign_changes) > 1:
            raise RuntimeError("profile likelihood in beta is not unimodal on the grid")
        if len(sign_changes) == 0:
            beta = lo_b if vals[0] <= 0 else hi_b
        else:
            i = sign_changes[0]
            beta = brentq(g, grid[i], grid[i + 1], xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=500)
    else:
        width = 0.25
        while True:
            a, b = max(lo_b, guess - width), min(hi_b, guess + width)
            ga, gb = g(a), g(b)
            if ga > 0 and gb < 0:
                beta = brentq(g, a, b, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=500)
                break
            if a == lo_b and ga <= 0:
                beta = lo_b
                break
            if b == hi_b and gb >= 0:
                beta = hi_b
                break
            width *= 2
    lam, d = profile(P, beta, u)
    return beta, lam, d


# ============================================================================ effective numbers and summaries
def effective_number(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_number_sampled(q: np.ndarray, N: int, seed: tuple) -> float:
    rng = np.random.default_rng(seed)
    q = np.asarray(q, float)
    s = rng.multinomial(int(N), q / q.sum(), size=ESAMP_DRAWS) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(s > 0, s * np.log(np.where(s > 0, s, 1.0)), 0.0).sum(axis=1)
    return float(np.exp(H).mean())


def clear_rule(pi, bi) -> bool:
    return bool((pi[0] > 0 and bi[0] > 0) or (pi[1] < 0 and bi[1] < 0))


def boot_p(v) -> float:
    v = np.asarray(v, float)
    return float(min(1.0, 2.0 * min(np.mean(v <= 0), np.mean(v >= 0))))


def summary_level(t, v) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": float(t), "percentile": [float(lo), float(hi)], "basic": [float(2 * t - hi), float(2 * t - lo)]}


def summary_point_pct(t, v) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": float(t), "percentile": [float(lo), float(hi)]}


def summary_diff(t, v) -> dict:
    d = summary_level(t, v)
    d["p"] = boot_p(v)
    d["clear"] = clear_rule(d["percentile"], d["basic"])
    return d


def summary_centered(t, err) -> dict:
    e_lo, e_hi = np.percentile(err, [2.5, 97.5])
    pi = [float(t + e_lo), float(t + e_hi)]
    bi = [float(t - e_hi), float(t - e_lo)]
    return {"point": float(t), "percentile": pi, "basic": bi, "p": boot_p(t + np.asarray(err)), "clear": clear_rule(pi, bi)}


def holm(pvals) -> list:
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="stable")
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return [float(a) for a in adj]


def ls_slope(x, Y):
    x = np.asarray(x, float)
    xc = x - x.mean()
    Y = np.asarray(Y, float)
    return (Y - Y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)
