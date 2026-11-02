"""Independent implementation of Amendment 10 (H1w) of ANALYSIS_PLAN.md, written from the plan text only.

Nothing from the overload package is imported. Data are read from the three CSV files of the
Aguiar, Boccardi, Kashaev and Kim (2023) package, columns x1 (menu index) and x2 (chosen option).
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np
from scipy.optimize import brentq, minimize_scalar

COSTS = ("low", "medium", "high")

# Prizes and lotteries from the authors' README.
PRIZES = np.array([50.0, 48.0, 30.0, 14.0, 12.0, 10.0, 0.0])
_l0 = np.array([0, 0, 0, 0, 1, 0, 0], float)
_l1 = np.array([0.5, 0, 0, 0, 0, 0, 0.5])
_l2 = np.array([0, 0, 0.5, 0, 0, 0.5, 0])
_a = np.array([0, 0.4, 0, 0.3, 0, 0, 0.3])
LOTTERIES = np.vstack([_l0, _l1, _l2, 0.5 * _l1 + 0.5 * _l2, 0.5 * _l1 + 0.5 * _a, 0.5 * _l2 + 0.5 * _a])
EV = LOTTERIES @ PRIZES  # 12, 25, 20, 22.5, 24.2, 21.7
EV_PLAN = np.array([12.0, 25.0, 20.0, 22.5, 24.2, 21.7])
assert np.allclose(EV, EV_PLAN)


def julia_powerset(items):
    """Combinatorics.jl powerset order: by size, then lexicographic."""
    out = []
    for k in range(len(items) + 1):
        out.extend(list(c) for c in itertools.combinations(items, k))
    return out


POWERSET = julia_powerset([1, 2, 3, 4, 5])  # index x1 - 1
MENU_IDS = list(range(2, 33))  # x1 = 1 is the empty menu, excluded
# Options on each menu in option order: 0 first, then lotteries ascending.
MENU_OPTS = {m: np.array([0] + POWERSET[m - 1], dtype=int) for m in MENU_IDS}
MENU_N = {m: len(MENU_OPTS[m]) for m in MENU_IDS}
SIZES = np.array([2, 3, 4, 5, 6])


def load_counts(data_dir):
    """counts[cost][m] = array of choice counts per option of menu m, in option order."""
    counts = {}
    for c in COSTS:
        d = np.loadtxt(os.path.join(data_dir, f"menu_choice_{c}.csv"), delimiter=",", skiprows=1, dtype=int)
        d = d[d[:, 0] != 1]
        cc = {}
        for m in MENU_IDS:
            ch = d[d[:, 0] == m, 1]
            opts = MENU_OPTS[m]
            bad = ~np.isin(ch, opts)
            if bad.any():
                raise ValueError(f"{c}: menu {m} has choices off the menu: {np.unique(ch[bad])}")
            cc[m] = np.array([(ch == o).sum() for o in opts], dtype=float)
        counts[c] = cc
    return counts


# ---------------------------------------------------------------- outcome and growth

def best_mask(values, m):
    v = values[MENU_OPTS[m]]
    return v >= v.max() - 1e-12


def miss_by_size(cnt, values, m_prob=None):
    """Observed miss counts and totals per set size. If m_prob is given (dict menu -> model miss
    probability), returns the model's expected misses instead."""
    miss = np.zeros(len(SIZES))
    tot = np.zeros(len(SIZES))
    for m in MENU_IDS:
        j = MENU_N[m] - 2
        N = cnt[m].sum()
        tot[j] += N
        if m_prob is None:
            miss[j] += cnt[m][~best_mask(values, m)].sum()
        else:
            miss[j] += N * m_prob[m]
    return miss, tot


def wls_slope(y, x, w):
    w = np.asarray(w, float)
    keep = w > 0
    y, x, w = y[keep], x[keep], w[keep]
    xb = np.sum(w * x) / w.sum()
    yb = np.sum(w * y) / w.sum()
    return float(np.sum(w * (x - xb) * (y - yb)) / np.sum(w * (x - xb) ** 2))


LOG2N = np.log2(SIZES)


def growth(rate, tot):
    return wls_slope(rate, LOG2N, tot)


# ---------------------------------------------------------------- CL

def cl_probs(v, lam):
    z = lam * (v - v.max())
    e = np.exp(z)
    return e / e.sum()


def cl_fit(cnt, values, menus):
    """ML sensitivity lam >= 0 of softmax(lam v) on the pooled choices of the given menus.
    The negative log likelihood is convex, so the estimate is the root of the score."""
    data = [(values[MENU_OPTS[m]], cnt[m]) for m in menus if cnt[m].sum() > 0]

    def score(lam):
        s = 0.0
        for v, c in data:
            p = cl_probs(v, lam)
            s += c @ v - c.sum() * (p @ v)
        return s

    if score(0.0) <= 0:
        return 0.0
    hi = 1.0
    while score(hi) > 0:
        hi *= 2.0
        if hi > 1e8:
            return float("inf")
    return float(brentq(score, 0.0, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))


# ---------------------------------------------------------------- sample until return

def _esp(q):
    """Elementary symmetric polynomials e_0..e_len(q) of q."""
    e = np.zeros(len(q) + 1)
    e[0] = 1.0
    for x in q:
        e[1:] = e[1:] + x * e[:-1]
    return e


def sr_probs(v, lam):
    q = cl_probs(v, lam)
    n = len(q)
    fact = np.array([math.factorial(k + 1) for k in range(n)], float)  # (k+1)! for k = 0..n-1
    P = np.empty(n)
    for i in range(n):
        e = _esp(np.delete(q, i))  # e_0..e_{n-1}
        P[i] = q[i] ** 2 * (fact @ e)
    return P


def _loglik(probfun, cnt, values, menus, lam):
    ll = 0.0
    for m in menus:
        c = cnt[m]
        if c.sum() == 0:
            continue
        p = probfun(values[MENU_OPTS[m]], lam)
        ll += c @ np.log2(np.maximum(p, 1e-300))
    return ll


SR_GRID = np.concatenate([[0.0], np.logspace(-3, 2, 251)])


class SRTable:
    """Precomputed log2 SR probabilities of each menu's options at every grid value, so that the
    grid search for a resample is a dot product."""

    def __init__(self, values):
        self.values = values
        self.tab = {m: np.array([np.log2(np.maximum(sr_probs(values[MENU_OPTS[m]], g), 1e-300)) for g in SR_GRID])
                    for m in MENU_IDS}

    def fit(self, cnt, menus):
        ll = np.zeros(len(SR_GRID))
        for m in menus:
            if cnt[m].sum() > 0:
                ll += self.tab[m] @ cnt[m]
        k = int(np.argmax(ll))
        lo = SR_GRID[max(k - 1, 0)]
        hi = SR_GRID[min(k + 1, len(SR_GRID) - 1)]
        f = lambda lam: -_loglik(sr_probs, cnt, self.values, menus, lam)  # noqa: E731
        r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
        best = (ll[k], SR_GRID[k])
        if -r.fun > best[0]:
            best = (-r.fun, float(r.x))
        return float(best[1])


# ---------------------------------------------------------------- held out forecasts

def held_out(cnt, values, model, srtab=None):
    """Each set size forecast from a fit to the other sizes. Returns forecast miss rate per size,
    forecast growth, bits per choice (mean log2 probability of the chosen option) and bits above
    uniform, and the fitted sensitivities per held out size."""
    miss_prob = {}
    logp_sum = 0.0
    above_sum = 0.0
    nchoice = 0.0
    lams = {}
    for n in SIZES:
        test = [m for m in MENU_IDS if MENU_N[m] == n]
        train = [m for m in MENU_IDS if MENU_N[m] != n]
        if model == "CL":
            lam = cl_fit(cnt, values, train)
            pf = cl_probs
        else:
            lam = srtab.fit(cnt, train)
            pf = sr_probs
        lams[int(n)] = lam
        for m in test:
            p = pf(values[MENU_OPTS[m]], lam)
            miss_prob[m] = float(1.0 - p[best_mask(values, m)].sum())
            lp = np.log2(np.maximum(p, 1e-300))
            logp_sum += cnt[m] @ lp
            above_sum += cnt[m] @ (lp + np.log2(len(p)))
            nchoice += cnt[m].sum()
    miss, tot = miss_by_size(cnt, values, miss_prob)
    rate = miss / tot
    return {"rate": rate, "growth": growth(rate, tot), "bits": logp_sum / nchoice,
            "bits_above_uniform": above_sum / nchoice, "lambda": lams}


# ---------------------------------------------------------------- CRRA

def crra_u(x, r):
    if abs(r - 1.0) < 1e-12:
        return np.log(x + 1.0)
    return (x + 1.0) ** (1.0 - r) / (1.0 - r)


def crra_values(r):
    return LOTTERIES @ crra_u(PRIZES, r)


def crra_fit(cnt):
    """ML of (r, lambda) for CL on expected CRRA utility, all menus of one cost. lambda is profiled
    out by root finding (convex for fixed r), r by a grid then bounded Brent search."""
    def prof(r):
        v = crra_values(r)
        lam = cl_fit(cnt, v, MENU_IDS)
        return _loglik(cl_probs, cnt, v, MENU_IDS, lam), lam

    grid = np.round(np.arange(-3.0, 6.0001, 0.05), 10)
    ll = np.array([prof(r)[0] for r in grid])
    k = int(np.argmax(ll))
    if k == 0 or k == len(grid) - 1:
        raise RuntimeError(f"CRRA r at the grid edge: {grid[k]}")
    res = minimize_scalar(lambda r: -prof(r)[0], bounds=(grid[k - 1], grid[k + 1]), method="bounded",
                          options={"xatol": 1e-10})
    r = float(res.x) if -res.fun >= ll[k] else float(grid[k])
    ll_r, lam = prof(r)
    return r, lam, ll_r


# ---------------------------------------------------------------- one full analysis

def analyse(counts, r_crra, srtabs):
    out = {}
    for c in COSTS:
        cnt = counts[c]
        miss, tot = miss_by_size(cnt, EV)
        rate = miss / tot
        g_obs = growth(rate, tot)
        cl = held_out(cnt, EV, "CL")
        sr = held_out(cnt, EV, "SR", srtabs)
        vcr = crra_values(r_crra)
        miss_c, tot_c = miss_by_size(cnt, vcr)
        g_crra = growth(miss_c / tot_c, tot_c)
        out[c] = {"rate": rate, "n_choices": tot, "growth_observed": g_obs, "growth_CL": cl["growth"],
                  "growth_SR": sr["growth"], "growth_observed_crra": g_crra,
                  "people_minus_CL": g_obs - cl["growth"], "people_minus_SR": g_obs - sr["growth"],
                  "bits_CL": cl["bits"], "bits_SR": sr["bits"],
                  "bits_above_uniform_CL": cl["bits_above_uniform"], "bits_above_uniform_SR": sr["bits_above_uniform"],
                  "rate_CL": cl["rate"], "rate_SR": sr["rate"], "rate_crra": miss_c / tot_c,
                  "lambda_CL": cl["lambda"], "lambda_SR": sr["lambda"]}
    out["H1w"] = out["high"]["growth_observed"] - out["low"]["growth_observed"]
    out["H1w_medium_minus_low"] = out["medium"]["growth_observed"] - out["low"]["growth_observed"]
    out["H1w_crra"] = out["high"]["growth_observed_crra"] - out["low"]["growth_observed_crra"]
    return out


def resample(counts, b):
    rng = np.random.default_rng(100000 + b)
    new = {}
    for c in COSTS:
        new[c] = {}
        for m in MENU_IDS:
            cn = counts[c][m]
            N = int(cn.sum())
            new[c][m] = rng.multinomial(N, cn / N).astype(float)
    return new


def summary(point, reps):
    v = np.asarray(reps, float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    pi = [float(lo), float(hi)]
    bi = [float(2 * point - hi), float(2 * point - lo)]
    excl = lambda iv: iv[0] > 0 or iv[1] < 0  # noqa: E731
    p = float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))
    return {"point": float(point), "percentile": pi, "basic": bi, "p": p, "clear": bool(excl(pi) and excl(bi))}
