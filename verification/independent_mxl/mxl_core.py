"""Blind check (VERIFY_BRIEF_MXL.md): mixed logit and fitted overload forecasts.

Panels hold trials grouped in blocks (a set size, or a development display): block = (key, trial index, S, chosen,
values for the miss, n). Persons index every trial. Models:
  CL, CL_comp, DbS, MLBA   softmax(lam s), one sensitivity (root finding, ../independent_cdm/cdm_core.py)
  SR                       race on softmax(lam v), grid of Amendment 5 + bounded Brent, checked on a dense grid
  MXL                      CL with log lam ~ N(mu, sigma) across persons, one lam per person in the fit.
MXL likelihood. Every trial's log P(chosen | lam) is tabulated once on a fine grid of u = log lam (spacing 0.005
over [-10, 5]); a person's log likelihood is the sum of their trials' rows, and the person's integral over
N(u; mu, sigma) is a trapezoid sum on that grid in log space (checked against a grid of half the spacing).
The fit searches a coarse (mu, sigma) grid and polishes the best local minima with L-BFGS-B (sigma in [0.05, 1.5]).
Predictions average softmax(e^u s) over N(mu, sigma) by Gauss-Hermite quadrature (80 nodes, checked against 160).
Nothing here prints identifiers or participant, trial or choice counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_cdm"))
import cdm_core as cc  # noqa: E402

U_GRID = np.arange(-10.0, 5.0 + 1e-9, 0.005)
SIG_LO, SIG_HI = 0.05, 1.5
GH_NODES = 80
LOG_FLOOR = cc.LOG_FLOOR


def trap_logw(u):
    d = np.diff(u)
    w = np.zeros(len(u))
    w[:-1] += d / 2
    w[1:] += d / 2
    return np.log(w)


# ============================================================================ static softmax models and SR
def fit_softmax(blocks, S_key, w):
    items = [(b["n"], b["idx"], b[S_key], b["chosen"]) for b in blocks]
    return cc.fit_softmax_groups([b[S_key] for b in blocks], items, w)


def sr_ll(lam, blocks, w):
    tot = 0.0
    for b in blocks:
        wg = w[b["idx"]]
        on = wg > 0
        if on.any():
            tot += float(wg[on] @ cc.sr_log_chosen(lam, b["V"][on], b["chosen"][on]))
    return tot


def fit_sr(blocks, w):
    """Amendment 5 grid (0 and 21 log-spaced values 0.01..100) refined by bounded Brent between the best grid
    value's neighbours; global check against a dense grid of 121 values (returned as the excess)."""
    G = cc.SR_GRID
    LL = np.array([sr_ll(x, blocks, w) for x in G])
    g = int(np.argmax(LL))
    res = minimize_scalar(lambda x: -sr_ll(x, blocks, w), bounds=(G[max(g - 1, 0)], G[min(g + 1, len(G) - 1)]),
                          method="bounded", options={"xatol": 1e-10, "maxiter": 2000})
    lam, ll = float(res.x), -float(res.fun)
    dense = np.concatenate([[0.0], np.logspace(-3, 2, 120)])
    excess = max(sr_ll(x, blocks, w) for x in dense) - ll
    return lam, excess


def softmax_probs(lam, S):
    return np.exp(cc.log_softmax(lam, S))


def sr_probs(lam, V):
    return np.exp(cc.sr_log_probs(lam, V))


# ============================================================================ mixed logit
class MXL:
    """Log likelihood of CL summed per (person, block) on the u grid, tabulated once per panel."""

    def __init__(self, blocks, person, P, grid=U_GRID, chunk=40):
        self.grid = grid
        self.logw = trap_logw(grid)
        self.LLpb = np.zeros((P, len(blocks), len(grid)))
        lam = np.exp(grid)
        for k, bl in enumerate(blocks):
            S, c, pid = bl["S_CL"], bl["chosen"], person[bl["idx"]]
            sc = S[np.arange(len(c)), c]
            for a in range(0, len(grid), chunk):
                L = lam[a:a + chunk]
                Z = L[None, :, None] * S[:, None, :]
                ll = L[None, :] * sc[:, None] - logsumexp(Z, axis=2)  # (T_b, g)
                np.add.at(self.LLpb[:, k, a:a + chunk], pid, ll)
        self.has = np.zeros((P, len(blocks)), bool)
        for k, bl in enumerate(blocks):
            self.has[np.unique(person[bl["idx"]]), k] = True

    def person_ll(self, keep_blocks):
        kb = np.asarray(keep_blocks, bool)
        rows = self.has[:, kb].any(axis=1)
        return self.LLpb[rows][:, kb].sum(axis=1)

    def negll(self, mu, sig, LLp):
        lphi = -0.5 * ((self.grid - mu) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi)
        return -float(logsumexp(LLp + (lphi + self.logw)[None, :], axis=1).sum())

    def fit(self, keep_blocks):
        LLp = self.person_ll(keep_blocks)
        # where people's likelihoods live: individual maxima of u, clipped to the grid
        umax = self.grid[np.argmax(LLp, axis=1)]
        mus = np.arange(max(self.grid[0] + 1, np.min(umax) - 1), min(self.grid[-1] - 1, np.max(umax) + 1) + 1e-9, 0.05)
        sigs = np.geomspace(SIG_LO, SIG_HI, 25)
        F = np.array([[self.negll(m, s, LLp) for s in sigs] for m in mus])
        # local minima of the coarse grid (8-neighbourhood), best first
        Fp = np.pad(F, 1, constant_values=np.inf)
        is_min = np.ones(F.shape, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    is_min &= F <= Fp[1 + di:1 + di + F.shape[0], 1 + dj:1 + dj + F.shape[1]]
        cands = np.argwhere(is_min)
        cands = cands[np.argsort(F[is_min])][:5]
        best = None
        for i, j in cands:
            r = minimize(lambda z: self.negll(z[0], z[1], LLp), x0=[mus[i], sigs[j]], method="L-BFGS-B",
                         bounds=[(self.grid[0] + 1, self.grid[-1] - 1), (SIG_LO, SIG_HI)], options={"ftol": 1e-14, "gtol": 1e-9})
            if best is None or r.fun < best[2]:
                best = (float(r.x[0]), float(r.x[1]), float(r.fun))
        mu, sig, f = best
        return dict(mu=mu, sigma=sig, negll=f, n_local_minima=int(is_min.sum()), coarse_min=float(F.min()),
                    sigma_at_bound=bool(sig <= SIG_LO + 1e-6 or sig >= SIG_HI - 1e-6), LLp=LLp)


def mxl_probs(mu, sig, S, nodes=GH_NODES):
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)  # probabilists' Hermite: weight exp(-x^2/2)
    wq = wq / wq.sum()
    out = np.zeros(S.shape)
    for xi, wi in zip(x, wq):
        out += wi * softmax_probs(float(np.exp(mu + sig * xi)), S)
    return out
