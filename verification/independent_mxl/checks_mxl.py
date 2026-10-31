"""Simulated checks of the MXL code: recovery, integral accuracy, global optimum, prediction quadrature."""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxl_core as mc  # noqa: E402

rng = np.random.default_rng(3)
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def simulate(mu, sig, P, T, sizes, levels):
    lam_p = np.exp(mu + sig * rng.normal(size=P))
    blocks, person = [], []
    t0 = 0
    for n in sizes:
        V = rng.integers(levels[0], levels[1] + 1, size=(P * T, n)).astype(float)
        pid = np.repeat(np.arange(P), T)
        q = np.exp(mc.cc.log_softmax(1.0, lam_p[pid][:, None] * V))
        c = (rng.random((P * T, 1)) > np.cumsum(q, axis=1)).sum(axis=1).clip(max=n - 1)
        idx = np.arange(t0, t0 + P * T)
        blocks.append(dict(n=n, idx=idx, V=V, S_CL=V, chosen=c))
        person.append(pid)
        t0 += P * T
    return blocks, np.concatenate(person)


# 1. recovery with narrow person likelihoods (180 trials per person)
for mu, sig in ((0.7, 0.35), (-1.2, 0.8)):
    blocks, person = simulate(mu, sig, 60, 60, (9, 16, 25), (-3, 3) if mu > 0 else (1, 35))
    m = mc.MXL(blocks, person, 60)
    fit = m.fit([True] * 3)
    se_mu, se_sig = sig / np.sqrt(60), sig / np.sqrt(120)
    check(f"recovery mu {mu}, sigma {sig}: {fit['mu']:.3f}, {fit['sigma']:.3f} (about {abs(fit['mu'] - mu) / se_mu:.1f} and "
          f"{abs(fit['sigma'] - sig) / se_sig:.1f} s.e.); coarse-grid local minima {fit['n_local_minima']}",
          abs(fit["mu"] - mu) < 4 * se_mu and abs(fit["sigma"] - sig) < 4 * se_sig)
    # 2. integral accuracy: one person against adaptive quadrature, and the objective on a grid of half the spacing
    LLp = fit["LLp"]
    i = 0
    idx_tr = [(b, np.flatnonzero(person[b["idx"]] == i)) for b in blocks]

    def ll_person(u):
        lam = np.exp(u)
        return sum(float((lam * b["V"][r][np.arange(len(r)), b["chosen"][r]] - logsumexp(lam * b["V"][r], axis=1)).sum()) for b, r in idx_tr)

    peak = mc.U_GRID[np.argmax(LLp[i])]
    ref_shift = ll_person(peak)
    f = lambda u: np.exp(ll_person(u) - ref_shift) * np.exp(-0.5 * ((u - fit["mu"]) / fit["sigma"]) ** 2) / (fit["sigma"] * np.sqrt(2 * np.pi))  # noqa: E731
    val, _ = quad(f, peak - 2, peak + 2, points=[peak], epsabs=0, epsrel=1e-12, limit=400)
    lphi = -0.5 * ((mc.U_GRID - fit["mu"]) / fit["sigma"]) ** 2 - np.log(fit["sigma"]) - 0.5 * np.log(2 * np.pi)
    grid_val = logsumexp(LLp[i] + lphi + mc.trap_logw(mc.U_GRID))
    half = np.arange(mc.U_GRID[0], mc.U_GRID[-1] + 1e-9, 0.0025)
    m2 = mc.MXL(blocks, person, 60, grid=half)
    d_half = abs(m2.negll(fit["mu"], fit["sigma"], m2.person_ll([True] * 3)) - fit["negll"])
    check(f"integral: one person's log integral on the grid vs adaptive quadrature differs by {abs(grid_val - (np.log(val) + ref_shift)):.1e}; "
          f"the objective at the optimum on a grid of half the spacing differs by {d_half:.1e}",
          abs(grid_val - (np.log(val) + ref_shift)) < 1e-8 and d_half < 1e-6)
    # 3. predictions: Gauss-Hermite 80 vs 160 nodes
    S = blocks[2]["S_CL"][:200]
    dq = np.abs(mc.mxl_probs(fit["mu"], fit["sigma"], S, 80) - mc.mxl_probs(fit["mu"], fit["sigma"], S, 160)).max()
    check(f"population-average predictions with 80 vs 160 Gauss-Hermite nodes differ by at most {dq:.1e}", dq < 1e-7)

# 4. a low-sigma case: sigma pinned near the lower bound when everyone shares lambda
blocks, person = simulate(0.3, 0.0001, 40, 45, (9, 16), (-3, 3))
fit = mc.MXL(blocks, person, 40).fit([True, True])
check(f"common lambda: sigma goes to the lower bound ({fit['sigma']:.3f}, at bound {fit['sigma_at_bound']}), mu {fit['mu']:.3f}",
      fit["sigma"] < 0.15 and abs(fit["mu"] - 0.3) < 0.1)
print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
