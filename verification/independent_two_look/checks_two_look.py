"""Checks of two look satisficing (brief checks 1 and 2, and maxima of the fits) on simulated data only.

Usage: python -I checks_two_look.py SIM_DIR
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_two_look as rt  # noqa: E402
import two_look_core as tl  # noqa: E402
from two_look_core import srch  # noqa: E402

srch.assert_local(tl, rt, srch)
SIM_DIR = sys.argv[1]
assert os.path.realpath(SIM_DIR) != srch.REAL_DATA, "checks never touch the real snack data"
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def counts_of(v):
    return np.bincount((np.asarray(v, float) + 3).astype(int), minlength=7)[None, :].astype(float)


def items(levels, v):
    return levels[0, (np.asarray(v) + 3).astype(int)]


def tls_items(lam, v):
    q = np.exp(lam * v / 2 - np.max(lam * v / 2))
    return q / q.sum(), np.exp(lam * (v - 3) / 2)


def uniform_items(lam, v):
    return np.full(len(v), 1.0 / len(v)), np.exp(lam * (v - 3))


rng = np.random.default_rng(21)

# A. level form against the item-level exact route
worst = 0.0
for _ in range(40):
    n = int(rng.integers(9, 37))
    v = rng.integers(-3, 4, size=n).astype(float)
    lam = float(rng.uniform(0, 3))
    C = counts_of(v)
    for variant in ("tls", "uniform"):
        if variant == "tls":
            logq, logh = tl.tls_log_qh(lam, C)
            qi, hi = tls_items(lam, v)
        else:
            logq, logh = tl.uniform_log_qh(lam, C, np.array([float(n)]))
            qi, hi = uniform_items(lam, v)
        E, D = tl.two_look_expectations(logq, logh, C)
        Ex, Dx, _ = tl.exact_item_level(qi, hi)
        worst = max(worst, abs(E[0] / Ex - 1), abs(D[0] / Dx - 1))
check(f"E[looks] and E[distinct], level-form quadrature against the item-level elementary symmetric route, TLS and uniform attention "
      f"(max relative difference {worst:.1e})", worst < 1e-10)

# B. every h = 1 gives sample until return, with E[distinct] = E[looks] - 1
worst = 0.0
for _ in range(20):
    n = int(rng.integers(9, 37))
    v = rng.integers(-3, 4, size=n).astype(float)
    lam = float(rng.uniform(0, 3))
    C = counts_of(v)
    E, D = tl.two_look_expectations(srch.sr_logq(lam, C), np.zeros_like(C), C)
    Esr = srch.sr_expected_looks(lam, C)
    worst = max(worst, abs(E[0] / Esr[0] - 1), abs(D[0] - (E[0] - 1)) / D[0])
check(f"with every h = 1 the process is sample until return: E[looks] equals SR's and E[distinct] = E[looks] - 1 (max relative difference {worst:.1e})",
      worst < 1e-12)

# C. TLS choice probabilities equal SR's at the same lambda (exact registered-look race against sr_probs)
worst, worst_u, worst_id = 0.0, 0.0, 0.0
for _ in range(30):
    n = int(rng.integers(9, 37))
    v = rng.integers(-3, 4, size=n).astype(float)
    lam = float(rng.uniform(0, 3))
    C = counts_of(v)
    Psr, Esr = srch.sr_probs(lam, C)
    psr = items(Psr, v)
    _, _, ch = tl.exact_item_level(*tls_items(lam, v))
    _, _, chu = tl.exact_item_level(*uniform_items(lam, v))
    worst = max(worst, np.max(np.abs(ch / psr - 1)))
    worst_u = max(worst_u, np.max(np.abs(chu / psr - 1)))
    logq, logh = tl.tls_log_qh(lam, C)
    E, _ = tl.two_look_expectations(logq, logh, C, need_distinct=False)
    qi, hi = tls_items(lam, v)
    worst_id = max(worst_id, abs(E[0] / (Esr[0] / np.sum(qi * hi)) - 1))
check(f"TLS choice probabilities (exact race of registered looks) equal SR's at the same lambda (max relative difference {worst:.1e}); "
      f"so do uniform attention's ({worst_u:.1e}); TLS E[looks] = SR's E[looks] / P ({worst_id:.1e})",
      worst < 1e-10 and worst_u < 1e-10 and worst_id < 1e-12)

# D. simulation of the process
S = 200_000
for label, v, lam, variant in (("TLS, top rating 3", np.array([3, 3, 2, 0, -1, -3, 1, 2, 3, 0, 2, -2], float), 1.2, "tls"),
                               ("TLS, top rating 1", np.array([1, 0, 0, -1, 1, -2, 0, 1, -3], float), 0.9, "tls"),
                               ("uniform attention", np.array([3, 3, 2, 0, -1, -3, 1, 2, 3, 0, 2, -2], float), 0.8, "uniform")):
    qi, hi = tls_items(lam, v) if variant == "tls" else uniform_items(lam, v)
    choice, nlooks, distinct, _ = tl.simulate_two_look(np.tile(qi, (S, 1)), np.tile(hi, (S, 1)), rng)
    C = counts_of(v)
    logq, logh = tl.tls_log_qh(lam, C) if variant == "tls" else tl.uniform_log_qh(lam, C, np.array([float(len(v))]))
    E, D = tl.two_look_expectations(logq, logh, C)
    p = items(srch.sr_probs(lam, C)[0], v)
    freq = np.bincount(choice, minlength=len(v)) / S
    z = np.max(np.abs(freq - p) / np.sqrt(p * (1 - p) / S))
    zE = abs(nlooks.mean() - E[0]) / (nlooks.std() / np.sqrt(S))
    zD = abs(distinct.mean() - D[0]) / (distinct.std() / np.sqrt(S))
    check(f"simulation, {label}: looks {nlooks.mean():.4f} against {E[0]:.4f} ({zE:.1f} s.e.), distinct {distinct.mean():.4f} against "
          f"{D[0]:.4f} ({zD:.1f} s.e.), choices within {z:.1f} s.e. of SR's probabilities", zE < 4.5 and zD < 4.5 and z < 4.5)

# E. maxima of the SR (= TLS) fits on the synthetic TLS data: original sample and one replicate
data = srch.load_search_data(SIM_DIR)
pre = rt.Pre(data)
dense = np.concatenate([[0.0], np.logspace(-5, 3, 641)])
tab = np.array([srch.sr_chosen(x, pre.C, data["lc"], need_grad=False)[0] for x in dense])
worst_gap, worst_score = -np.inf, 0.0
for b in (-1, 0):
    wp = np.ones(pre.P) if b < 0 else np.bincount(np.random.default_rng(rt.SEED_BASE + b).integers(0, pre.P, size=pre.P), minlength=pre.P).astype(float)
    w = wp[pre.pidx]
    for k in range(4):
        wk = np.where(pre.train_mask[k], w, 0.0)
        lam, info = pre.sr.fit(wk)
        ll = pre.sr.loglik(lam, wk)
        worst_gap = max(worst_gap, float((tab @ wk).max() - ll))
        worst_score = max(worst_score, abs(pre.sr.score(lam, wk)) / abs(ll))
check(f"every sensitivity fit (original sample and a replicate, four folds each) is the global maximum on a dense grid "
      f"(largest excess {worst_gap:.1e}) and a stationary point (relative score {worst_score:.1e})", worst_gap <= 1e-9 and worst_score < 1e-12)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
