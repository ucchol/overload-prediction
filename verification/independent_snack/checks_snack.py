"""Checks of the snack implementation on simulated data with an item-level reference, then on the real data.

Prints derived quantities only (no participant or trial counts).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_snack as rs  # noqa: E402
import snack_core as sc  # noqa: E402

ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


# ----------------------------------------------------------------------------- item-level reference
def item_scores(family, v):
    n = len(v)
    d = v[:, None] - v[None, :]
    if family == "CL":
        return v.astype(float)
    if family == "DbS_avg":
        return np.sign(d).sum(axis=1) / (n - 1)
    return (np.exp(-np.abs(d)) * d).sum(axis=1) / (n - 1)


def item_score_fn(trials, family, beta, support):
    """Score of the weighted log likelihood in lambda, written item by item."""
    pre = []
    for v, looked, ch, w in trials:
        n = len(v)
        s = item_scores(family, v) * ((n - 1) / 7.0) ** beta
        A = np.ones(n, bool) if support == "all" else looked
        pre.append((s[A], s[ch], w))

    def g(lam):
        tot = 0.0
        for s, sch, w in pre:
            z = lam * s
            q = np.exp(z - z.max())
            q /= q.sum()
            tot += w * (sch - (q * s).sum())
        return tot

    return g


def item_log2_score(trial, family, beta, support, lam):
    v, looked, ch, _ = trial
    n = len(v)
    s = item_scores(family, v) * ((n - 1) / 7.0) ** beta
    A = np.ones(n, bool) if support == "all" else looked.copy()
    z = np.where(A, lam * s, -np.inf)
    q = np.exp(z - z[A].max())
    q /= q.sum()
    return np.log2(n * q[ch])


def simulate(rng, per_size, lam, beta, family="CL", support="all", look=(0.8, 0.65, 0.5, 0.4), subjects=12):
    subj, chs, Vs, Ls = [], [], [], []
    for k, n in enumerate(sc.SET_SIZES):
        V = rng.integers(-3, 4, size=(per_size, n)).astype(float)
        L = rng.random((per_size, n)) < look[k]
        L[np.arange(per_size), rng.integers(0, n, per_size)] = True  # at least one item looked at
        ch = np.zeros(per_size, int)
        for t in range(per_size):
            s = item_scores(family, V[t]) * ((n - 1) / 7.0) ** beta
            A = np.ones(n, bool) if support == "all" else L[t]
            z = np.where(A, lam * s, -np.inf)
            q = np.exp(z - z[A].max())
            q /= q.sum()
            ch[t] = rng.choice(n, p=q)
        subj.append(rng.integers(0, subjects, per_size) * 7 + 3)
        chs.append(ch)
        Vs.append(V)
        Ls.append(L)
    return sc.build_data(subj, chs, Vs, Ls), (subj, chs, Vs, Ls)


rng = np.random.default_rng(11)

# 1. Masked softmax fit against a direct item-level fit on the available items, every family and support.
data, raw = simulate(rng, 60, 0.9, 0.3, support="seen")
w_trial = rng.integers(0, 3, size=len(data["choice"])).astype(float)  # bootstrap-like integer weights
trials_by = []
start = 0
for k in range(4):
    for t in range(len(raw[1][k])):
        trials_by.append((raw[2][k][t], raw[3][k][t], raw[1][k][t], w_trial[start + t]))
    start += len(raw[1][k])
probs = rs.build_problems(data)
maxrel, maxlog = 0.0, 0.0
for fam in sc.FAMILIES:
    for ver in sc.VERSIONS:
        P = probs[(fam, ver)]
        for beta in (-1.0, 0.0, 0.5):
            mult = np.exp(beta * P.logA)[None]
            lam = sc.fit_lambda(P, mult, w_trial[P.trials][None])[0]
            incl = [trials_by[i] for i in P.trials]
            g = item_score_fn(incl, fam, beta, ver)
            ref = brentq(g, 0.0, 50.0, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500) if g(0.0) > 0 else 0.0
            maxrel = max(maxrel, abs(lam - ref) / max(ref, 1e-300))
            bits = sc.log2_scores(P, np.array([lam]), mult)[0]
            ref_bits = np.array([item_log2_score(tr, fam, beta, ver, lam) for tr in incl])
            maxlog = max(maxlog, np.abs(bits - ref_bits).max())
check(f"level-count fit equals item-level fit on available items, 3 families x 2 supports x 3 betas (max rel diff {maxrel:.1e})", maxrel < 1e-10)
check(f"level-count log2(n q) equals item-level value (max abs diff {maxlog:.1e})", maxlog < 1e-12)

# 2. Two-item closed form inside the machinery: only items of ratings 1 and 0 looked at, choose 1 in 3 of 4 trials.
#    Softmax over the two looked-at items gives q = 1/(1 + exp(-lam)), so lam = ln 3 at beta = 0.
V = [np.zeros((4, n)) for n in sc.SET_SIZES]
Lm = [np.zeros((4, n), bool) for n in sc.SET_SIZES]
chs = []
for k in range(4):
    V[k][:, 0] = 1.0
    Lm[k][:, :2] = True
    chs.append(np.array([0, 0, 0, 1]))
d2 = sc.build_data([np.arange(4)] * 4, chs, V, Lm)
P2 = sc.Problem(d2, "CL", "seen", np.arange(len(d2["choice"])))
lam2 = sc.fit_lambda(P2, np.ones((1, len(d2["choice"]))), np.ones((1, len(d2["choice"]))))[0]
check(f"two looked-at items closed form lambda = ln 3 (got {lam2:.15f})", abs(lam2 - np.log(3)) < 1e-13)
d3 = sc.build_data([np.arange(4)] * 4, [np.array([1, 1, 1, 0])] * 4, V, Lm)
P3 = sc.Problem(d3, "CL", "seen", np.arange(len(d3["choice"])))
lam3 = sc.fit_lambda(P3, np.ones((1, len(d3["choice"]))), np.ones((1, len(d3["choice"]))))[0]
check("lambda = 0 when choices go against the ratings", lam3 == 0.0)

# 3. Recovery of a known beta from simulated choices, all items and looked-at items.
for support, lam0, beta0 in (("all", 0.8, -0.4), ("seen", 1.1, 0.35)):
    sim, _ = simulate(np.random.default_rng(5 if support == "all" else 6), 1500, lam0, beta0, support=support)
    pr = rs.build_problems(sim)
    P = pr["C3_all"] if support == "all" else pr["C3_seen"]
    u = np.ones(len(P.trials))
    beta, lam, dprof = sc.fit_joint(P, u)
    m = np.exp(beta * P.logA)
    x = lam * m
    m1, var, _ = sc.moments(P, x[None])
    l1, l2 = (P.dc - m1[0]), -var[0]
    H = np.array([[np.sum(u * l2 * m * m), np.sum(u * (l2 * m * lam * m * P.logA + l1 * m * P.logA))],
                  [0.0, np.sum(u * (l2 * (lam * m * P.logA) ** 2 + l1 * lam * m * P.logA ** 2))]])
    H[1, 0] = H[0, 1]
    se = np.sqrt(np.diag(np.linalg.inv(-H)))
    check(f"beta recovery ({support}): true {beta0:+.2f}, fitted {beta:+.4f} (se {se[1]:.4f}); lambda true {lam0}, fitted {lam:.4f} (se {se[0]:.4f})",
          abs(beta - beta0) < 4 * se[1] and abs(lam - lam0) < 4 * se[0] and np.all(np.linalg.eigvalsh(-H) > 0))
    grad_lam = np.sum(u * m * l1)
    check(f"joint optimum ({support}): d/dlambda {grad_lam:.1e}, d/dbeta {dprof:.1e} relative to scale", abs(grad_lam) < 1e-9 * np.sum(u) and abs(dprof) < 1e-9 * np.sum(u))
    ll0 = sc.loglik(P, np.array([lam]), m[None], u[None])[0]
    worse = all(sc.loglik(P, np.array([lam * fl]), np.exp((beta + db) * P.logA)[None], u[None])[0] <= ll0 + 1e-9
                for fl in (1 - 1e-5, 1, 1 + 1e-5) for db in (-1e-5, 0, 1e-5))
    check(f"joint log likelihood lower at every neighbouring (lambda, beta) ({support})", worse)

# 4. Profile derivative against a finite difference of the profile log likelihood.
P = probs[("CL", "all")]
u = np.ones(len(P.trials))
lam_a, d_a = sc.profile(P, 0.2, u)
h = 1e-5
pl = lambda bt: sc.loglik(P, np.array([sc.profile(P, bt, u)[0]]), np.exp(bt * P.logA)[None], u[None])[0]  # noqa: E731
fd = (pl(0.2 + h) - pl(0.2 - h)) / (2 * h)
check(f"profile derivative {d_a:.6f} matches finite difference {fd:.6f}", abs(d_a - fd) < 1e-5 * max(1, abs(fd)))

# 5. Effective numbers.
check("sampled effective number of a point mass is 1", sc.effective_number_sampled(np.array([1.0, 0, 0]), 50, (1, 2, 0, 0)) == 1.0)
check("effective number of uniform over 9 is 9", abs(sc.effective_number(np.ones(9) / 9) - 9) < 1e-12)
check("holm and centered summaries", sc.holm([0.01, 0.04]) == [0.02, 0.04] and sc.summary_centered(1.0, np.array([-0.5, 0.0, 0.5]))["clear"])

# ----------------------------------------------------------------------------- real data, original sample
data = sc.load_data()
probs = rs.build_problems(data)
w = np.ones(len(data["choice"]))
ev = rs.evaluate_sample(data, probs, w, b=-1)
worst_step, worst_brent, zero_fits, ll_ok = 0.0, 0.0, 0, True
for (fam, ver), lam in ev["lam"].items():
    P = probs[(fam, ver)]
    lam = lam.ravel()
    mult = np.exp(rs.PROB_BETAS[:, None] * P.logA[None])
    u = w[P.trials][None] * (P.size_idx[None] != rs.PROB_FOLDS[:, None])
    m1, var, _ = sc.moments(P, lam[:, None] * mult)
    g = (u * mult * (P.dc[None] - m1)).sum(axis=1)
    hss = -(u * mult ** 2 * var).sum(axis=1)
    zero_fits += int((lam == 0).sum())
    worst_step = max(worst_step, float(np.max(np.abs(g / hss) / lam)))
    ll = sc.loglik(P, lam, mult, u)
    for f in (1 - 1e-6, 1 + 1e-6):
        ll_ok &= bool(np.all(sc.loglik(P, lam * f, mult, u) <= ll + 1e-9))
    for kk in range(len(lam)):
        def gg(x, kk=kk):
            a, _, _ = sc.moments(P, np.array([[x]]) * mult[kk:kk + 1])
            return float((u[kk] * mult[kk] * (P.dc - a[0])).sum())
        ref = brentq(gg, 0.5 * lam[kk], 2 * lam[kk], xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
        worst_brent = max(worst_brent, abs(ref - lam[kk]) / ref)
check(f"real data C2: all held-out fold lambdas interior ({zero_fits} at zero), max relative Newton step left {worst_step:.1e}", zero_fits == 0 and worst_step < 1e-10)
check(f"real data C2: every fold lambda agrees with brentq (max rel diff {worst_brent:.1e})", worst_brent < 1e-10)
check("real data C2: log likelihood lower at lambda * (1 +- 1e-6) for every fold fit", ll_ok)
for key in ("C3_all", "C3_seen"):
    beta, lam, dprof = ev[key]
    P = probs[key]
    u = w[P.trials]
    m = np.exp(beta * P.logA)
    m1, _, _ = sc.moments(P, (lam * m)[None])
    gl = float(np.sum(u * m * (P.dc - m1[0])))
    ll0 = sc.loglik(P, np.array([lam]), m[None], u[None])[0]
    worse = all(sc.loglik(P, np.array([lam * fl]), np.exp((beta + db) * P.logA)[None], u[None])[0] <= ll0 + 1e-9
                for fl in (1 - 1e-6, 1, 1 + 1e-6) for db in (-1e-6, 0, 1e-6))
    inside = sc.BETA_BOUNDS[0] < beta < sc.BETA_BOUNDS[1]
    check(f"real data {key}: interior beta {inside}, d/dlambda {gl:.1e}, d/dbeta {dprof:.1e}, neighbours lower {worse}",
          inside and abs(gl) < 1e-8 * len(u) and abs(dprof) < 1e-8 * len(u) and worse)
for name, lam in ev["alone"].items():
    worst = 0.0
    for s in range(4):
        Ps = probs["alone"][name][s]
        m1, var, _ = sc.moments(Ps, np.full((1, len(Ps.trials)), lam[s]))
        g = float(np.sum(Ps.dc - m1[0]))
        h = -float(np.sum(var[0]))
        worst = max(worst, abs(g / h) / lam[s])
    check(f"real data lambda alone ({name}): all positive, max relative Newton step left {worst:.1e}", bool(np.all(lam > 0)) and worst < 1e-10)

# Fast paths against slow paths: subset fold fits against dense masked fits, and warm Newton C3 fits
# against the grid and Brent profile search, on the original sample and on a bootstrap replicate.
dense_diff = 0.0
for (fam, ver), lam in ev["lam"].items():
    P = probs[(fam, ver)]
    mult = np.exp(rs.PROB_BETAS[:, None] * P.logA[None])
    train = P.size_idx[None] != rs.PROB_FOLDS[:, None]
    lam_dense = sc.fit_lambda(P, mult, w[P.trials][None] * train)
    dense_diff = max(dense_diff, float(np.max(np.abs(lam_dense - lam.ravel()) / lam_dense)))
    bits = sc.log2_scores(P, lam_dense, mult)
    held = w[P.trials][None] * (~train)
    L_dense = ((held * bits).sum(axis=1) / held.sum(axis=1)).reshape(5, 4)
    dense_diff = max(dense_diff, float(np.max(np.abs(L_dense - ev["L_by"][(fam, ver)]))))
check(f"subset fold fits equal dense masked fits (max diff {dense_diff:.1e})", dense_diff < 1e-10)
draw = np.random.default_rng(20000 + 7).integers(0, data["P"], size=data["P"])
wb = np.bincount(draw, minlength=data["P"]).astype(float)[data["pidx"]]
for key in ("C3_all", "C3_seen"):
    P = probs[key]
    b_grid, l_grid, _ = sc.fit_joint(P, wb[P.trials])
    b_newt, l_newt, _ = sc.fit_joint(P, wb[P.trials], start=(ev[key][0], ev[key][1]))
    b_far, l_far, _ = sc.fit_joint(P, w[P.trials], start=(ev[key][0] + 0.3, ev[key][1] * 1.3))
    check(f"{key}: warm Newton equals grid+Brent on a replicate (beta diff {abs(b_grid - b_newt):.1e}, lambda rel diff {abs(l_grid - l_newt) / l_grid:.1e}); "
          f"far start recovers the original (beta diff {abs(b_far - ev[key][0]):.1e})",
          abs(b_grid - b_newt) < 1e-9 and abs(l_grid - l_newt) / l_grid < 1e-9 and abs(b_far - ev[key][0]) < 1e-9)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
