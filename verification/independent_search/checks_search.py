"""Checks of the SR model and the Amendment 5 pipeline on simulated data only (never the real snack data).

Usage: python -I checks_search.py SIM_DIR   (SIM_DIR: synthetic data written by simulate.py)
Prints derived values only.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_search as rs  # noqa: E402
import search_core as srch  # noqa: E402
import simulate as sim  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, rs, sim, sc)
SIM_DIR = sys.argv[1]
assert os.path.realpath(SIM_DIR) != srch.REAL_DATA, "checks never touch the real snack data"
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def counts_of(v):
    return np.bincount((np.asarray(v, float) + 3).astype(int), minlength=7)[None, :].astype(float)


def item_probs_from_levels(P, v):
    return P[0, (np.asarray(v) + 3).astype(int)]


rng = np.random.default_rng(3)

# ----------------------------------------------------------------------------- 1. closed forms and birthday problem
worst = 0.0
for lam in (0.0, 0.4, 1.3, 3.0):
    v = [1, 0]
    q1 = math.exp(lam) / (math.exp(lam) + 1)
    q2 = 1 - q1
    P, E = srch.sr_probs(lam, counts_of(v))
    p = item_probs_from_levels(P, v)
    Px, Ex = srch.sr_exact([q1, q2])
    worst = max(worst, abs(p[0] - q1 ** 2 * (1 + 2 * q2)), abs(p[1] - q2 ** 2 * (1 + 2 * q1)), abs(E[0] - (2 + 2 * q1 * q2)),
                abs(Px[0] - q1 ** 2 * (1 + 2 * q2)), abs(Ex - (2 + 2 * q1 * q2)))
P, E = srch.sr_probs(0.7, counts_of([2, 2]))
worst = max(worst, abs(P[0, 5] - 0.5), abs(E[0] - 2.5))
check(f"two items: P(1) = q1^2 (1 + 2 q2) and E[looks] = 2 + 2 q1 q2, level form and exact (max error {worst:.1e})", worst < 1e-12)
worst = 0.0
for n in srch.SET_SIZES:
    _, E = srch.sr_probs(0.9, counts_of([1] * n))
    ref = sum(math.prod(1 - j / n for j in range(k)) for k in range(n + 1))
    worst = max(worst, abs(E[0] / ref - 1))
_, E365 = srch.sr_exact(np.full(365, 1 / 365))
check(f"equal ratings: E[looks] equals the birthday sum for n = 9, 16, 25, 36 (max relative error {worst:.1e}); 365 days give {E365:.4f}",
      worst < 1e-12 and abs(E365 - 24.6166) < 5e-5)

# level form against the item-level exact computation, random displays with ties
worst, worst_sum = 0.0, 0.0
for _ in range(40):
    n = int(rng.integers(9, 37))
    v = rng.integers(-3, 4, size=n).astype(float)
    lam = float(rng.uniform(0, 3))
    P, E = srch.sr_probs(lam, counts_of(v))
    q = np.exp(lam * v - np.max(lam * v))
    q /= q.sum()
    Px, Ex = srch.sr_exact(q)
    worst = max(worst, np.max(np.abs(item_probs_from_levels(P, v) / Px - 1)), abs(E[0] / Ex - 1))
    worst_sum = max(worst_sum, abs((counts_of(v) * P).sum() - 1))
    lp, _ = srch.sr_chosen(lam, counts_of(v), np.array([int(v[0]) + 3]))
    worst = max(worst, abs(np.exp(lp[0]) / Px[0] - 1))
check(f"level form equals the exact elementary symmetric computation on displays of 9 to 36 items (max relative error {worst:.1e}); "
      f"probabilities sum to one (max error {worst_sum:.1e})", worst < 1e-11 and worst_sum < 1e-12)

# ----------------------------------------------------------------------------- 2. simulation of the SR process
v = np.array([3, 3, 2, 0, -1, -3, 1, 2, 3, 0, 2, -2], float)
lam = 0.8
S = 400_000
choice, nlooks, counts = sim.simulate_sr(np.tile(v, (S, 1)), lam, rng)
P, E = srch.sr_probs(lam, counts_of(v))
p = item_probs_from_levels(P, v)
freq = np.bincount(choice, minlength=len(v)) / S
z = np.abs(freq - p) / np.sqrt(p * (1 - p) / S)
zE = abs(nlooks.mean() - E[0]) / (nlooks.std() / np.sqrt(S))
distinct_ok = np.array_equal((counts > 0).sum(axis=1), nlooks - 1)
check(f"simulated SR process: item choice frequencies within {z.max():.1f} s.e., mean looks {nlooks.mean():.4f} against {E[0]:.4f} "
      f"({zE:.1f} s.e.), distinct items looked at equal looks - 1 in every draw", z.max() < 4.5 and zE < 4.5 and distinct_ok)

# derivative of log P(chosen) against central differences
worst = 0.0
for lam in (0.0, 0.5, 1.7, 4.0):
    n = int(rng.integers(9, 37))
    V = rng.integers(-3, 4, size=(6, n)).astype(float)
    C = np.vstack([counts_of(x) for x in V])
    lc = np.array([int(x[rng.integers(n)]) + 3 for x in V])
    _, d = srch.sr_chosen(lam, C, lc)
    h = 1e-5
    f = lambda x: srch.sr_chosen(x, C, lc, need_grad=False)[0]  # noqa: E731
    fd = (f(lam + h) - f(lam - h)) / (2 * h) if lam >= h else (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
    worst = max(worst, float(np.max(np.abs(d - fd))))
check(f"analytic d log P(chosen) / d lambda equals finite differences (max abs difference {worst:.1e})", worst < 1e-7)


# ----------------------------------------------------------------------------- 3. parameter recovery
def random_trials(lam, T, rng):
    ns = rng.integers(9, 37, size=T)
    C = np.zeros((T, 7))
    lc = np.zeros(T, int)
    for n in np.unique(ns):
        idx = np.flatnonzero(ns == n)
        V = rng.integers(-3, 4, size=(len(idx), n)).astype(float)
        ch, _, _ = sim.simulate_sr(V, lam, rng)
        for j, t in enumerate(idx):
            C[t] = counts_of(V[j])[0]
            lc[t] = int(V[j, ch[j]]) + 3
    return C, lc


for lam_true in (0.6, 1.5):
    C, lc = random_trials(lam_true, 6000, rng)
    fitter = srch.SRFitter(C, lc)
    w = np.ones(len(lc))
    lam_hat, info = fitter.fit(w)
    h = 1e-4 * lam_hat
    info_obs = -(fitter.score(lam_hat + h, w) - fitter.score(lam_hat - h, w)) / (2 * h)
    se = 1 / math.sqrt(info_obs)
    check(f"recovery at lambda {lam_true}: fitted {lam_hat:.4f} (s.e. {se:.4f}, {abs(lam_hat - lam_true) / se:.1f} s.e. away), "
          f"score at the fit {fitter.score(lam_hat, w):.1e}, grid maxima {info['n_local']}",
          abs(lam_hat - lam_true) < 4 * se and info["refined"] and info_obs > 0)

# ----------------------------------------------------------------------------- 4. helpers
data = srch.load_search_data(SIM_DIR)
ref = sc.load_data(SIM_DIR)
same = all(np.array_equal(data[k], ref[k]) for k in ("choice", "C_all", "size_idx", "pidx", "lc")) and data["P"] == ref["P"]
check("loader keeps exactly the trials of the snack inclusion rule (same choices, counts, participants)", same)
import pandas as pd  # noqa: E402

raw_rows_ok, timed_ok, looked_ok = True, True, True
start = 0
for k, n in enumerate(srch.SET_SIZES):
    df = pd.read_csv(os.path.join(SIM_DIR, f"setsize-{n}_desc-data.csv"))
    kept = np.flatnonzero(data["size_idx"] == k)
    raw_rows_ok &= len(kept) == len(df) - 3  # three injected exclusions per set size
    rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
    V = df[[f"item_value_{i}" for i in range(n)]].to_numpy(float)
    ch = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
    keep = np.isfinite(V).all(axis=1) & (ch == np.floor(ch)) & (ch >= 0) & (ch <= n - 1)
    timed_ok &= np.array_equal(data["timed"][kept], np.isfinite(rt[keep]) & (rt[keep] > 0)) and (~data["timed"][kept]).sum() == 3
    G = df[[f"cumulative_gaze_{i}" for i in range(n)]].to_numpy(float)
    looked_ok &= np.array_equal(data["n_looked"][kept], (np.nan_to_num(G[keep]) > 0).sum(axis=1).astype(float))
check("edge cases: missing rating, out of range and non-integer choices excluded; missing, zero and negative rt not timed; "
      "missing gaze counts as zero", raw_rows_ok and timed_ok and looked_ok)
feat_ok = True
for k in range(4):
    V = data["V_by_size"][k]
    kept = np.flatnonzero(data["size_idx"] == k)
    feat_ok &= np.allclose(data["top_rating"][kept], V.max(axis=1)) and np.allclose(data["rating_sd"][kept], V.std(axis=1, ddof=0))
    feat_ok &= np.array_equal(data["top_count"][kept], (V == V.max(axis=1, keepdims=True)).sum(axis=1).astype(float))
    feat_ok &= np.array_equal(data["y_top"][kept], (V[np.arange(len(V)), data["choice"][kept]] == V.max(axis=1)).astype(float))
check("display features (highest rating, rating sd with ddof 0, count at the highest rating) and the top-choice flag", feat_ok)

# static models: probabilities, entropy, soft-target fit
P_cl = sc.Problem(data, "CL", "all", np.arange(len(data["choice"])))
q = srch.static_level_probs(P_cl, 1.3)
check("static level probabilities sum to one over items", np.allclose((P_cl.C * q).sum(axis=1), 1.0, atol=1e-13))
V9 = data["V_by_size"][0][:5]
idx9 = np.flatnonzero(data["size_idx"] == 0)[:5]
z = 1.3 * V9
qi = np.exp(z - z.max(axis=1, keepdims=True))
qi /= qi.sum(axis=1, keepdims=True)
H_item = -(qi * np.log2(qi)).sum(axis=1)
check("entropy in bits from levels equals the item-level entropy", np.allclose(srch.entropy_bits(P_cl.C[idx9], q[idx9]), H_item, atol=1e-12))
Ph = P_cl.subset(np.flatnonzero(data["size_idx"] == 2))
target = srch.static_level_probs(Ph, 0.85)
lam_soft = sc.fit_lambda(srch.soft_problem(Ph, target), np.ones((1, len(Ph.trials))), np.ones((1, len(Ph.trials))))[0]
check(f"implied sensitivity: soft targets from CL at 0.85 give back {lam_soft:.12f}", abs(lam_soft - 0.85) < 1e-10)

# time law: exact recovery, boundaries, global minimum against a brute-force grid
e = np.repeat([9.0, 16.0, 25.0], 30)
w = rng.integers(1, 3, size=len(e)).astype(float)
t0, tau, obj = srch.fit_time_law(e, np.log(1.5 + 0.2 * e), w)
check(f"time law recovers an exact law (t0 {t0:.6f}, tau {tau:.6f}, objective {obj:.1e})", abs(t0 - 1.5) < 1e-6 and abs(tau - 0.2) < 1e-7 and obj < 1e-12)
t0, tau, obj = srch.fit_time_law(e, np.log(5.0 - 0.1 * e), w)
check(f"time law with a falling relation takes tau = 0 (tau {tau})", tau == 0.0)
t0, tau, obj = srch.fit_time_law(e, np.log(0.3 * e), w)
check(f"time law with a proportional relation puts t0 at its bound (t0 {t0:.1e}, tau {tau:.6f})", t0 <= 1e-6 and abs(tau - 0.3) < 1e-6)


def brute(e, y, w):
    lt0 = np.linspace(np.log(srch.T0_MIN), np.log(1e3), 241)
    ltau = np.linspace(-14, 6, 241)
    best = float(w @ (y - (w @ y) / w.sum()) ** 2)  # tau = 0
    ue, inv = np.unique(e, return_inverse=True)
    Wu, Yu, Qu = (np.bincount(inv, weights=x) for x in (w, w * y, w * y * y))
    for a in lt0:
        Lg = np.logaddexp(a, ltau[:, None] + np.log(ue)[None, :])
        obj = Qu.sum() - 2 * Lg @ Yu + (Lg * Lg) @ Wu
        best = min(best, float(obj.min()))
    return best


worst = -np.inf
for trial in range(12):
    if trial % 3 == 0:
        e = np.repeat(rng.choice([9.0, 16.0, 25.0, 36.0], size=3, replace=False), 40)
    elif trial % 3 == 1:
        e = np.repeat(rng.choice([72.0, 240.0, 600.0, 1260.0], size=3, replace=False), 40)
    else:
        e = rng.uniform(2, 12, size=150)
    y = np.log(rng.uniform(0.2, 2) + rng.uniform(0, 0.5) * e) + rng.normal(0, 0.4, size=len(e))
    w = rng.integers(0, 4, size=len(e)).astype(float)
    t0, tau, obj = srch.fit_time_law(e, y, w)
    worst = max(worst, obj - brute(e, y, w))
    worst = max(worst, abs(obj - srch.time_law_objective(t0, tau, e, y, w)))
check(f"time law objective at or below a 241 x 241 brute-force grid on random problems (largest excess {worst:.1e})", worst <= 1e-9)

# within-participant rank correlations against scipy
class FakeData(dict):
    pass


npart = 6
fd = FakeData()
pid = np.repeat(np.arange(npart), 20)
size = np.tile(np.repeat(np.arange(4), 5), npart)
rt = rng.integers(500, 4000, size=len(pid)).astype(float)
rt[(pid == 1) & (size == 2)] = 1234.0  # constant response time
fd.update(pidx=pid, size_idx=size, P=npart, rt_ms=rt, timed=np.ones(len(pid), bool))
fd["timed"][(pid == 2) & (size == 0)] = [True, True, False, False, False]  # fewer than three timed trials
g = srch.Groups(fd)
x = rng.integers(0, 4, size=len(g.trials)).astype(float)  # many ties
x[g.gid == g.gid[np.flatnonzero((pid[g.trials] == 3) & (size[g.trials] == 1))[0]]] = 2.0  # constant forecast
rg = g.correlations(x)
worst, nan_ok = 0.0, True
for j in range(g.ng):
    m = g.gid == j
    xv, yv = x[m], rt[g.trials][m]
    if np.ptp(xv) == 0 or np.ptp(yv) == 0:
        nan_ok &= bool(np.isnan(rg[j]))
    else:
        worst = max(worst, abs(rg[j] - spearmanr(xv, yv).correlation))
excluded_ok = not np.any((pid[g.trials] == 2) & (size[g.trials] == 0))
wp = rng.integers(0, 3, size=npart).astype(float)
by_hand = []
for p_ in range(npart):
    vals = [rg[j] for j in range(g.ng) if g.gpart[j] == p_ and np.isfinite(rg[j])]
    if vals and wp[p_] > 0:
        by_hand.append((wp[p_], np.mean(vals)))
hand = sum(a * b for a, b in by_hand) / sum(a for a, _ in by_hand)
check(f"group Spearman equals scipy with ties (max difference {worst:.1e}); constant groups skipped; groups under three timed trials left out; "
      "participant weighting", worst < 1e-12 and nan_ok and excluded_ok and abs(g.participant_mean(rg, wp) - hand) < 1e-12)

# S4 score by hand
y = np.array([1.0, 0.0, 1.0, 0.0])
p = np.array([0.8, 0.3, 1.0, 0.0])
w = np.array([1.0, 2.0, 1.0, 1.0])
hand = (math.log2(0.8) + 2 * math.log2(0.7) + math.log2(1 - 1e-15) + math.log2(1 - 1e-15)) / 5
check("S4 score by hand, with clipping at 1e-15", abs(rs.bernoulli_score(y, p, w) - hand) < 1e-12)

# ----------------------------------------------------------------------------- 5. maxima of every fit on the synthetic data
pre = rs.Pre(data)
dense = np.concatenate([[0.0], np.logspace(-5, 3, 641)])  # 80 per decade, offset from the fitting grid
tab = np.array([srch.sr_chosen(x, pre.C, pre.lc, need_grad=False)[0] for x in dense])
worst_gap, worst_rel_score, worst_static, worst_time, n_fits = -np.inf, 0.0, 0.0, -np.inf, 0
for b in (-1, 0, 1):
    wp = np.ones(pre.P) if b < 0 else np.bincount(np.random.default_rng(rs.SEED_BASE + b).integers(0, pre.P, size=pre.P), minlength=pre.P).astype(float)
    w = wp[pre.pidx]
    ev = rs.evaluate(pre, wp)
    for k in range(4):
        wk = np.where(pre.train_mask[k], w, 0.0)
        lam = ev["lam"]["SR"][k]
        ll = pre.sr.loglik(lam, wk)
        worst_gap = max(worst_gap, float((tab @ wk).max() - ll))
        worst_rel_score = max(worst_rel_score, abs(pre.sr.score(lam, wk)) / abs(ll))
        n_fits += 1
        for fam in srch.STATIC:
            Pt = pre.train[fam][k]
            u = w[Pt.trials]
            m1, var, _ = sc.moments(Pt, np.full((1, len(u)), ev["lam"][fam][k]))
            g_ = float(u @ (Pt.dc - m1[0]))
            hss = -float(u @ var[0])
            worst_static = max(worst_static, abs(g_ / hss) / ev["lam"][fam][k])
        # time law fits of this fold against the brute-force grid
        tr = pre.timed & pre.train_mask[k] & (w > 0)
        for m in ("CL", "DbS_avg", "SR"):
            eff = pre.effort_fixed[m] if m != "SR" else srch.sr_expected_looks(lam, pre.C)
            if m == "SR" and b >= 0:
                continue  # SR's brute-force comparison on the original sample's four folds
            t0, tau, obj = srch.fit_time_law(eff[tr], pre.log_rt[tr], w[tr])
            worst_time = max(worst_time, obj - brute(eff[tr], pre.log_rt[tr], w[tr]))
check(f"SR: no lambda on a dense grid (0 and 1e-5 to 1e3, 80 per decade) beats any of the {n_fits} fold fits "
      f"(largest excess {worst_gap:.1e}); relative score at the fits {worst_rel_score:.1e}", worst_gap <= 1e-9 and worst_rel_score < 1e-12)
check(f"static models: every fold fit is a stationary point of a concave likelihood (largest relative Newton step left {worst_static:.1e})", worst_static < 1e-10)
check(f"time laws of the synthetic folds at or below the brute-force grid (largest excess {worst_time:.1e})", worst_time <= 1e-9)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
