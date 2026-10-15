"""Checks of the Amendment 6 implementation on simulated data only (brief tests 1, 2 and 4, and the loader).

Usage: python -I checks_gaze.py SIM_DIR
"""

from __future__ import annotations

import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "independent_two_look"))
sys.path.insert(2, os.path.join(os.path.dirname(HERE), "independent_search"))
import gaze_core as gc  # noqa: E402
import run_gaze as rg  # noqa: E402
import simulate as sim_sr  # noqa: E402  (SR process simulation of the search work)
import two_look_core as tl  # noqa: E402
from gaze_core import sc, srch  # noqa: E402

srch.assert_local(gc, rg, sim_sr, tl, srch, sc)
SIM_DIR = sys.argv[1]
assert os.path.realpath(SIM_DIR) != gc.GAZE_DIR, "checks never touch the real gaze data"
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


rng = np.random.default_rng(17)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def sample_choices(logq, rng):
    Q = np.exp(logq)
    return (rng.random((len(Q), 1)) > np.cumsum(Q, axis=1)).sum(axis=1).clip(max=Q.shape[1] - 1)


# ----------------------------------------------------------------------------- 1. small-set race formulas
worst_closed, worst_general, worst_tls, worst_sum = 0.0, 0.0, 0.0, 0.0
for _ in range(60):
    n = int(rng.choice([2, 3]))
    v = rng.integers(0, 11, size=n).astype(float)
    lam = float(rng.uniform(0, 3))
    V = v[None, :]
    P = np.exp(gc.sr_log_probs(lam, V))[0]
    E = gc.sr_expected_looks(lam, V)[0]
    q = softmax(lam * v)
    if n == 2:
        closed_P = np.array([q[0] ** 2 * (1 + 2 * q[1]), q[1] ** 2 * (1 + 2 * q[0])])
        closed_E = 2 + 2 * q[0] * q[1]
    else:
        closed_P = np.array([q[i] ** 2 * (1 + 2 * (q.sum() - q[i]) + 6 * np.prod(np.delete(q, i))) for i in range(3)])
        e2 = q[0] * q[1] + q[0] * q[2] + q[1] * q[2]
        closed_E = 2 + 2 * e2 + 6 * np.prod(q)
    Px, Ex = srch.sr_exact(q)
    worst_closed = max(worst_closed, np.max(np.abs(P - closed_P)), abs(E - closed_E))
    worst_general = max(worst_general, np.max(np.abs(P / Px - 1)), abs(E / Ex - 1))
    worst_sum = max(worst_sum, abs(P.sum() - 1))
    hq = softmax(lam * v / 2)
    hh = np.exp(lam * (v - gc.V_TOP) / 2)
    Etls_x, _, ch_x = tl.exact_item_level(hq, hh)
    worst_tls = max(worst_tls, abs(gc.tls_expected_looks(lam, V)[0] / Etls_x - 1), np.max(np.abs(ch_x / Px - 1)))
check(f"SR choice probabilities and expected looks equal the closed forms for two and three items (max error {worst_closed:.1e}) "
      f"and the earlier general code (max relative difference {worst_general:.1e}); probabilities sum to one ({worst_sum:.1e})",
      worst_closed < 1e-14 and worst_general < 1e-12 and worst_sum < 1e-14)
check(f"TLS expected looks with v_top = 10 equal the earlier two look code, and TLS chooses as SR (max relative difference {worst_tls:.1e})",
      worst_tls < 1e-12)

S = 300_000
for n, v, lam in ((2, np.array([7.0, 4.0]), 0.6), (3, np.array([9.0, 9.0, 5.0]), 0.5), (3, np.array([3.0, 1.0, 2.0]), 0.8)):
    V = np.tile(v, (S, 1))
    ch, nl, _ = sim_sr.simulate_sr(V, lam, rng)
    P = np.exp(gc.sr_log_probs(lam, v[None, :]))[0]
    E = gc.sr_expected_looks(lam, v[None, :])[0]
    z = np.max(np.abs(np.bincount(ch, minlength=n) / S - P) / np.sqrt(P * (1 - P) / S))
    zE = abs(nl.mean() - E) / (nl.std() / np.sqrt(S))
    q = np.tile(softmax(lam * v / 2), (S, 1))
    h = np.tile(np.exp(lam * (v - gc.V_TOP) / 2), (S, 1))
    ch2, nl2, _, _ = tl.simulate_two_look(q, h, rng)
    E2 = gc.tls_expected_looks(lam, v[None, :])[0]
    z2 = np.max(np.abs(np.bincount(ch2, minlength=n) / S - P) / np.sqrt(P * (1 - P) / S))
    zE2 = abs(nl2.mean() - E2) / (nl2.std() / np.sqrt(S))
    check(f"simulation, values {v.astype(int).tolist()} at lambda {lam}: SR choices within {z:.1f} s.e., SR looks {nl.mean():.4f} against "
          f"{E:.4f} ({zE:.1f} s.e.); TLS choices within {z2:.1f} s.e. of SR's probabilities, TLS looks {nl2.mean():.3f} against {E2:.3f} ({zE2:.1f} s.e.)",
          max(z, zE, z2, zE2) < 4.5)

# DbS and MLBA scores: by hand, and against the snack implementation (integer ratings -3..3, kappa = 1, scale 1)
hand_ok = np.allclose(gc.dbs_scores(np.array([[7.0, 4.0]])), [[1.0, -1.0]]) and np.allclose(gc.dbs_scores(np.array([[5.0, 5.0, 2.0]])), [[0.5, 0.5, -1.0]])
d = 0.6 * 3.0
hand_ok &= np.allclose(gc.mlba_scores(np.array([[7.0, 4.0]])), [[np.exp(-d) * d, -np.exp(-d) * d]])
v3 = np.array([[9.0, 6.0, 5.0]])
m3 = [(np.exp(-1.8) * 1.8 + np.exp(-2.4) * 2.4) / 2, (-np.exp(-1.8) * 1.8 + np.exp(-0.6) * 0.6) / 2, (-np.exp(-2.4) * 2.4 - np.exp(-0.6) * 0.6) / 2]
hand_ok &= np.allclose(gc.mlba_scores(v3), [m3])
worst = 0.0
for _ in range(50):
    n = int(rng.choice([2, 3]))
    v = rng.integers(-3, 4, size=n).astype(float)
    C = np.bincount((v + 3).astype(int), minlength=7)[None, :].astype(float)
    for fam, mine in (("DbS_avg", gc.dbs_scores(v[None, :])), ("MLBA_avg", gc.mlba_scores(v[None, :], scale=1.0))):
        lev = sc.level_scores(fam, C, np.array([float(n)]))[0]
        worst = max(worst, float(np.max(np.abs(mine[0] - lev[(v + 3).astype(int)]))))
check(f"DbS_avg and MLBA_avg scores by hand (MLBA on values times 0.6) and equal to the snack implementation at scale 1 (max difference {worst:.1e})",
      hand_ok and worst < 1e-14)

# ----------------------------------------------------------------------------- forecasts rounded to ten significant digits before ranking (item 6)
xs = np.concatenate([rng.lognormal(0.0, 3.0, 4000), rng.uniform(0.0, 10.0, 4000), np.round(rng.uniform(0.0, 10.0, 2000), 3),
                     [0.0, 1.0, 2.5, 1 / 3, 2 / 3, 0.1 + 0.2, 123456789.01234, 9.99999999995, 1.00000000005, 1e-12]])
fmt_ok = all(a == float(format(b, ".10g")) for a, b in zip(gc.round10(xs).tolist(), xs.tolist()))
K = 3000
V1 = rng.integers(0, 6, size=(K, 3)).astype(float)
V2 = V1 + rng.integers(1, 5, size=(K, 1))  # the same values shifted by a constant
V3 = V1[:, [2, 0, 1]]  # the same values in another order
W1 = np.round(rng.uniform(1.0, 10.0, size=(K, 3)), 2)
share_diff, ties, distinct = {}, True, True
for lam_x in (0.13, 0.5, 0.77, 1.3, 2.9):
    sr_l = lambda V: gc.sr_expected_looks(lam_x, V)  # noqa: E731
    tls_l = lambda V: gc.tls_expected_looks(lam_x, V)  # noqa: E731
    ent_l = lambda V: gc.entropy_bits(gc.log_softmax(lam_x, V))  # noqa: E731
    for label, a, b in (("SR looks, shifted", sr_l(V1), sr_l(V2)), ("SR looks, reordered", sr_l(V1), sr_l(V3)),
                        ("CL entropy, shifted", ent_l(V1), ent_l(V2)), ("TLS looks, reordered", tls_l(V1), tls_l(V3)),
                        ("mean value, reordered", W1.mean(axis=1), W1[:, [2, 0, 1]].mean(axis=1))):
        share_diff[label] = max(share_diff.get(label, 0.0), float(np.mean(a != b)))
        ties &= bool(np.all(gc.round10(a) == gc.round10(b)))
    distinct &= bool(np.all(gc.round10(tls_l(V1)) != gc.round10(tls_l(V2))))  # TLS depends on v - 10: no tie
x_toy = np.array([2.0, np.nextafter(2.0, 3.0), 3.0, 1.0, 2.0])
rt_toy = np.array([500.0, 400.0, 600.0, 300.0, 450.0])
g_toy = srch.Groups({"pidx": np.zeros(5, int), "size_idx": np.zeros(5, int), "P": 1, "timed": np.ones(5, bool), "rt_ms": rt_toy})
r_ref = spearmanr([2.0, 2.0, 3.0, 1.0, 2.0], rt_toy)[0]
r_rounded, r_raw = g_toy.correlations(gc.round10(x_toy))[0], g_toy.correlations(x_toy)[0]
check("forecasts rounded by float(format(x, '.10g')) element by element; forecasts equal in exact arithmetic tie after rounding "
      "(share of pairs that differ before rounding: " + ", ".join(f"{k} {v:.2f}" for k, v in share_diff.items())
      + f"), TLS looks of shifted displays stay distinct, and a one-ulp difference no longer splits a tie in the grouped Spearman "
      f"({r_rounded:.4f} with rounding, scipy on the tied values {r_ref:.4f}, {r_raw:.4f} without)",
      fmt_ok and ties and distinct and abs(r_rounded - r_ref) < 1e-14 and abs(r_raw - r_ref) > 1e-3)

# ----------------------------------------------------------------------------- 2. recovery of lambda
for n in (2, 3):
    for lam_true in (0.3, 0.9):
        T = 6000
        V = rng.integers(0, 11, size=(T, n)).astype(float)
        ch_cl = sample_choices(gc.log_softmax(lam_true, V), rng)
        ch_sr, _, _ = sim_sr.simulate_sr(V, lam_true, rng)
        w = np.ones(T)
        lam_cl = gc.fit_cl(V, ch_cl, w)
        fit = gc.SRFit(V, ch_sr)
        lam_sr, info = fit.fit(w)
        hh = 1e-4
        ll_cl = lambda x: float(gc.softmax_log_chosen(x, V, ch_cl).sum())  # noqa: E731
        se_cl = 1 / math.sqrt(-(ll_cl(lam_cl + hh) - 2 * ll_cl(lam_cl) + ll_cl(lam_cl - hh)) / hh ** 2)
        se_sr = 1 / math.sqrt(-(fit.loglik(lam_sr + hh, w) - 2 * fit.loglik(lam_sr, w) + fit.loglik(lam_sr - hh, w)) / hh ** 2)
        check(f"recovery, {n} items, lambda {lam_true}: CL {lam_cl:.4f} ({abs(lam_cl - lam_true) / se_cl:.1f} s.e.), "
              f"SR {lam_sr:.4f} ({abs(lam_sr - lam_true) / se_sr:.1f} s.e.), SR dense-grid excess {info['dense_excess']:.1e}",
              abs(lam_cl - lam_true) < 4 * se_cl and abs(lam_sr - lam_true) < 4 * se_sr and info["dense_excess"] <= 1e-9)
for name, score_fn, lam_true in (("DbS", gc.dbs_scores, 1.2), ("MLBA", gc.mlba_scores, 3.0)):
    T = 6000
    V = rng.integers(0, 11, size=(T, 3)).astype(float)
    Ssc = score_fn(V)
    chs = sample_choices(gc.log_softmax(lam_true, Ssc), rng)
    lam_hat = gc.fit_softmax(Ssc, chs, np.ones(T))
    ll = lambda x: float(gc.softmax_log_chosen(x, Ssc, chs).sum())  # noqa: E731
    hh = 1e-4
    se = 1 / math.sqrt(-(ll(lam_hat + hh) - 2 * ll(lam_hat) + ll(lam_hat - hh)) / hh ** 2)
    check(f"recovery of {name} at lambda {lam_true}: {lam_hat:.4f} ({abs(lam_hat - lam_true) / se:.1f} s.e.)", abs(lam_hat - lam_true) < 4 * se)

# ----------------------------------------------------------------------------- loader
try:
    gc.load_dataset(gc.GAZE_DIR, "krajbich2010")
    refused = False
except PermissionError:
    refused = True
check("the loader refuses the real gaze folder unless explicitly allowed (nothing is read)", refused)
data = {n: gc.load_dataset(SIM_DIR, n) for n in gc.DATASETS}
ok_fold, ok_incl = True, True
for n, d in data.items():
    raw = pd.read_csv(os.path.join(SIM_DIR, f"{n}.csv"))
    ids = np.unique(raw["subject"].dropna().to_numpy())
    perm_ids = np.random.default_rng(20261014).permutation(ids)  # the sorted participants, permuted, fresh generator
    fold_by_id = {pid: j % 10 for j, pid in enumerate(perm_ids)}
    ok_fold &= all(d["fold"][d["pidx"] == i][0] == fold_by_id[pid] for i, pid in enumerate(ids))
    ok_fold &= all(np.unique(d["fold"][d["pidx"] == i]).size == 1 for i in range(d["P"]))
    ok_incl &= len(d["choice"]) == len(raw) - 6 and np.all(d["rt_ms"] > 0) and np.isfinite(d["V"]).all()
check("folds: a participant's fold is its position in default_rng(20261014).permutation of the sorted participants, modulo 10", ok_fold)
check("inclusion: missing value, invalid or non-integer choice, missing, zero or negative rt excluded", ok_incl)
fd = data["folke2016"]
check("Folke's confidence attached by row (missing ratings kept as missing)", fd["conf"] is not None and np.isnan(fd["conf"]).any() and np.isfinite(fd["conf"]).mean() > 0.9)
bad = os.path.join(HERE, "sim", "bad_copy")
os.makedirs(bad, exist_ok=True)
for f in ("folke2016.csv", "folke2016_original.csv", "krajbich2010.csv"):
    shutil.copy(os.path.join(SIM_DIR, f), bad)
orig = pd.read_csv(os.path.join(bad, "folke2016_original.csv"))
orig.loc[30, "Choice_RT"] += 1.0
orig.to_csv(os.path.join(bad, "folke2016_original.csv"), index=False)
try:
    gc.load_dataset(bad, "folke2016")
    rt_guard = False
except ValueError:
    rt_guard = True
orig.iloc[:-1].to_csv(os.path.join(bad, "folke2016_original.csv"), index=False)
try:
    gc.load_dataset(bad, "folke2016")
    row_guard = False
except ValueError:
    row_guard = True
k10 = pd.read_csv(os.path.join(bad, "krajbich2010.csv"))
k10.loc[4, "item_value_0"] = 11.0  # row 4 is excluded (invalid choice): no stop
k10.to_csv(os.path.join(bad, "krajbich2010.csv"), index=False)
try:
    gc.load_dataset(bad, "krajbich2010")
    excluded_ok = True
except ValueError:
    excluded_ok = False
k10.loc[30, "item_value_0"] = 10.5  # an included trial above 10: stop
k10.to_csv(os.path.join(bad, "krajbich2010.csv"), index=False)
try:
    gc.load_dataset(bad, "krajbich2010")
    top_guard = False
except ValueError:
    top_guard = True
shutil.rmtree(bad)
check("the loader stops with an error when a response time differs or the row counts differ", rt_guard and row_guard)
check("the analysis stops when an included value lies above 10, and not for a value above 10 in an excluded row", top_guard and excluded_ok)


# ----------------------------------------------------------------------------- 4. maxima of every fit
def brute(e, y, w):
    """Best weighted squared error found by brute force: the boundary tau = 0, a 241 x 241 grid over (log t0, log tau)
    with t0 from 1e-9, and L-BFGS-B polishing (t0 >= 1e-9) from the grid's local minima (at most 25, best first)."""
    lt0 = np.linspace(np.log(srch.T0_MIN), np.log(1e3), 241)
    ltau = np.linspace(-14.0, 6.0, 241)
    ue, inv = np.unique(e, return_inverse=True)
    Wu, Yu, Qu = (np.bincount(inv, weights=x) for x in (w, w * y, w * y * y))
    le = np.log(ue)

    def obj(z):
        L = np.logaddexp(z[0], z[1] + le)
        return float(Qu.sum() - 2 * L @ Yu + (L * L) @ Wu)

    G = np.empty((len(lt0), len(ltau)))
    for i, a in enumerate(lt0):
        L = np.logaddexp(a, ltau[:, None] + le[None, :])
        G[i] = Qu.sum() - 2 * L @ Yu + (L * L) @ Wu
    best = min(float(w @ (y - (w @ y) / w.sum()) ** 2), float(G.min()))
    Gp = np.pad(G, 1, constant_values=np.inf)
    is_min = np.ones(G.shape, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_min &= G <= Gp[1 + di:1 + di + G.shape[0], 1 + dj:1 + dj + G.shape[1]]
    starts = np.argwhere(is_min)[np.argsort(G[is_min], kind="stable")][:25]
    for i, j in starts:
        r = minimize(obj, x0=[lt0[i], ltau[j]], method="L-BFGS-B", bounds=[(np.log(srch.T0_MIN), 20.0), (-60.0, 20.0)])
        best = min(best, float(r.fun))
    return best


worst_static, worst_dense, worst_time = 0.0, -np.inf, -np.inf
for name in gc.DATASETS:
    pre = rg.DataPre(data[name])
    for b in (-1, 0):
        wp = np.ones(pre.P) if b < 0 else np.bincount(np.random.default_rng(rg.BOOT_SEED + b).integers(0, pre.P, size=pre.P), minlength=pre.P).astype(float)
        w = wp[pre.pidx]
        folds = [np.where(pre.train[f], w, 0.0) for f in range(gc.N_FOLDS)] + [w]  # ten folds and the full sample
        for i, wt in enumerate(folds):
            for m in gc.STATIC:
                Sm = pre.S[m]
                lam = gc.fit_softmax(Sm, pre.choice, wt)
                q = np.exp(gc.log_softmax(lam, Sm))
                mean = (q * Sm).sum(axis=1)
                g = float(wt @ (Sm[np.arange(pre.T), pre.choice] - mean))
                hss = float(wt @ (q * (Sm - mean[:, None]) ** 2).sum(axis=1))
                if lam > 0:  # interior: the Newton step left must vanish relative to lambda
                    worst_static = max(worst_static, abs(g / hss) / lam)
                else:  # boundary: the score at zero must not be positive
                    worst_static = max(worst_static, max(g, 0.0) / max(hss, 1e-300))
            lam_sr, info = pre.sr.fit(wt)
            worst_dense = max(worst_dense, info["dense_excess"])
            if i < 2 or i == gc.N_FOLDS:  # time laws of two folds and of the full sample against brute force
                on = wt > 0
                for E in (gc.sr_expected_looks(lam_sr, pre.V), gc.tls_expected_looks(lam_sr, pre.V)):
                    t0, tau, obj = srch.fit_time_law(E[on], pre.y[on], wt[on])
                    bf = brute(E[on], pre.y[on], wt[on])
                    worst_time = max(worst_time, (obj - bf) / bf)
check(f"CL, DbS and MLBA: every fit (ten folds and the full sample) is a stationary point of a concave likelihood "
      f"(largest relative Newton step left {worst_static:.1e})", worst_static < 1e-10)
check(f"SR: the amendment's grid and bounded Brent search (xatol 1e-10) reach the global maximum in every fit (largest dense-grid excess {worst_dense:.1e})",
      worst_dense <= 1e-9)
check(f"time laws (SR and TLS effort; two folds and the full sample per dataset, original and resampled weights) at or below "
      f"brute force, a grid polished from each of its local minima (largest relative excess {worst_time:.1e})", worst_time <= 1e-10)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
