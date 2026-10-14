"""Checks of the search models, effort, global fits and the summary statistics.

Prints derived values only (no identifiers, participant counts or choice counts).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovl_answers as ov  # noqa: E402
import process_core as pc  # noqa: E402
import run_process as rp  # noqa: E402
from process_core import core  # noqa: E402

pc.assert_local(ov, pc, rp, core)
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def sr_from_q(q):
    """sr_batch on values log q with lambda = 1, so that softmax(lambda v) = q."""
    q = np.asarray(q, float)
    logP, _, E = pc.sr_batch(1.0, np.log(q)[None], np.ones((1, len(q)), bool), need_grad=False)
    return np.exp(logP[0]), E[0]


rng = np.random.default_rng(1)

# 1. Two options, closed forms q1^2 (1 + 2 q2) and 2 + 2 q1 q2.
worst = 0.0
for q1 in (0.5, 0.3, 0.9, 0.01):
    q = np.array([q1, 1 - q1])
    P, E = sr_from_q(q)
    Pd, Ed = pc.sr_dp(q)
    target = np.array([q1 ** 2 * (1 + 2 * (1 - q1)), (1 - q1) ** 2 * (1 + 2 * q1)])
    worst = max(worst, np.abs(P - target).max(), abs(E - (2 + 2 * q1 * (1 - q1))), np.abs(Pd - target).max(), abs(Ed - (2 + 2 * q1 * (1 - q1))))
check(f"two options: P = q1^2 (1 + 2 q2) and E[looks] = 2 + 2 q1 q2, by quadrature and by k! e_k (max error {worst:.1e})", worst < 1e-12)

# 2. Birthday problem: 365 equal options.
_, E365 = pc.sr_dp(np.full(365, 1 / 365))
ref = sum(math.prod(1 - j / 365 for j in range(k)) for k in range(366))
check(f"365 equal options: E[looks] = {E365:.4f} (birthday problem 24.6166, direct sum {ref:.4f})", abs(E365 - 24.6166) < 5e-5 and abs(E365 - ref) < 1e-10)

# 3. Quadrature equals the exact elementary symmetric computation at the production sizes.
worst, worst_sum = 0.0, 0.0
for n in (3, 8, 16, 30, 32):
    for spread in (0.1, 1.0, 4.0, 12.0):
        q = np.exp(spread * rng.normal(size=n))
        q /= q.sum()
        P, E = sr_from_q(q)
        Pd, Ed = pc.sr_dp(q)
        worst = max(worst, np.abs(P / Pd - 1).max(), abs(E / Ed - 1))
        worst_sum = max(worst_sum, abs(P.sum() - 1))
check(f"quadrature equals the k! e_k computation for n up to 32 (max relative difference {worst:.1e}), probabilities sum to one (max error {worst_sum:.1e})",
      worst < 1e-12 and worst_sum < 1e-13)

# 4. Simulation of the race (sample until return).
q = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05])
P, E = sr_from_q(q)
S = 400_000
looks = rng.choice(len(q), size=(S, len(q) + 1), p=q)
seen = np.zeros((S, len(q)), bool)
chosen = np.full(S, -1)
nlooks = np.zeros(S)
rows = np.arange(S)
for t in range(len(q) + 1):
    col = looks[:, t]
    hit = (chosen < 0) & seen[rows, col]
    chosen[hit] = col[hit]
    nlooks[hit] = t + 1
    seen[rows, col] = True
freq = np.bincount(chosen, minlength=len(q)) / S
z = np.abs(freq - P) / np.sqrt(P * (1 - P) / S)
zE = abs(nlooks.mean() - E) / (nlooks.std() / np.sqrt(S))
check(f"simulated race: choice frequencies within {z.max():.1f} s.e. of the exact values, mean looks {nlooks.mean():.4f} against {E:.4f} ({zE:.1f} s.e.)", z.max() < 4.5 and zE < 4.5)

# 5. Simulation of accept in reading order.
v = np.array([0.4, 1.0, 0.7, 1.0, 0.2, 0.9])
lam = 2.5
logP, _, insp = pc.acc_batch(lam, v[None], np.ones((1, len(v)), bool), need_grad=False)
P = np.exp(logP[0])
a = 1 / (1 + np.exp(-lam * (v - v.max())))
choice = np.full(S, -1)
count = np.zeros(S)
pos = 0
while (choice < 0).any():
    live = choice < 0
    count[live] += 1
    acc = live & (rng.random(S) < a[pos])
    choice[acc] = pos
    pos = (pos + 1) % len(v)
freq = np.bincount(choice, minlength=len(v)) / S
z = np.abs(freq - P) / np.sqrt(P * (1 - P) / S)
zE = abs(count.mean() - insp[0]) / (count.std() / np.sqrt(S))
check(f"simulated reading-order acceptance: frequencies within {z.max():.1f} s.e., inspections {count.mean():.4f} against {insp[0]:.4f} ({zE:.1f} s.e.)", z.max() < 4.5 and zE < 4.5)
hand = a * np.cumprod(np.concatenate([[1.0], 1 - a[:-1]]))
check("accept in reading order equals a_i prod_{j<i}(1 - a_j) normalised", np.allclose(P, hand / hand.sum(), atol=1e-15))

# 6. Analytic derivatives against central differences, with padded options.
worst = 0.0
with np.errstate(invalid="ignore"):
    for fn in (pc.sr_batch, pc.acc_batch):
        for lam in (0.0, 0.3, 2.0, 7.0):
            V = rng.normal(size=(3, 9))
            mask = np.ones((3, 9), bool)
            mask[1, 6:] = False
            _, d, _ = fn(lam, V, mask)
            h = 1e-5
            f = lambda x: fn(x, V, mask, need_grad=False)[0]  # noqa: E731
            if lam - h >= 0:  # central difference, error O(h^2)
                fd = (f(lam + h) - f(lam - h)) / (2 * h)
            else:  # second-order one-sided difference at the boundary, error O(h^2)
                fd = (-3 * f(lam) + 4 * f(lam + h) - f(lam + 2 * h)) / (2 * h)
            worst = max(worst, float(np.abs(np.where(mask, d - fd, 0.0)).max()))
check(f"analytic d log P / d lambda equals finite differences for both search models (max abs difference {worst:.1e})", worst < 1e-6)

# 7. EBA effort by hand.
#    o1 {a1, a2}, o2 {a1}, o3 {}; weights 0.6 and 0.4, so W = W_split = 1 at the start.
#    checks({o1, o2}) = 2 * 1.0 / 0.4 + checks({o1}) = 5, checks(all) = 3 * 1 + 0.6 * 5 + 0.4 * 0 = 6.
val = pc.eba_checks(np.array([[1, 1], [1, 0], [0, 0]], bool), np.array([0.6, 0.4]))
check(f"EBA effort hand example equals 6 (got {val:.12f})", abs(val - 6) < 1e-12)
#    A non-splitting aspect still counts in W and A: o1 {a1, a3}, o2 {a3}, weights 0.5, 0.2, 0.3 with a3 held by all.
#    checks(all) = 2 * 1.0 / 0.5 + 1 * checks({o1}) = 4.
val = pc.eba_checks(np.array([[1, 0, 1], [0, 0, 1]], bool), np.array([0.5, 0.2, 0.3]))
check(f"EBA effort with a non-splitting aspect equals 4 (got {val:.12f})", abs(val - 4) < 1e-12)
check("EBA effort with no splitting aspect equals |S| A", pc.eba_checks(np.array([[1, 0], [1, 0]], bool), np.array([0.5, 0.5])) == 4)
check("EBA effort of a single option is zero", pc.eba_checks(np.array([[1, 1]], bool), np.array([0.5, 0.5])) == 0)

# 8. Time law, forecast and regret statistics by hand.
eff = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
sec = 10 + 2 * eff
pred, err = pc.time_law(eff, sec)
check("time law recovers an exact line t0 + tau effort with zero error", np.allclose(pred, sec) and err.max() < 1e-12)
pred, _ = pc.time_law(eff, 40 - 2 * eff)  # decreasing: NNLS keeps tau at 0, so predictions are means of the others
check("time law with a decreasing relation falls back to the mean (tau = 0)", np.allclose(pred, [np.mean(np.delete(40 - 2 * eff, j)) for j in range(6)]))
tg = np.array([1.0, 2.0, 4.0, 3.0, 6.0, 5.0])
fs = pc.forecast_stats(eff, tg)
loo = [abs(np.polyval(np.polyfit(np.delete(eff, j), np.delete(tg, j), 1), eff[j]) - tg[j]) for j in range(6)]
check("calibration error equals leave-one-out line fits (numpy.polyfit)", abs(fs["error"] - np.mean(loo)) < 1e-12)
check("constant error equals leave-one-out means", abs(fs["constant_error"] - np.mean([abs(np.delete(tg, j).mean() - tg[j]) for j in range(6)])) < 1e-12)
check("Spearman with ties uses average ranks, and a constant forecast gives 0",
      abs(pc.spearman0(np.array([8, 8, 16, 16, 32, 32.0]), tg) - np.corrcoef([1.5, 1.5, 3.5, 3.5, 5.5, 5.5], [1, 2, 4, 3, 6, 5])[0, 1]) < 1e-12
      and pc.spearman0(np.ones(6), tg) == 0.0)
rs = pc.regret_stats(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]))
check("regret error is the plain mean absolute error, constant error the leave-one-out mean error",
      abs(rs["error"] - 7 / 3) < 1e-12 and abs(rs["constant_error"] - (abs(25.5 - 12) + abs(22.5 - 18) + abs(15 - 33)) / 3) < 1e-12)

# 9. Overload measures by hand on a small array.
Sx = rng.integers(1, 6, size=(7, 12, 6)).astype(float)
A = ov.aligned(Sx)
check("aligned answers: 6 - s1, 6 - s2, s3..s6", np.array_equal(A[..., :2], 6 - Sx[..., :2]) and np.array_equal(A[..., 2:], Sx[..., 2:]))
rows = (A - A.mean(axis=1, keepdims=True)).reshape(-1, 6)
alpha_hand = 6 / 5 * (1 - sum(np.var(rows[:, j], ddof=1) for j in range(6)) / np.var(rows.sum(axis=1), ddof=1))
out = ov.analyse(Sx, rng.integers(0, 101, size=(7, 12)).astype(float), rng.uniform(5, 300, size=(7, 12)))
check("Cronbach's alpha within participants matches a hand computation", abs(out["alpha_within_person"] - alpha_hand) < 1e-12)
X = rng.normal(size=(7, 6))
R = ov.double_center(X)
check("double centering leaves zero row and column means", np.abs(R.mean(axis=0)).max() < 1e-12 and np.abs(R.mean(axis=1)).max() < 1e-12)

# 10. The "both" scope is the two tasks' trials concatenated, file trials first.
fake = {t: {"fc": {f: np.arange(6) + (0 if t == "file" else 10) for f in rp.FORECASTS},
            "rfc": {k: np.arange(6.0) for k in rp.REGRET_MODELS},
            "ov_target": np.arange(6.0) + (0 if t == "file" else 6), "rg_target": np.arange(6.0)} for t in ("file", "laptop")}
fc, _, ov_t, _ = rp.scope_inputs(fake, "both")
check("both scope stacks file trials before laptop trials", np.array_equal(fc["options"], np.r_[np.arange(6), np.arange(6) + 10]) and np.array_equal(ov_t, np.arange(12.0)))

# 11. Real data: every search lambda is the global maximum (dense evaluation) and a stationary point.
if '--skip-dense' in sys.argv:
    print('\nALL CHECKS PASSED' if ok_all else '\nSOME CHECKS FAILED')
    sys.exit(0)
dense = np.concatenate([[0.0], np.logspace(-5, 5, 5001)])
for tname in core.TASKS:
    task = core.build_task(tname)
    cnt = core.counts_from_weights(task, np.ones(task["P"]))
    for name in pc.SEARCH_MODELS:
        sm = pc.SearchModel(task, name)
        worst_gap, worst_score, n_multi, unrefined, lams = -np.inf, 0.0, 0, 0, []
        for h in range(6):
            tr = np.array([c for c in range(6) if c != h])
            lam, info = sm.fit(cnt, tr)
            lams.append(lam)
            n_multi += info["n_local"] > 1
            unrefined += not info["refined"]
            ll_dense = max(sm.loglik(x, cnt, tr) for x in dense)
            worst_gap = max(worst_gap, ll_dense - info["loglik"])
            if lam > 0:
                worst_score = max(worst_score, abs(sm.score(lam, cnt, tr)) / max(1.0, abs(info["loglik"])))
            elif sm.score(0.0, cnt, tr) > 0:
                worst_score = np.inf
        check(f"{tname} {name}: lambdas {min(lams):.4g} to {max(lams):.4g}; no dense lambda beats the fit (largest excess {worst_gap:.1e}); "
              f"relative score at the fit {worst_score:.1e}; folds with several grid maxima {n_multi}, unrefined {unrefined}",
              worst_gap <= 1e-9 and worst_score < 1e-10 and unrefined == 0)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
