"""Hand checks of the independent implementation, then optimality checks on the real data.

Prints only derived quantities (no participant or choice counts).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


# ----------------------------------------------------------------------------- softmax closed forms
# Two options with scores (0, d): q1/q0 = exp(lambda d), so the MLE is ln(n1/n0)/d.
d = 2.5
V = np.zeros((1, 2, 4))
V[0, 0, :2] = [0.0, d]
V[0, 1, :2] = [0.0, d]
valid = np.zeros((2, 4), bool)
valid[:, :2] = True
cnt = np.zeros((2, 4))
cnt[0, :2] = [3, 9]
cnt[1, :2] = [5, 5]
T = np.array([[True, False], [True, True]])  # fold 0: trial 0 only, fold 1: pooled
lam = core.fit_softmax(V, valid, cnt, T)
check("two-option softmax, one trial: lambda = ln(3)/d", abs(lam[0, 0] - math.log(3) / d) < 1e-13)
check("two-option softmax, pooled: lambda = ln(14/8)/d", abs(lam[0, 1] - math.log(14 / 8) / d) < 1e-13)
cnt2 = cnt.copy()
cnt2[0, :2] = [9, 3]
lam2 = core.fit_softmax(V, valid, cnt2, np.array([[True, False]]))
check("two-option softmax, negative derivative at 0: lambda = 0", lam2[0, 0] == 0.0)

# Three options with general scores against brentq on the same score function.
rng = np.random.default_rng(7)
V3 = np.zeros((1, 3, 5))
valid3 = np.zeros((3, 5), bool)
cnt3 = np.zeros((3, 5))
for c, n in enumerate([3, 5, 4]):
    V3[0, c, :n] = rng.normal(size=n)
    valid3[c, :n] = True
    cnt3[c, :n] = rng.integers(1, 9, size=n) + 3 * np.argsort(np.argsort(V3[0, c, :n]))  # favour high scores
T3 = np.ones((1, 3), bool)
lam3 = core.fit_softmax(V3, valid3, cnt3, T3)[0, 0]


def score_scalar(x):
    s = 0.0
    for c in range(3):
        v = V3[0, c, valid3[c]]
        k = cnt3[c, valid3[c]]
        w = np.exp(x * v - (x * v).max())
        w /= w.sum()
        s += (k * v).sum() - k.sum() * (w * v).sum()
    return s


if score_scalar(0.0) > 0:
    ref = brentq(score_scalar, 0.0, 100.0, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)
    check(f"general softmax agrees with brentq (rel diff {abs(lam3 - ref) / ref:.1e})", abs(lam3 - ref) / ref < 1e-12)
else:
    check("general softmax gives 0 when score at 0 <= 0", lam3 == 0.0)

# ----------------------------------------------------------------------------- EBA by hand
# Aspects: A{1}, B{2}, C{3}, D{1,2}, E{2,3}, weights (0.5, 0.3, 0.2).
# Draw 1 (0.5) -> {A, D} -> aspect 2 splits -> D. Draw 3 (0.2) -> {C, E} -> aspect 2 -> E.
# Draw 2 (0.3) -> {B, D, E}: aspects 1 and 3 split with weights 0.5/0.7 (-> D) and 0.2/0.7 (-> E).
H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], bool)
w = np.array([0.5, 0.3, 0.2])
P = core.eba_from_aspects(H, w)
expect = np.array([0, 0, 0, 0.5 + 0.3 * 0.5 / 0.7, 0.2 + 0.3 * 0.2 / 0.7])
check(f"EBA hand example {np.round(P, 6)}", np.allclose(P, expect, atol=1e-15))

# Ties and the uniform rule: A{1,2}, B{1,2}, C{1}, D{2}, E{} -> A and B share everything, 1/2 each.
H2 = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 0]], bool)
P2 = core.eba_from_aspects(H2, np.array([0.7, 0.3]))
check("EBA uniform choice among aspect-identical survivors", np.allclose(P2, [0.5, 0.5, 0, 0, 0]))

# Median aspects: column [1, 2, 3, 3] has median 2.5, column [1, 1, 1, 2] has median 1 (all hold it),
# column [4, 1, 2, 2] has median 2 (options 0, 2, 3). r = (0.5, 0.25, 0.25).
Xm = np.array([[1, 1, 4], [2, 1, 1], [3, 1, 2], [3, 2, 2]], float)
Hm, wm = core.eba_aspects(Xm, np.array([0.5, 0.25, 0.25]), "median")
check("median aspects", (Hm == np.array([[0, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1]], bool)).all())
# Aspect 1 splits {0..3} -> {2, 3}; aspect 3 splits -> {0, 2, 3}, then aspect 1 -> {2, 3}.
# Options 2 and 3 have identical aspects -> 1/2 each.
Pm = core.eba_probs(Xm, np.array([0.5, 0.25, 0.25]), "median")
check(f"EBA median example {np.round(Pm, 6)}", np.allclose(Pm, [0, 0, 0.5, 0.5]))
Hb, _ = core.eba_aspects(Xm, np.array([0.5, 0.25, 0.25]), "best")
check("best-level aspects", (Hb == np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0]], bool)).all())
# With best aspects: aspects 1 {2,3}, 2 {3}, 3 {0}. All split the full set; weights .5 .25 .25.
# Draw 1 (.5) -> {2,3} -> aspect 2 splits -> 3. Draw 2 (.25) -> 3. Draw 3 (.25) -> 0.
Pb = core.eba_probs(Xm, np.array([0.5, 0.25, 0.25]), "best")
check(f"EBA best example {np.round(Pb, 6)}", np.allclose(Pb, [0.25, 0, 0, 0.75]))

# ----------------------------------------------------------------------------- lapse closed form
# Two options with EBA probabilities (1, 0): P1 = eps/2, so the MLE is eps = 2 n1 / N when <= 1.
A = np.zeros((1, 1, 2))
A[0, 0] = [1.0, 0.0]
cntl = np.array([[3.0, 1.0]])
eps = core.fit_lapse(A, np.ones((1, 2), bool), cntl, np.array([2.0]), np.ones((1, 1), bool))
check(f"lapse closed form eps = 0.5 (got {eps[0, 0]:.15f})", abs(eps[0, 0] - 0.5) < 1e-14)
cntl2 = np.array([[1.0, 3.0]])  # 2 n1/N = 1.5 > 1, so eps = 1
eps2 = core.fit_lapse(A, np.ones((1, 2), bool), cntl2, np.array([2.0]), np.ones((1, 1), bool))
check("lapse at upper bound 1", eps2[0, 0] == 1.0)
A3 = np.zeros((1, 1, 2))
A3[0, 0] = [0.5, 0.5]
eps3 = core.fit_lapse(A3, np.ones((1, 2), bool), cntl, np.array([2.0]), np.ones((1, 1), bool))
check("lapse 0 when EBA is already uniform-optimal direction (score <= 0)", eps3[0, 0] == 0.0)
# three options a = (0.6, 0.4, 0), counts (5, 1, 2): compare with brentq.
A4 = np.zeros((1, 1, 3))
A4[0, 0] = [0.6, 0.4, 0.0]
c4 = np.array([[5.0, 1.0, 2.0]])
e4 = core.fit_lapse(A4, np.ones((1, 3), bool), c4, np.array([3.0]), np.ones((1, 1), bool))[0, 0]
g = lambda e: sum(k * (1 / 3 - a) / ((1 - e) * a + e / 3) for k, a in zip(c4[0], A4[0, 0]))  # noqa: E731
ref4 = brentq(g, 1e-12, 1 - 1e-12, xtol=1e-16, rtol=4 * np.finfo(float).eps)
check(f"lapse agrees with brentq (rel diff {abs(e4 - ref4) / ref4:.1e})", abs(e4 - ref4) / ref4 < 1e-12)

# ----------------------------------------------------------------------------- measures by hand
q = np.array([[[0.5, 0.25, 0.25, 0.0]]])
cm = np.array([[2.0, 1.0, 1.0, 0.0]])
nv = np.array([3.0])
L = core.log_scores(q, cm, nv)[0, 0]
hand = (2 * math.log2(1.5) + 2 * math.log2(0.75)) / 4
check(f"log score by hand ({L:.6f})", abs(L - hand) < 1e-14)
Ls = core.lstar(cm, nv)[0]
hand_s = math.log2(3) - 1.5 - 2 / (2 * 4 * math.log(2))
check(f"L* by hand ({Ls:.6f})", abs(Ls - hand_s) < 1e-14)
O = core.overlaps(np.array([[[0.2, 0.3, 0.5, 0.0]]]), cm)[0, 0]
check("overlap by hand", abs(O - (0.2 + 0.25 + 0.25)) < 1e-15)
check("effective number of uniform over 4 is 4", abs(core.effective_number(np.ones(4) / 4) - 4) < 1e-12)
check("sampled effective number of a point mass is 1",
      core.effective_number_sampled(np.array([1.0, 0, 0]), 10, (1, 0, 0, 0)) == 1.0)
check("completeness by hand", abs(core.completeness(np.array([0.5, 1.0]), np.array([1.0, 2.0])) - 50.0) < 1e-12)
check("holm", core.holm([0.01, 0.04]) == [0.02, 0.04] and core.holm([0.04, 0.03]) == [0.06, 0.06])
check("bootstrap p", core.boot_p(np.array([-1, 1, 2, 3])) == 0.5)
check("slope", abs(core.ls_slope(np.array([3.0, 4.0, 5.0]), np.array([1.0, 3.0, 5.0])) - 2.0) < 1e-14)
s = core.summary_centered(1.0, np.array([-0.5, 0.0, 0.5]))
check("centered intervals are mirror images", np.allclose(s["percentile"], [0.525, 1.475]) and np.allclose(s["basic"], [0.525, 1.475]))

# ----------------------------------------------------------------------------- comparison scores by hand
X = np.array([[1, 2], [2, 2], [3, 1]], float)
r = np.array([0.75, 0.25])
dbs, mlba = core.comparison_scores(X, r)
check(f"DbS_sum by hand {dbs}", np.allclose(dbs, [-1.25, 0.25, 1.0]))
e1, e2 = math.exp(-1), math.exp(-2)
m_hand = [0.75 * (-e1 - 2 * e2) + 0.25 * e1, 0.25 * e1, 0.75 * (2 * e2 + e1) + 0.25 * (-2 * e1)]
check(f"MLBA_sum by hand {np.round(mlba, 8)}", np.allclose(mlba, m_hand, atol=1e-15))

# ----------------------------------------------------------------------------- real data optimality
for name in core.TASKS:
    task = core.build_task(name)
    cnt = core.counts_from_weights(task, np.ones(task["P"]))
    C = cnt.shape[0]
    T = core.fold_masks(C, include_full=True)
    lam = core.fit_softmax(task["V"], task["valid"], cnt, T)
    S, D = core.softmax_score(lam, task["V"], task["valid"], cnt, T)
    S0, _ = core.softmax_score(np.zeros_like(lam), task["V"], task["valid"], cnt, T)
    interior = lam > 0
    rel = np.abs(S[interior]) / np.abs(D[interior] * lam[interior])  # relative step left to the root
    check(f"{name}: softmax score ~ 0 at interior optima (max relative Newton step {rel.max():.1e})", rel.max() < 1e-10)
    check(f"{name}: lambda = 0 only with nonpositive score at 0 ({(~interior).sum()} such fits)",
          bool(np.all(S0[~interior] <= 1e-9 * np.abs(S0).max())))
    ll = core.softmax_loglik(lam, task["V"], task["valid"], cnt, T)
    worse = True
    for f in (1 - 1e-6, 1 + 1e-6):
        worse &= bool(np.all(core.softmax_loglik(lam * f, task["V"], task["valid"], cnt, T) <= ll + 1e-9))
    check(f"{name}: log likelihood lower at lambda * (1 +- 1e-6)", worse)
    # brentq on every (model, fold) as an independent root finder
    maxrel = 0.0
    for m in range(lam.shape[0]):
        for f in range(lam.shape[1]):
            if lam[m, f] == 0:
                continue

            def sc(x, m=m, f=f):
                s_, _ = core.softmax_score(np.array([[x]]), task["V"][m:m + 1], task["valid"], cnt, T[f:f + 1])
                return s_[0, 0]

            ref = brentq(sc, 0.0, 10 * lam[m, f], xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
            maxrel = max(maxrel, abs(ref - lam[m, f]) / ref)
    check(f"{name}: every lambda agrees with brentq (max rel diff {maxrel:.1e})", maxrel < 1e-10)
    # lapse
    eps = core.fit_lapse(task["A"], task["valid"], cnt, task["nvec"], T)
    Se, De = core.lapse_score(eps, task["A"], task["valid"], cnt, task["nvec"], T)
    inner = (eps > 0) & (eps < 1)
    rel_e = np.abs(Se[inner] / (De[inner] * eps[inner]))
    check(f"{name}: lapse score ~ 0 at interior optima (max relative step {rel_e.max() if inner.any() else 0:.1e}); "
          f"{(~inner).sum()} boundary fits", (not inner.any()) or rel_e.max() < 1e-10)
    lle = core.lapse_loglik(eps, task["A"], task["valid"], cnt, task["nvec"], T)
    ok = True
    for f in (1 - 1e-6, 1 + 1e-6):
        ok &= bool(np.all(core.lapse_loglik(np.clip(eps * f, 0, 1), task["A"], task["valid"], cnt, task["nvec"], T) <= lle + 1e-9))
    check(f"{name}: lapse log likelihood lower at eps * (1 +- 1e-6)", ok)
    # EBA probabilities are distributions
    sums = task["A"].sum(axis=2)
    check(f"{name}: EBA probabilities sum to one", np.allclose(sums, 1.0, atol=1e-12))
    # beta = 1 on averaged scores reproduces the summed models up to the factor 7
    ev = core.evaluate(task, cnt, include_full=True)
    mi = {m: i for i, m in enumerate(core.ALL_MODELS)}
    for avg, summ in (("DbS_avg", "DbS_sum"), ("MLBA_avg", "MLBA_sum")):
        a_ = mi[core.beta_name(avg, 1.0)]
        s_ = mi[summ]
        same = np.allclose(ev["L_ho"][a_], ev["L_ho"][s_], atol=1e-10) and np.allclose(ev["L_fit"][a_], ev["L_fit"][s_], atol=1e-10)
        ratio = ev["lam"][s_] / ev["lam"][a_]
        check(f"{name}: {avg} beta=+1 equals {summ} (lambda ratio {ratio.min():.12f}..{ratio.max():.12f})",
              same and np.allclose(ratio, 1 / 7, rtol=1e-9))
    pr = np.isclose(ev["Q_ho"].sum(axis=2), 1.0).all()
    check(f"{name}: predictions sum to one", pr)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
