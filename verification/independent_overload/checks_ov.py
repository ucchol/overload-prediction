"""Checks of the Amendment 8 implementation on simulated data only (brief tests 1, 2 and 4, loader and features).

Usage: python -I checks_ov.py SIM_DIR   (SIM_DIR holds trials_cl.dta from simulate_ov.py)
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ov_core as oc  # noqa: E402
from ov_core import cc, srch  # noqa: E402

srch.assert_local(oc, cc, srch)
SIM = sys.argv[1]
rng = np.random.default_rng(31)
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def panel(n_list, V_list, chosen):
    n = np.array(n_list)
    return oc.make_panel("test", n, V_list, np.array(chosen), np.zeros(len(n), int), "reutskaja")


def race_brute(q):
    """Race probabilities by enumerating every sequence of distinct looks followed by a return."""
    n = len(q)
    P = np.zeros(n)
    for k in range(1, n + 1):
        for seq in itertools.permutations(range(n), k):
            p = np.prod(q[list(seq)])
            for i in seq:
                P[i] += p * q[i]
    return P


# ----------------------------------------------------------------------------- 1. miss probabilities
worst = 0.0
Vs = [np.array(v, float) for v in ([3, 5, 5, 1], [2, 7, 4, 7, 7], [1, 2, 3, 4], [6, 6, 6, 6], [9, 1, 4, 8, 2])]
ns = [len(v) for v in Vs]
for lam in (0.0, 0.4, 1.3):
    for m in oc.MODELS:
        for v, n in zip(Vs, ns):
            P = panel([4, 9, 16], [np.array([1.0, 2, 3, 4]), np.ones(9), np.ones(16)], [0, 0, 0])  # placeholder sizes
            # direct computation on this display
            s = lam * v * (oc.f_comp(n) if m == "CL_comp" else 1.0)
            q = np.exp(s - s.max())
            q /= q.sum()
            top = v == v.max()
            direct = 1 - (race_brute(np.exp(lam * v - (lam * v).max()) / np.exp(lam * v - (lam * v).max()).sum())[top].sum() if m == "SR" else q[top].sum())
            G = type("G", (), {})()
            pf = object.__new__(oc.PanelFits)
            pf.items = [(n, np.array([0]), v[None, :], np.array([0]))]
            pf.S = {"CL": [v[None, :]], "CL_comp": [oc.f_comp(n) * v[None, :]]}
            pf.top_idx = [np.array([int(np.argmax(v))])]
            pf.ntop = [np.array([int(top.sum())])]
            worst = max(worst, abs(pf.miss_prob(m, lam, 0)[0] - direct))
check(f"miss probabilities of CL, CL_comp and SR equal a direct computation on small sets with ties at the top "
      f"(SR by enumerating every sequence of looks; max difference {worst:.1e})", worst < 1e-13)
v = np.array([4.0, 9, 9, 2, 7, 1, 5, 8, 3])
lam = 0.35
pf = object.__new__(oc.PanelFits)
pf.items = [(9, np.array([0]), v[None, :], np.array([0]))]
pf.top_idx, pf.ntop = [np.array([1])], [np.array([2])]
p_miss = pf.miss_prob("SR", lam, 0)[0]
q = np.exp(lam * v - (lam * v).max())
q /= q.sum()
Sn = 300_000
looks = rng.choice(9, size=(Sn, 10), p=q)
seen = np.zeros((Sn, 9), bool)
choice = np.full(Sn, -1)
for t in range(10):
    c = looks[:, t]
    hit = (choice < 0) & seen[np.arange(Sn), c]
    choice[hit] = c[hit]
    seen[np.arange(Sn), c] = True
sim_miss = np.mean(v[choice] < v.max())
z = abs(sim_miss - p_miss) / math.sqrt(p_miss * (1 - p_miss) / Sn)
check(f"SR's miss probability against a simulation of the race (9 options, two tied at the top): {p_miss:.4f} against "
      f"{sim_miss:.4f} ({z:.1f} s.e.)", z < 4)


# ----------------------------------------------------------------------------- 2. recovery
def sim_panel(sizes, T, lam, model, levels):
    n_l, V_l, ch = [], [], []
    for n in sizes:
        for _ in range(T):
            v = rng.integers(levels[0], levels[1] + 1, size=n).astype(float)
            if model == "SR":
                q = np.exp(lam * v - (lam * v).max())
                q /= q.sum()
                seen = set()
                while True:
                    j = int(rng.choice(n, p=q))
                    if j in seen:
                        break
                    seen.add(j)
            else:
                s = lam * v * (oc.f_comp(n) if model == "CL_comp" else 1.0)
                q = np.exp(s - s.max())
                j = int(rng.choice(n, p=q / q.sum()))
            n_l.append(n)
            V_l.append(v)
            ch.append(j)
    return oc.PanelFits(panel(n_l, V_l, ch))


for model, lam_true, sizes, levels in (("CL", 0.3, (10, 20, 40), (1, 35)), ("CL_comp", 0.3, (10, 20, 40), (1, 35)),
                                       ("SR", 0.25, (10, 20, 40), (1, 35)), ("CL_comp", 0.8, (4, 9, 16), (1, 11)),
                                       ("SR", 0.7, (4, 9, 16), (1, 11))):
    pf = sim_panel(sizes, 700, lam_true, model, levels)
    w = np.ones(pf.P["T"])
    lam_hat, info = pf.fit(model, w)
    if model == "SR":
        ll = lambda x: pf.sr.loglik(x, w)  # noqa: E731
    else:
        def ll(x):
            return sum(float(cc.log_softmax(x, S)[np.arange(len(c)), c].sum()) for S, (_, _, _, c) in zip(pf.S[model], pf.items))
    h = 1e-4 * lam_hat
    se = 1 / math.sqrt(-(ll(lam_hat + h) - 2 * ll(lam_hat) + ll(lam_hat - h)) / h ** 2)
    extra = f", dense-grid excess {info['dense_excess']:.1e}" if model == "SR" else ""
    check(f"recovery of {model} at lambda {lam_true} on set sizes {sizes}: {lam_hat:.4f} ({abs(lam_hat - lam_true) / se:.1f} s.e.){extra}",
          abs(lam_hat - lam_true) < 4 * se and (model != "SR" or info["dense_excess"] <= 1e-9))

# ----------------------------------------------------------------------------- 4. weighted area under the curve
worst = 0.0
for rep in range(30):
    N = int(rng.integers(5, 60))
    score = rng.integers(0, 4, size=N).astype(float)  # many ties
    if rep % 3 == 0:
        score[rng.random(N) < 0.3] = -np.inf
    miss = (rng.random(N) < 0.4).astype(float)
    w = rng.integers(0, 4, size=N).astype(float)
    num = den = 0.0
    for i in range(N):
        for j in range(N):
            if miss[i] and not miss[j] and w[i] > 0 and w[j] > 0:
                ww = w[i] * w[j]
                den += ww
                num += ww * (1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
    a = oc.wauc(score, miss, w)
    if den == 0:
        worst = max(worst, 0.0 if np.isnan(a) else np.inf)
    else:
        worst = max(worst, abs(a - num / den))
check(f"weighted area under the curve equals a brute force pairwise count with ties and minus infinity "
      f"(max difference {worst:.1e}; undefined without a miss or a hit)", worst < 1e-13)

# ----------------------------------------------------------------------------- features, loader, real files
P = panel([4, 4, 9, 16], [np.array([5.0, 5, 4, 1]), np.array([3.0, 3, 3, 3]), np.array([8.0, 7, 7, 6, 1, 1, 1, 1, 1]),
                          np.arange(1.0, 17)], [2, 0, 0, 15])
feat_ok = (P["near"].tolist() == [1.0, 0.0, 2.0, 1.0] and P["minus_gap"][0] == -1 and np.isneginf(P["minus_gap"][1])
           and P["minus_gap"][2] == -1 and P["miss"].tolist() == [1.0, 0.0, 0.0, 0.0])
check("near best count, minus gap (minus infinity when every option ties at the top) and misses by hand", feat_ok)
R = oc.read_reutskaja(os.path.join(SIM, "trials_cl.dta"))
D = pd.read_stata(os.path.join(SIM, "trials_cl.dta"))
first = D.groupby(["subject", "trialid"]).first()
ok = all(np.array_equal(v, first[[f"rating{i}" for i in range(1, k + 1)]].to_numpy(float)[t] + 6) for t, (v, k) in enumerate(zip(R["V"], R["n"])))
ok &= np.array_equal(R["chosen"], first["chosendestring"].to_numpy(int) - 1)
fx = D.groupby(["subject", "trialid"])["Roinum"].apply(lambda s: set(s.astype(int)))
ok &= all(a == b for a, b in zip(R["fixated"], fx))
Pr = oc.make_panel("reutskaja", R["n"], R["V"], R["chosen"], np.zeros(len(R["n"]), int), "reutskaja", R["fixated"])
direct = [all(j + 1 not in f for j in np.flatnonzero(v == v.max())) for v, f in zip(R["V"], fx)]
ok &= np.array_equal(Pr["best_unseen"], np.array(direct))
check("Stata loader: ratings shifted by 6 in screen order, chosen index, fixated positions and best never fixated agree "
      "with a direct reading", ok)
refused = []
for fn, path in ((oc.read_reutskaja, os.path.join(oc.REUT_DIR, oc.REUT_FILE)), (cc.load_cdm, os.path.join(cc.CDM_DIR, cc.CLICK_FILE))):
    try:
        fn(path)
        refused.append(False)
    except PermissionError:
        refused.append(True)
check("both loaders refuse the real files unless explicitly allowed (nothing is read)", all(refused))
print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
