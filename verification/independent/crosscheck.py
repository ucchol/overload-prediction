"""Cross-checks of the vectorized pipeline by deliberately plain code, plus Amendment 1 diagnostics.

Prints derived quantities only (no participant or choice counts).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
boot = np.load(os.path.join(HERE, "boot_B1000.npz"))
MI = {m: i for i, m in enumerate(core.ALL_MODELS)}


def plain_scores(t, r, model):
    """Scores written out longhand from the Section 3 table."""
    X, n = t["X"], t["n"]
    K = X.shape[1]
    if model == "QR":
        return t["U"].copy()
    if model == "CL":
        return np.array([sum(r[k] * X[i, k] for k in range(K)) for i in range(n)])
    out = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(K):
                d = X[i, k] - X[j, k]
                out[i] += r[k] * (np.sign(d) if model.startswith("DbS") else np.exp(-abs(d)) * d)
    if model.endswith("avg"):
        out /= n - 1
    return out


def plain_fit(vs, cs):
    """Maximise the pooled log likelihood over lambda >= 0 with a bounded scalar search."""
    def nll(lam):
        return -sum((c * (lam * v - logsumexp(lam * v))).sum() for v, c in zip(vs, cs))
    hi = 1.0
    while nll(2 * hi) < nll(hi):
        hi *= 2
    res = minimize_scalar(nll, bounds=(0.0, 4 * hi), method="bounded", options={"xatol": 1e-12, "maxiter": 1000})
    return res.x if nll(res.x) < nll(0.0) else 0.0


print("== plain refit of R0 softmax models (held out completeness, fitted completeness)")
for name in core.TASKS:
    task = core.build_task(name)
    cnt = core.counts_from_weights(task, np.ones(task["P"]))
    trials = task["trials"]
    Ls = core.lstar(cnt, task["nvec"])
    for model in ("QR", "CL", "DbS_avg", "MLBA_avg", "DbS_sum", "MLBA_sum"):
        vs = [plain_scores(t, task["r"], model) for t in trials]
        cs = [cnt[c, : t["n"]] for c, t in enumerate(trials)]
        L_ho = []
        for h in range(6):
            lam = plain_fit([v for c, v in enumerate(vs) if c != h], [x for c, x in enumerate(cs) if c != h])
            q = np.exp(lam * vs[h] - logsumexp(lam * vs[h]))
            L_ho.append((cs[h] * np.log2(trials[h]["n"] * q)).sum() / cs[h].sum())
        lam_all = plain_fit(vs, cs)
        L_fit = [(c_ * np.log2(t["n"] * np.exp(lam_all * v - logsumexp(lam_all * v)))).sum() / c_.sum()
                 for v, c_, t in zip(vs, cs, trials)]
        K_ho = 100 * sum(L_ho) / Ls.sum()
        K_fit = 100 * sum(L_fit) / Ls.sum()
        e = R[name]["R0"][model]
        print(f"{name:6s} {model:9s} plain K_ho {K_ho:8.4f} vs results {e['completeness_ho']['point']:8.4f} | "
              f"plain K_fit {K_fit:8.4f} vs {e['completeness_fit']:8.4f} | lambda {lam_all:.6g} vs {e['lambda']:.6g}")

print("\n== EBA by Monte Carlo simulation of the elimination process (max abs error vs exact)")
rng = np.random.default_rng(123)
for name in core.TASKS:
    task = core.build_task(name)
    for c, t in enumerate(task["trials"]):
        for e_i, rule in enumerate(("median", "best")):
            H, w = core.eba_aspects(t["X"], task["r"], rule)
            n = t["n"]
            sims = 20000
            hits = np.zeros(n)
            for _ in range(sims):
                S = list(range(n))
                while True:
                    held = H[S].sum(axis=0)
                    split = np.where((held > 0) & (held < len(S)))[0]
                    if len(split) == 0:
                        hits[S[rng.integers(len(S))]] += 1
                        break
                    k = rng.choice(split, p=w[split] / w[split].sum())
                    S = [i for i in S if H[i, k]]
            err = np.abs(hits / sims - task["A"][e_i, c, :n]).max()
            print(f"{name:6s} {t['name']:7s} {rule:6s} max abs error {err:.4f} (MC s.e. at p=0.5 is {0.5 / np.sqrt(sims):.4f})")

print("\n== Amendment 1 diagnostics for CL (beta = 0), file and laptop")
tasks = {n: core.build_task(n) for n in core.TASKS}
B = boot["file__L"].shape[0]
ratio = {n: np.zeros(B) for n in core.TASKS}
K_re = {n: np.zeros(B) for n in core.TASKS}
for b in range(B):
    rng = np.random.default_rng(10000 + b)
    draws = {n: rng.integers(0, tasks[n]["P"], size=tasks[n]["P"]) for n in core.TASKS}
    for n in core.TASKS:
        t = tasks[n]
        cnt = core.counts_from_weights(t, np.bincount(draws[n], minlength=t["P"]).astype(float))
        Ls_b = core.lstar(cnt, t["nvec"])
        ratio[n][b] = Ls_b.sum()
        K_re[n][b] = 100 * boot[f"{n}__L"][b, MI[core.beta_name('CL', 0.0)]].sum() / Ls_b.sum()
for n in core.TASKS:
    t = tasks[n]
    Ls = core.lstar(core.counts_from_weights(t, np.ones(t["P"])), t["nvec"])
    K_fix = 100 * boot[f"{n}__L"][:, MI[core.beta_name('CL', 0.0)]].sum(axis=1) / Ls.sum()
    O_b = boot[f"{n}__O"][:, MI[core.beta_name('CL', 0.0)]].mean(axis=1)
    print(f"{n:6s} mean replicate benchmark / full sample benchmark - 1 = {100 * (ratio[n].mean() / Ls.sum() - 1):.1f} percent")
    print(f"{n:6s} re-estimated benchmark: CL percentile [{np.percentile(K_re[n], 2.5):.1f}, {np.percentile(K_re[n], 97.5):.1f}], median {np.median(K_re[n]):.1f}")
    print(f"{n:6s} fixed benchmark: CL replicate median {np.median(K_fix):.1f}; CL overlap replicate median {100 * np.median(O_b):.1f}")
