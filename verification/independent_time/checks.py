"""Self checks of the independent Amendment 12 implementation, on simulated inputs only."""
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import time_core as tc  # noqa: E402


def esym(q):
    e = np.zeros(len(q) + 1)
    e[0] = 1.0
    for x in q:
        e[1:] = e[1:] + x * e[:-1]
    return e


def exact_sr(q):
    """Direct formula with elementary symmetric polynomials by recursion (small n)."""
    n = len(q)
    p = np.array([q[i] ** 2 * sum(math.factorial(k + 1) * c for k, c in enumerate(esym(np.delete(q, i))))
                  for i in range(n)])
    looks = sum(math.factorial(k) * c for k, c in enumerate(esym(q)))
    return p, looks


def brute_sr(q):
    """Exact enumeration over draw sequences that stop at the first repeat."""
    n = len(q)
    p = np.zeros(n)
    looks = 0.0
    for k in range(1, n + 1):
        for seq in itertools.permutations(range(n), k):
            pr = np.prod(q[list(seq)])
            for i in seq:  # the (k+1)-th draw repeats i
                p[i] += pr * q[i]
                looks += pr * q[i] * (k + 1)
    return p, looks


def simulate_sr(q, rng, reps=200000):
    n = len(q)
    counts = np.zeros(n)
    total = 0
    draws = rng.choice(n, size=(reps, n + 1), p=q)
    for r in range(reps):
        seen = set()
        for j, x in enumerate(draws[r]):
            if x in seen:
                counts[x] += 1
                total += j + 1
                break
            seen.add(x)
    return counts / reps, total / reps


def main():
    rng = np.random.default_rng(1)
    ok = True

    # 1. SR probabilities sum to one, match the polynomial formula, enumeration and simulation
    for n in (2, 3, 5, 7):
        v = rng.uniform(0, 1, n)
        for lam in (0.0 + 1e-9, 1.0, 5.0):
            q = tc.softmax_rows(lam, v[None, :])
            p = tc.sr_probs(q)[0]
            looks = math.exp(tc.sr_log_looks(q)[0])
            pe, le = exact_sr(q[0])
            ok &= abs(p.sum() - 1) < 1e-12 and np.allclose(p, pe, atol=1e-12) and abs(looks - le) < 1e-10
            if n <= 5:
                pb, lb = brute_sr(q[0])
                ok &= np.allclose(p, pb, atol=1e-12) and abs(looks - lb) < 1e-10
    q = tc.softmax_rows(3.0, rng.uniform(0, 1, 4)[None, :])
    ps, ls = simulate_sr(q[0], rng)
    p = tc.sr_probs(q)[0]
    l = math.exp(tc.sr_log_looks(q)[0])
    sim_ok = np.max(np.abs(ps - p)) < 0.005 and abs(ls - l) < 0.02
    print("SR vs simulation", np.round(p, 4), np.round(ps, 4), round(l, 4), round(ls, 4), sim_ok)
    ok &= sim_ok
    # large n: sums to one, chosen log prob agrees with full distribution, padding does not matter
    for n in (16, 36, 40):
        V = rng.uniform(0, 1, (5, n))
        for lam in (0.01, 3.0, 30.0, 300.0):
            q = tc.softmax_rows(lam, V)
            P = tc.sr_probs(q)
            ok &= np.allclose(P.sum(axis=1), 1, atol=1e-10)
            pe, le = exact_sr(q[0])
            ok &= np.allclose(P[0], pe, rtol=1e-8, atol=1e-14)
            ok &= abs(np.exp(tc.sr_log_looks(q)[0]) - le) / le < 1e-9
            Vp = np.hstack([V, np.full((5, 4), np.nan)])
            qp = tc.softmax_rows(lam, Vp)
            ch = rng.integers(0, n, 5)
            ok &= np.allclose(tc.sr_log_prob(qp, ch), np.log(P[np.arange(5), ch]), rtol=1e-9)
    print("SR checks", ok)

    # 2. CL entropy and fitting recover a known sensitivity
    V = rng.uniform(0, 1, (4000, 6))
    q = tc.softmax_rows(4.0, V)
    ch = np.array([rng.choice(6, p=r) for r in q])
    lam = tc.fit_lambda("cl", V, ch, np.ones(len(ch)))
    ok_cl = abs(lam - 4.0) < 0.4 and np.allclose(tc.cl_entropy_bits(np.full((1, 8), 1 / 8)), 3.0)
    qs = tc.sr_probs(tc.softmax_rows(4.0, V))
    ch = np.array([rng.choice(6, p=r / r.sum()) for r in qs])
    lam_sr = tc.fit_lambda("sr", V, ch, np.ones(len(ch)))
    ok_cl &= abs(lam_sr - 4.0) < 0.6
    print("fit CL", round(lam, 3), "fit SR", round(lam_sr, 3), ok_cl)
    ok &= ok_cl

    # 3. weighted regression recovers known coefficients, minimum norm with a zero and a collinear column
    X = rng.normal(size=(3000, 4))
    X[:, 3] = X[:, 1] - X[:, 2]
    X = np.column_stack([X, np.zeros(3000)])
    beta = np.array([0.25, -0.6, -0.8, 0.0, 0.0])
    y = X @ beta + 0.01 * rng.normal(size=3000)
    w = rng.integers(0, 3, 3000).astype(float)
    b = tc.wls(X, y, w)
    ok_r = np.allclose(X @ b, X @ beta, atol=0.01) and abs(b[4]) < 1e-12 and abs(b[0] - 0.25) < 0.01
    X2 = rng.normal(size=(3000, 4))
    b2 = tc.wls(X2, X2 @ np.array([0.25, -0.6, -0.8, 0.3]) + 0.01 * rng.normal(size=3000), w)
    ok_r &= np.allclose(b2, [0.25, -0.6, -0.8, 0.3], atol=0.01)
    print("regression", np.round(b2, 3), ok_r)
    ok &= ok_r

    # 4. centering and folds
    pid = rng.integers(0, 50, 1000)
    x = rng.normal(size=1000) + pid
    xc = tc.center(x, pid, 50)
    ok_f = np.allclose(np.bincount(pid, weights=xc, minlength=50), 0, atol=1e-9)
    for P in (7, 23, 100):
        f = tc.folds(P)
        perm = np.random.default_rng(20261014).permutation(np.arange(P))
        ok_f &= all(f[perm[j]] == j % 10 for j in range(P))
        cnt = np.bincount(f, minlength=10)
        ok_f &= cnt.max() - cnt[cnt > 0].min() <= 1
    print("centering and folds", ok_f)
    ok &= ok_f

    # 5. end to end on simulated times: four feature law recovered, transferred law fitted on the others
    sets = {}
    for name in tc.SETS:
        P, Tp = 40, 30
        pidx = np.repeat(np.arange(P), Tp)
        n = rng.choice([4, 9, 16], size=P * Tp)
        V = np.full((P * Tp, 16), np.nan)
        for t in range(P * Tp):
            V[t, : n[t]] = rng.uniform(0, 1, n[t])
        d = {"name": name, "V": V, "n": n, "choice": np.zeros(P * Tp, int), "pid": pidx, "P": P}
        Xr = tc.raw_features(d)
        logt = rng.normal(0, 0.5, P)[pidx] + Xr @ np.array([0.25, -0.6, -0.8, 0.3]) + 0.05 * rng.normal(size=P * Tp)
        d["seconds"] = np.exp(logt)
        d["choice"] = np.array([rng.choice(n[t], p=tc.softmax_rows(3.0, V[t:t + 1])[0, : n[t]]) for t in range(P * Tp)])
        sets[name] = tc.prepare(d)
    ones = {k: np.ones(len(sets[k]["y"])) for k in tc.SETS}
    tr = tc.transfer(sets, ones)
    ok_e = all(np.allclose(tr[k]["coef"], [0.25, -0.6, -0.8, 0.3], atol=0.05) for k in tc.SETS)
    ws = tc.within_set(sets["snack"], ones["snack"])
    ok_e &= ws["S2"] > ws["S1"] and ws["S2"] > 0.9 and abs(ws["S3"] - ws["S2"]) < 0.01
    print("end to end", {k: round(v, 3) for k, v in ws.items()}, ok_e)
    ok &= ok_e

    print("ALL CHECKS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
