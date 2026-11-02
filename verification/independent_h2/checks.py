"""Self checks on simulated data only. Exits nonzero on failure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


# 1. cue construction
r, d, l, fr = core.trial_cues([2, 2, 0, 1, 0, 2], [10, 10, 20, 40, 10, 10], 4)
check(fr == 0 and r.tolist() == [1, 0, 0, 0], "return marks the first returned item after merging repeats")
check(np.allclose(d, [30 / 100, 40 / 100, 30 / 100, 0]), "dwell is the share of gaze time")
check(l.tolist() == [0, 0, 1, 0], "last marks the last gazed item")
r, d, l, fr = core.trial_cues([1, 1, 1], [5, 5, 5], 3)
check(fr == -1 and r.sum() == 0, "consecutive gazes on one item are not a return")
r, d, l, fr = core.trial_cues([], [], 3)
check(r.sum() == 0 and d.sum() == 0 and l.sum() == 0 and fr == -1, "no gaze, every cue zero")
r, d, l, fr = core.trial_cues([0, 1, 2, 1, 0], [1, 1, 1, 1, 1], 3)
check(fr == 1, "first return is the earliest revisit, not the earliest item revisited")


# simulated groups
def sim(lam, beta, cue_kind, sizes=(4, 9, 16), T=1500, P=30, seed=0):
    rng = np.random.default_rng(seed)
    groups = []
    for n in sizes:
        V = rng.integers(-3, 4, size=(T, n)).astype(float)
        C = {k: np.zeros((T, n)) for k in core.CUES}
        fr = np.full(T, -1)
        for t in range(T):
            j = rng.integers(n)
            C["return"][t, j] = 1
            fr[t] = j
            x = rng.random(n)
            C["dwell"][t] = x / x.sum()
            C["last"][t, rng.integers(n)] = 1
        u = lam * V + beta * C[cue_kind]
        q = np.exp(u - u.max(1, keepdims=True))
        q /= q.sum(1, keepdims=True)
        ch = np.array([rng.choice(n, p=qq) for qq in q])
        groups.append(core.Group(n, V, C, ch, rng.integers(0, P, size=T), fr))
    return groups, P


# 2. gradient and Hessian check
g, P = sim(0.7, 1.2, "return", T=200)
data = [(core._feats(x, "dwell"), x.choice, np.random.default_rng(1).integers(0, 3, len(x.choice)).astype(float))
        for x in g]
th = np.array([0.4, -0.8])
f, gr, H = core.nll_grad_hess(th, data)
eps = 1e-6
num_g = np.array([(core.nll_grad_hess(th + eps * e, data)[0] - core.nll_grad_hess(th - eps * e, data)[0]) / (2 * eps)
                  for e in np.eye(2)])
num_H = np.array([(core.nll_grad_hess(th + eps * e, data)[1] - core.nll_grad_hess(th - eps * e, data)[1]) / (2 * eps)
                  for e in np.eye(2)])
check(np.allclose(gr, num_g, rtol=1e-6, atol=1e-5), f"gradient matches finite differences (max diff {np.max(np.abs(gr-num_g)):.2e})")
check(np.allclose(H, num_H, rtol=1e-5, atol=1e-4), "Hessian matches finite differences")

# 3. recover a known beta
for cue, lam, beta in (("return", 0.6, 1.5), ("dwell", 0.6, 3.0), ("last", 0.5, -1.0)):
    g, P = sim(lam, beta, cue, T=3000, seed=7)
    W = [np.ones(len(x.choice)) for x in g]
    th = core.fit(g, W, cue)
    check(abs(th[0] - lam) < 0.06 and abs(th[1] - beta) < 0.25 * max(1, abs(beta)),
          f"recover {cue}: lambda {th[0]:.3f} (true {lam}), beta {th[1]:.3f} (true {beta})")
    f0, g0, _ = core.nll_grad_hess(th, [(core._feats(x, cue), x.choice, w) for x, w in zip(g, W)])
    check(np.max(np.abs(g0)) < 1e-8, f"score function zero at the fit ({np.max(np.abs(g0)):.1e})")

# 4. weights equal duplicated trials
g, P = sim(0.6, 1.0, "return", T=300, seed=3)
counts = np.random.default_rng(5).integers(0, 3, size=P)
W = core.weights_from_counts(g, counts)
dup = []
for x, w in zip(g, W):
    idx = np.repeat(np.arange(len(x.choice)), w.astype(int))
    dup.append(core.Group(x.n, x.V[idx], {k: x.C[k][idx] for k in core.CUES}, x.choice[idx], x.part[idx],
                          x.first_ret[idx]))
a1 = core.fit(g, W, "return")
a2 = core.fit(dup, [np.ones(len(x.choice)) for x in dup], "return")
check(np.allclose(a1, a2, atol=1e-8), "weighted fit equals fit on duplicated trials")

# 5. gains: CL data give near zero return gain, return data a positive gain over dwell
g, P = sim(0.6, 0.0, "return", T=2000, seed=11)
o = core.analyse(g, np.ones(P, int))
check(abs(o["return_gain"]) < 0.01, f"CL alone: return gain {o['return_gain']:+.4f} near zero")
g, P = sim(0.6, 2.0, "return", T=2000, seed=12)
o = core.analyse(g, np.ones(P, int))
check(o["return_gain"] > 0.1 and o["return_minus_dwell"] > 0.1,
      f"return rule: gain {o['return_gain']:+.3f}, minus dwell {o['return_minus_dwell']:+.3f}")
check(abs(o["CL_bits_above_uniform"] - o["CL_log2_score"]
          - np.average([np.log2(x.n) for x in g], weights=[len(x.choice) for x in g])) < 1e-10,
      "bits above uniform = log2 score + mean log2 n")

# 6. intervals
s = core.summarize(1.0, [0.5, 1.5, 2.0, 0.9, 1.1])
lo, hi = np.percentile([0.5, 1.5, 2.0, 0.9, 1.1], [2.5, 97.5])
check(np.allclose(s["basic"], [2 - hi, 2 - lo]) and s["clear"], "basic interval and clear rule")
s = core.summarize(0.1, [-0.5, 0.2, 0.3, 0.4])
check(not s["clear"], "interval covering zero is not clear")

# 7. bootstrap draws reproducible
rng = np.random.default_rng(110000)
x = rng.integers(0, 49, size=49)
y = np.random.default_rng(110000).integers(0, 49, size=49)
check((x == y).all(), "replicate draws reproducible from the seed")
print("all checks passed")
