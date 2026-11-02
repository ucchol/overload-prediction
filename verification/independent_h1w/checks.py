"""Checks of the independent H1w implementation on known cases and simulated data only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

fails = 0


def check(name, ok):
    global fails
    print(("ok   " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


ps = core.POWERSET
check("powerset has 32 subsets, empty first", len(ps) == 32 and ps[0] == [])
check("powerset singletons then pairs", ps[1:6] == [[1], [2], [3], [4], [5]] and ps[6] == [1, 2] and ps[15] == [4, 5])
check("powerset full menu last", ps[31] == [1, 2, 3, 4, 5] and ps[26] == [1, 2, 3, 4])
check("expected values", np.allclose(core.EV, [12, 25, 20, 22.5, 24.2, 21.7]))
check("menu 32 has six options", core.MENU_N[32] == 6 and list(core.MENU_OPTS[7]) == [0, 1, 2])

rng = np.random.default_rng(1)
for n in range(2, 7):
    v = rng.uniform(10, 25, n)
    for lam in (0.0, 0.3, 2.0):
        P = core.sr_probs(v, lam)
        check(f"SR sums to one n={n} lam={lam}", abs(P.sum() - 1) < 1e-12)
    P0 = core.sr_probs(v, 0.0)
    check(f"SR uniform at lam=0 n={n}", np.allclose(P0, 1.0 / n))

# SR by simulation of the process: draw from q with replacement until an option repeats, choose it.
v = np.array([12, 25, 20, 22.5])
q = core.cl_probs(v, 0.2)
sim = np.zeros(4)
draws = rng.choice(4, size=(200000, 5), p=q)
for row in draws:
    seen = set()
    for x in row:
        if x in seen:
            sim[x] += 1
            break
        seen.add(x)
sim /= sim.sum()
check("SR formula matches simulated sample until return", np.max(np.abs(sim - core.sr_probs(v, 0.2))) < 0.005)

# WLS slope against an explicit weighted design solve.
y = rng.normal(size=5); w = rng.integers(10, 900, 5).astype(float)
X = np.column_stack([np.ones(5), core.LOG2N])
beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
check("WLS slope", abs(core.wls_slope(y, core.LOG2N, w) - beta[1]) < 1e-12)

# CL and SR recover a known sensitivity from large simulated samples over the real menus.
def sim_counts(probfun, values, lam, N, seed):
    r = np.random.default_rng(seed)
    return {m: r.multinomial(N, probfun(values[core.MENU_OPTS[m]], lam)).astype(float) for m in core.MENU_IDS}

c = sim_counts(core.cl_probs, core.EV, 0.25, 20000, 2)
lam = core.cl_fit(c, core.EV, core.MENU_IDS)
check(f"CL recovers 0.25 (got {lam:.4f})", abs(lam - 0.25) < 0.01)
srt = core.SRTable(core.EV)
c = sim_counts(core.sr_probs, core.EV, 0.25, 20000, 3)
lam = srt.fit(c, core.MENU_IDS)
check(f"SR recovers 0.25 (got {lam:.4f})", abs(lam - 0.25) < 0.01)
# SR fit is at least as good as every grid point and nearby points.
ll = lambda x: core._loglik(core.sr_probs, c, core.EV, core.MENU_IDS, x)  # noqa: E731
check("SR fit is a local maximum", ll(lam) >= max(ll(lam * 1.001), ll(lam * 0.999)))
check("SR fit beats grid", ll(lam) >= max(ll(g) for g in core.SR_GRID) - 1e-9)
# CL score of zero at fit
c = sim_counts(core.cl_probs, core.EV, 0.0, 50, 4)
check("CL lambda >= 0", core.cl_fit(c, core.EV, core.MENU_IDS) >= 0)

# CRRA recovers r on simulated choices.
r0, l0 = 0.6, 2.0
vv = core.crra_values(r0)
c = sim_counts(core.cl_probs, vv, l0, 20000, 5)
r, lam, _ = core.crra_fit(c)
check(f"CRRA recovers r=0.6 (got {r:.3f}, lambda {lam:.3f})", abs(r - r0) < 0.05)
check("CRRA r=1 is the log limit", np.allclose(core.crra_values(1.0), core.crra_values(1.0 + 1e-7) - core.crra_values(1.0 + 1e-7).mean() + core.crra_values(1.0).mean(), atol=1e-5))

# Held-out forecast: with choices from CL itself, people minus CL growth near zero.
counts = {cst: sim_counts(core.cl_probs, core.EV, 0.3, 3000, 10 + i) for i, cst in enumerate(core.COSTS)}
out = core.analyse(counts, 0.0, srt)
check(f"CL data, people minus CL growth small ({out['low']['people_minus_CL']:.4f})", abs(out["low"]["people_minus_CL"]) < 0.02)
check("CRRA r=0 equals EV ranking, growth matches", abs(out["low"]["growth_observed_crra"] - out["low"]["growth_observed"]) < 1e-12)

print("failures:", fails)
sys.exit(1 if fails else 0)
