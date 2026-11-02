"""Does the return rule test separate the processes it should? Simulated choices on the real snack displays.

Usage: python -I scripts/simulate_return_rule.py data/snack

Only the displays (liking ratings) are read. Gaze sequences and choices are simulated, never read.
Four worlds generate a sequence of looks and a choice on every display.
  SR       attention drawn by value, the first option looked at twice is chosen.
  SR_part  as SR, but people keep looking for three times as long and choose the first returned option on
           60 percent of trials, otherwise by CL. A partial return rule.
  CL_scan  looks drawn by value for as long as in SR_part, choice by CL independently of the looks.
  Dwell    looks as in CL_scan, choice by value plus how often the option was looked at.
Each world is scored by the held out gain, in bits per choice, from adding to CL one indicator built from the
looks, the first returned option, the most looked at option or the last looked at option, each set size
predicted from fits to the others.
"""

import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from overload.snack import load_snack

LAM = 1.0
rng = np.random.default_rng(2026110101)


def sr_sequence(q):
    seen, seq = set(), []
    while True:
        i = int(rng.choice(q.size, p=q))
        seq.append(i)
        if i in seen:
            return seq, i
        seen.add(i)


def extend(seq, q, factor=3):
    target = max(len(seq) * factor, len(seq))
    while len(seq) < target:
        seq.append(int(rng.choice(q.size, p=q)))
    return seq


def features(seq, n):
    seen, first_ret = set(), None
    for i in seq:
        if i in seen and first_ret is None:
            first_ret = i
        seen.add(i)
    r = np.zeros(n)
    if first_ret is not None:
        r[first_ret] = 1.0
    d = np.bincount(seq, minlength=n).astype(float)
    m = np.zeros(n)
    m[np.argmax(d)] = 1.0
    last = np.zeros(n)
    last[seq[-1]] = 1.0
    return {"return": r, "most": m, "last": last, "dwell": np.log1p(d)}


def simulate(world, V):
    out = []
    for v in V:
        q = softmax(LAM * v)
        seq, ret = sr_sequence(q)
        if world == "SR":
            c = ret
        elif world == "SR_part":
            seq = extend(seq, q)
            c = ret if rng.random() < 0.6 else int(rng.choice(v.size, p=q))
        elif world == "CL_scan":
            seq = extend(seq, q)
            c = int(rng.choice(v.size, p=q))
        else:
            seq = extend(seq, q)
            d = np.bincount(seq, minlength=v.size)
            c = int(rng.choice(v.size, p=softmax(LAM * v + 1.0 * np.log1p(d))))
        out.append((features(seq, v.size), c))
    return out


def nll(theta, rows, key):
    lam, beta = theta
    f = 0.0
    for v, F, c in rows:
        u = lam * v + (beta * F[key] if key else 0.0)
        f -= u[c] - logsumexp(u)
    return f


def fit(rows, key):
    x0 = np.array([LAM, 0.0])
    if key is None:
        res = minimize(lambda t: nll([t[0], 0.0], rows, None), [LAM], method="L-BFGS-B", bounds=[(0, 50)])
        return np.array([res.x[0], 0.0])
    res = minimize(nll, x0, args=(rows, key), method="L-BFGS-B", bounds=[(0, 50), (-50, 50)])
    return res.x


def main(folder):
    data = load_snack(folder)
    blocks = [(b.n, b.values) for b in data.blocks]
    for world in ("SR", "SR_part", "CL_scan", "Dwell"):
        rows_by = {n: [(v, F, c) for v, (F, c) in zip(V, simulate(world, V))] for n, V in blocks}
        res = {}
        for key in (None, "return", "most", "last", "dwell"):
            bits, N = 0.0, 0
            for n in rows_by:
                train = [r for m, rs in rows_by.items() if m != n for r in rs]
                th = fit(train, key)
                bits -= nll(th, rows_by[n], key) / np.log(2)
                N += len(rows_by[n])
            res[key] = bits / N
        base = res[None]
        gains = ", ".join(f"{k} {res[k] - base:+.3f}" for k in ("return", "most", "last", "dwell"))
        print(f"{world:8s} gain over CL in bits per choice: {gains}")


if __name__ == "__main__":
    main(sys.argv[1])
