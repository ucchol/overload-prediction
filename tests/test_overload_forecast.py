"""Forecasting choice overload (Amendment 8) on simulated data. No real data are used."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from overload import overload_forecast as of
from overload.experiments import overload_premonition as op
from overload.search import race_probabilities


def _panel(rng, lam, comp, people=30, per=30, sizes=(4, 9, 16)):
    T = people * per
    n = rng.choice(sizes, size=T)
    v = np.full((T, max(sizes)), np.nan)
    for t in range(T):
        v[t, : n[t]] = rng.integers(-5, 6, n[t])
    f = of.compensation(n) if comp else np.ones(T)
    z = lam * f[:, None] * np.where(np.isfinite(v), v, -np.inf)
    q = np.exp(z - z.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    choice = np.array([rng.choice(max(sizes), p=q[t]) for t in range(T)])
    part = np.repeat(np.arange(people), per)
    return of.Panel("reutskaja", v, n, choice, part, shift=6.0)


def test_weighted_auc_equals_a_pairwise_count():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 4, 40).astype(float)
    y = rng.random(40) < 0.4
    w = rng.integers(0, 3, 40).astype(float)
    num = den = 0.0
    for i, j in itertools.product(range(40), range(40)):
        if y[i] and not y[j] and w[i] > 0 and w[j] > 0:
            den += w[i] * w[j]
            num += w[i] * w[j] * (1.0 if s[i] > s[j] else 0.5 if s[i] == s[j] else 0.0)
    assert of.weighted_auc(s, y, w) == pytest.approx(num / den, rel=1e-12)


def test_sample_until_return_miss_probability_matches_the_item_race():
    rng = np.random.default_rng(2)
    p = _panel(rng, 0.8, False, people=4, per=10)
    fc, lam = of.miss_forecasts(p, np.ones(p.n.size))
    order = list(p.sizes)
    for t in range(p.n.size):
        v = p.values[t, : p.n[t]]
        q = np.exp(lam["SR"][order.index(p.n[t])] * (v - v.max()))
        pr = race_probabilities(q / q.sum())
        assert fc["SR"][t] == pytest.approx(1.0 - pr[v == v.max()].sum(), abs=1e-9)


@pytest.mark.parametrize(("comp", "wrong"), [(False, "compensation"), (True, "overload")])
def test_o1_never_reverses_the_simulated_process(comp, wrong):
    rng = np.random.default_rng(3)
    p = _panel(rng, 0.9, comp)
    op._STATE.update(groups={"reutskaja": ([p], 30)}, datasets={"reutskaja": ["reutskaja"]})
    point = op.run_replicate(-1)
    reps = [op.run_replicate(b) for b in range(20)]
    s = op.summarize_all(point, reps)["datasets"]["reutskaja"]
    assert s["O1"]["verdict"] != wrong
    assert s["O2"]["supported"]
