"""Mixed logit with a random sensitivity, on simulated choices."""

from __future__ import annotations

import numpy as np

from overload import mixed_logit as mx


def _simulate(rng, mu, sigma, people=150, per=25, n=6):
    lam = np.exp(mu + sigma * rng.normal(size=people))
    S = rng.normal(size=(people * per, n))
    person = np.repeat(np.arange(people), per)
    z = lam[person][:, None] * S
    q = np.exp(z - z.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    choice = np.array([rng.choice(n, p=r) for r in q])
    return S, choice, person


def test_recovers_mean_and_spread_of_the_sensitivity():
    rng = np.random.default_rng(1)
    S, c, p = _simulate(rng, np.log(1.5), 0.6)
    mu, sigma = mx.fit_mxl(S, c, p, np.ones(c.size))
    assert abs(mu - np.log(1.5)) < 0.15 and abs(sigma - 0.6) < 0.2


def test_no_spread_reduces_to_cl():
    rng = np.random.default_rng(2)
    S, c, p = _simulate(rng, np.log(1.0), 0.0)
    mu, sigma = mx.fit_mxl(S, c, p, np.ones(c.size))
    assert sigma < 0.25 and abs(mu) < 0.1
    q = mx.predict_mxl(mu, sigma, S)
    assert np.allclose(q.sum(1), 1.0)


def test_padded_options_get_no_probability():
    S = np.array([[1.0, 2.0, np.nan], [0.0, 1.0, 3.0]])
    q = mx.predict_mxl(0.0, 0.5, S)
    assert q[0, 2] == 0.0 and np.allclose(q.sum(1), 1.0)


def test_many_choices_per_person_recover_a_narrow_spread():
    rng = np.random.default_rng(4)
    S, c, p = _simulate(rng, np.log(2.0), 0.3, people=60, per=180, n=8)
    mu, sigma = mx.fit_mxl(S, c, p, np.ones(c.size), start=2.0)
    assert abs(mu - np.log(2.0)) < 0.12 and abs(sigma - 0.3) < 0.1


def test_gradient_matches_differences():
    rng = np.random.default_rng(5)
    S, c, p = _simulate(rng, 0.0, 0.5, people=30, per=20)
    captured = {}
    orig = mx.minimize

    def spy(fun, x0, **kw):
        captured["f"] = fun
        return orig(fun, x0, **kw)

    mx.minimize = spy
    try:
        mx.fit_mxl(S, c, p, np.ones(c.size))
    finally:
        mx.minimize = orig
    f = captured["f"]
    for th in (np.array([0.2, np.log(0.4)]), np.array([-0.5, np.log(1.0)])):
        _, g = f(th)
        num = [(f(th + 1e-6 * e)[0] - f(th - 1e-6 * e)[0]) / 2e-6 for e in np.eye(2)]
        assert np.allclose(g, num, rtol=1e-5, atol=1e-4)
