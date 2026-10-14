"""The search models and the effort counts, on cases with known answers. No participant data are needed."""

from __future__ import annotations

import numpy as np
import pytest

from overload.effort import accept_inspections, eba_checks
from overload.search import (
    accept_in_reading_order,
    race_length,
    race_length_exact,
    race_probabilities,
    race_probabilities_exact,
    sample_until_return,
)


def _simulate_race(q: np.ndarray, sims: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    wins, looks = np.zeros(q.size), 0
    draws = rng.choice(q.size, size=(sims, q.size + 1), p=q)
    for row in draws:
        seen = set()
        for t, x in enumerate(row):
            if x in seen:
                wins[x] += 1
                looks += t + 1
                break
            seen.add(x)
    return wins / sims, looks / sims


def test_two_options_closed_form():
    # option 1 wins with looks 11, 121 or 211, so P = q1^2 (1 + 2 q2), and E[looks] = 2 + 2 q1 q2
    q = np.array([0.6, 0.4])
    assert race_probabilities(q)[0] == pytest.approx(0.36 * 1.8, rel=1e-12)
    assert race_length(q) == pytest.approx(2 + 2 * 0.24, rel=1e-12)


@pytest.mark.parametrize("n", [2, 8, 32, 365])
def test_equal_options_give_the_birthday_problem(n):
    expected = sum(np.prod([(n - i) / n for i in range(k)]) for k in range(n + 1))
    assert race_length(np.ones(n) / n) == pytest.approx(expected, rel=1e-12)
    assert np.allclose(race_probabilities(np.ones(n) / n), 1.0 / n)


def test_quadrature_matches_the_exact_sums():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 40))
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.05, 3.0))
        assert np.allclose(race_probabilities(q), race_probabilities_exact(q), rtol=1e-12, atol=1e-15)
        assert race_length(q) == pytest.approx(race_length_exact(q), rel=1e-12)


def test_race_matches_simulation():
    q = np.random.default_rng(3).dirichlet(np.ones(9))
    p_sim, looks_sim = _simulate_race(q, 40000, seed=4)
    assert np.abs(p_sim - race_probabilities(q)).max() < 0.01
    assert looks_sim == pytest.approx(race_length(q), abs=0.03)


def test_sample_until_return_limits():
    v = np.array([1.0, 2.0, 2.0, 3.0])
    assert np.allclose(sample_until_return(0.0, v), 0.25)  # no sensitivity, every option alike
    p = sample_until_return(1.3, v)
    assert p[1] == pytest.approx(p[2])  # equal values, equal chances
    assert np.all(np.diff(p[[0, 1, 3]]) > 0)  # more value, more choices
    assert sample_until_return(60.0, v)[3] > 0.999  # very sensitive attention finds the best


def test_accept_in_reading_order_matches_simulation():
    v = np.array([0.2, 1.0, 0.7, 1.0, 0.1])
    lam = 2.0
    a = 1.0 / (1.0 + np.exp(-lam * (v - v.max())))
    rng = np.random.default_rng(5)
    counts, inspections = np.zeros(v.size), 0
    sims = 40000
    for _ in range(sims):
        k = 0
        while True:
            i = k % v.size
            inspections += 1
            if rng.random() < a[i]:
                counts[i] += 1
                break
            k += 1
    assert np.abs(counts / sims - accept_in_reading_order(lam, v)).max() < 0.01
    assert inspections / sims == pytest.approx(accept_inspections(lam, v), abs=0.03)


def test_eba_checks_by_hand():
    # one aspect splits {1, 2} from {0} (3 checks), then nothing splits the two left (2 checks)
    X = np.array([[0.0], [1.0], [1.0]])
    assert eba_checks(X, np.array([1.0])) == pytest.approx(5.0)
    # one option left after a single split costs only the split
    X = np.array([[0.0], [1.0]])
    assert eba_checks(X, np.array([1.0]), aspect="best") == pytest.approx(2.0)
