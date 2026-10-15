"""Two look satisficing, the search model with an absolute standard, on cases with known answers. No data needed."""

from __future__ import annotations

import numpy as np
import pytest

from overload.search import (
    race_length,
    race_probabilities,
    race_with_registration,
    race_with_registration_exact,
    sample_until_return,
    two_look_satisficing,
)


def _simulate(q: np.ndarray, h: np.ndarray, sims: int, seed: int) -> tuple[np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    wins, looks, distinct = np.zeros(q.size), 0, 0
    for _ in range(sims):
        hits, seen, k = np.zeros(q.size, int), set(), 0
        while True:
            i = int(rng.choice(q.size, p=q))
            k += 1
            seen.add(i)
            if rng.random() < h[i]:
                hits[i] += 1
                if hits[i] == 2:
                    wins[i] += 1
                    break
        looks += k
        distinct += len(seen)
    return wins / sims, looks / sims, distinct / sims


def test_with_every_look_registering_it_is_sample_until_return():
    q = np.random.default_rng(1).dirichlet(np.ones(12))
    p, looks, distinct = race_with_registration(q, np.ones(12))
    assert np.allclose(p, race_probabilities(q), rtol=1e-12)
    assert looks == pytest.approx(race_length(q), rel=1e-12)
    assert distinct == pytest.approx(race_length(q) - 1.0, rel=1e-12)


def test_quadrature_matches_the_exact_sums():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(2, 30))
        q = rng.dirichlet(np.ones(n) * rng.uniform(0.2, 3.0))
        h = rng.uniform(0.02, 1.0, size=n)
        a, b = race_with_registration(q, h), race_with_registration_exact(q, h)
        assert np.allclose(a[0], b[0], rtol=1e-11, atol=1e-15)
        assert a[1] == pytest.approx(b[1], rel=1e-11)
        assert a[2] == pytest.approx(b[2], rel=1e-11)


def test_closed_forms_match_simulation_of_the_process():
    rng = np.random.default_rng(3)
    q = rng.dirichlet(np.ones(6))
    h = rng.uniform(0.1, 1.0, size=6)
    p_sim, looks_sim, distinct_sim = _simulate(q, h, 30000, seed=4)
    p, looks, distinct = race_with_registration(q, h)
    assert np.abs(p_sim - p).max() < 0.012
    assert looks_sim == pytest.approx(looks, rel=0.02)
    assert distinct_sim == pytest.approx(distinct, rel=0.02)


def test_choices_equal_sample_until_return_at_the_same_sensitivity():
    v = np.array([-2.0, 0.0, 1.0, 1.0, 3.0, 2.0, -1.0])
    for lam in (0.0, 0.5, 1.2, 3.0):
        p, _, _ = two_look_satisficing(lam, v, v_top=3.0)
        assert np.allclose(p, sample_until_return(lam, v), rtol=1e-12)


def test_better_displays_are_decided_with_fewer_looks():
    # adding the same amount to every value leaves choices alone and shortens the search
    v = np.array([-2.0, -1.0, 0.0, 0.0, 1.0, -3.0, 0.0, -1.0, -2.0])
    p0, looks0, distinct0 = two_look_satisficing(1.2, v, v_top=3.0)
    p1, looks1, distinct1 = two_look_satisficing(1.2, v + 2.0, v_top=3.0)
    assert np.allclose(p0, p1, rtol=1e-12)
    assert looks1 < looks0 and distinct1 < distinct0


def test_values_above_the_top_of_the_scale_are_refused():
    with pytest.raises(ValueError):
        two_look_satisficing(1.0, np.array([1.0, 4.0]), v_top=3.0)
