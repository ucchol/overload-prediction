import numpy as np
import pytest

from overload.engine import concentration_seed
from overload.experiments.h1_narrowing import _centered, _slope


def test_concentration_seeds_are_distinct_and_fixed():
    seeds = {concentration_seed(t, j, b, r) for t in (0, 1) for j in range(6) for b in range(-1, 1000) for r in (1, 2)}
    assert len(seeds) == 2 * 6 * 1001 * 2
    assert concentration_seed(1, 5, 999, 2) == concentration_seed(1, 5, 999, 2)


def test_centered_intervals_sit_on_the_estimate():
    errors = np.random.default_rng(0).normal(0.0, 1.0, size=4000)
    out = _centered(3.0, errors)
    assert np.mean(out["percentile"]) == pytest.approx(3.0, abs=0.1)
    assert np.mean(out["basic"]) == pytest.approx(3.0, abs=0.1)
    assert out["clear"]
    assert out["p"] < 0.01


def test_centered_intervals_include_zero_for_a_null_estimate():
    errors = np.random.default_rng(1).normal(0.0, 1.0, size=4000)
    out = _centered(0.2, errors)
    assert not out["clear"]
    assert out["p"] > 0.5


def test_slope():
    x = np.log2([8, 8, 16, 16, 32, 30])
    assert _slope(x, 2.0 * x + 1.0) == pytest.approx(2.0)
