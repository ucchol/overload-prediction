import numpy as np
import pytest

from overload.metrics import (
    attainable,
    completeness,
    effective_number,
    log_score,
    overlap,
    sampled_effective_number,
    shares,
)


def test_uniform_prediction_scores_zero():
    counts = np.array([5.0, 0.0, 3.0, 2.0])
    assert log_score(counts, np.full(4, 0.25)) == pytest.approx(0.0, abs=1e-12)


def test_plugin_attainable_equals_log_score_of_shares():
    counts = np.array([7.0, 1.0, 0.0, 2.0, 10.0])
    assert attainable(counts, corrected=False) == pytest.approx(log_score(counts, shares(counts)), abs=1e-12)


def test_miller_correction_lowers_the_benchmark():
    counts = np.array([7.0, 1.0, 0.0, 2.0, 10.0])
    m, N = 4, counts.sum()
    gap = attainable(counts, corrected=False) - attainable(counts)
    assert gap == pytest.approx((m - 1) / (2 * N * np.log(2)), rel=1e-12)


@pytest.mark.parametrize("q", [np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05]), np.full(8, 1 / 8)])
def test_true_probabilities_reach_about_one_hundred_percent(q):
    """The corrected benchmark makes the true model score about 100 percent at realistic sample sizes."""
    rng = np.random.default_rng(1)
    samples = rng.multinomial(90, q, size=4000).astype(float)
    num = np.mean([log_score(c, q) for c in samples])
    den = np.mean([attainable(c) for c in samples])
    if den > 0.02:
        assert 100 * num / den == pytest.approx(100.0, abs=3.0)
    plug = np.mean([attainable(c, corrected=False) for c in samples])
    assert plug > num  # the plug-in benchmark is biased upward


def test_completeness_is_a_ratio_of_sums():
    assert completeness([0.2, 0.4], [0.5, 1.0]) == pytest.approx(40.0)


def test_overlap_bounds():
    p = np.array([0.5, 0.5, 0.0])
    assert overlap(p, p) == pytest.approx(1.0)
    assert overlap(p, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0)


def test_effective_number():
    assert effective_number(np.full(6, 1 / 6)) == pytest.approx(6.0)
    assert effective_number(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_sampled_effective_number_shows_finite_sample_concentration():
    q = np.full(16, 1 / 16)
    small = sampled_effective_number(q, 40, seed=3)
    large = sampled_effective_number(q, 40000, seed=3)
    assert small < large <= 16.0
    assert large == pytest.approx(16.0, rel=1e-3)
    assert sampled_effective_number(q, 40, seed=3) == small  # deterministic given the seed
