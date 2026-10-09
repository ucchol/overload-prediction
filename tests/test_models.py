import numpy as np
import pytest

from overload.data import Task, Trial
from overload.fit import evaluate
from overload.models import (
    Model,
    _mlba_sum,
    eba_probabilities,
    scaled,
    score_cl,
    score_dbs_avg,
    score_dbs_sum,
    score_mlba_avg,
    score_mlba_sum,
    set_size_factor,
)


def _trial(X, key="t", info="low"):
    X = np.asarray(X, float)
    return Trial(key, info, X.shape[0], tuple(f"o{i}" for i in range(X.shape[0])), X, np.zeros(X.shape[0]), "c")


def _task(seed=0, sizes=(8, 16, 32, 8, 16, 32), K=3):
    rng = np.random.default_rng(seed)
    trials = tuple(_trial(rng.integers(1, 5, size=(n, K)), key=f"t{j}") for j, n in enumerate(sizes))
    r = np.array([0.5, 0.3, 0.2])
    panel = []
    for _ in range(60):
        panel.append([(j, int(rng.integers(0, t.n))) for j, t in enumerate(trials)])
    return Task("synthetic", ("a", "b", "c"), trials, r, 1.0, panel)


def _softmax(v):
    e = np.exp(v - v.max())
    return e / e.sum()


def test_averaged_comparisons_are_summed_ones_over_rivals():
    t = _trial(np.random.default_rng(2).integers(1, 5, size=(12, 3)))
    r = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(score_dbs_avg(t, r) * 11, score_dbs_sum(t, r))
    np.testing.assert_allclose(score_mlba_avg(t, r) * 11, score_mlba_sum(t, r))


def test_mlba_without_distance_weighting_is_a_sharper_conditional_logit():
    t = _trial(np.random.default_rng(4).integers(1, 5, size=(9, 3)))
    r = np.array([0.5, 0.3, 0.2])
    v = _mlba_sum(t, r, kappa=0.0)
    np.testing.assert_allclose(_softmax(0.7 * v), _softmax(0.7 * t.n * score_cl(t, r)), atol=1e-12)


def test_set_size_factor():
    assert set_size_factor(8, 1.0) == pytest.approx(1.0)
    assert set_size_factor(32, 1.0) == pytest.approx(31 / 7)
    assert set_size_factor(32, -0.5) == pytest.approx((31 / 7) ** -0.5)


def test_beta_one_with_averaging_reproduces_summing():
    task = _task()
    counts = task.counts()
    summed = evaluate(Model("s", "softmax", score_dbs_sum), task, counts, alone=False)
    beta1 = evaluate(Model("a", "softmax", scaled(score_dbs_avg, 1.0)), task, counts, alone=False)
    for a, b in zip(summed.held_out + summed.fitted, beta1.held_out + beta1.fitted, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_eba_two_options_follows_aspect_weights():
    X = np.array([[2.0, 1.0], [1.0, 2.0]])
    p = eba_probabilities(X, np.array([0.75, 0.25]))
    np.testing.assert_allclose(p, [0.75, 0.25])


def test_eba_dominating_option_is_certain():
    X = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(eba_probabilities(X, np.array([0.75, 0.25])), [0.0, 1.0, 0.0])


def test_eba_is_uniform_without_splitting_aspects():
    X = np.array([[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]])
    np.testing.assert_allclose(eba_probabilities(X, np.array([0.6, 0.4])), np.full(3, 1 / 3))


def test_eba_ignores_attributes_without_weight():
    X = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(eba_probabilities(X, np.array([1.0, 0.0])), [1.0, 0.0])


@pytest.mark.parametrize("aspect", ["median", "best"])
def test_eba_returns_a_distribution(aspect):
    rng = np.random.default_rng(5)
    for n in (8, 16, 32):
        X = rng.integers(1, 6, size=(n, 5)).astype(float)
        p = eba_probabilities(X, np.array([0.4, 0.3, 0.2, 0.1, 0.0]), aspect)
        assert p.min() >= 0.0
        assert p.sum() == pytest.approx(1.0)


def test_eba_hand_computed_three_way_split():
    # Attribute weights 0.5 and 0.5. Aspect a: options 0 and 1. Aspect b: options 1 and 2.
    # Drawing a leaves {0, 1}, then b splits it and leaves {1}. Drawing b leaves {1, 2}, then a leaves {1}.
    X = np.array([[2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(eba_probabilities(X, np.array([0.5, 0.5])), [0.0, 1.0, 0.0])
    # Without a dominating option: drawing a (0.75) keeps {0, 1}, drawing b (0.25) keeps {2, 3},
    # and nothing splits the survivors further, so each pair splits its branch's probability evenly.
    X2 = np.array([[3.0, 1.0], [3.0, 1.0], [1.0, 3.0], [1.0, 3.0]])
    np.testing.assert_allclose(eba_probabilities(X2, np.array([0.75, 0.25])), [0.375, 0.375, 0.125, 0.125])
