import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from overload.fit import SeparationError, fit_lapse, fit_softmax, softmax_nll


def _random_problem(seed, sizes=(8, 16, 32)):
    rng = np.random.default_rng(seed)
    scores = [rng.normal(size=n) for n in sizes]
    counts = []
    for v in scores:
        p = np.exp(0.8 * v)
        counts.append(rng.multinomial(70, p / p.sum()).astype(float))
    return scores, counts


@pytest.mark.parametrize("seed", range(5))
def test_softmax_estimate_matches_brute_force(seed):
    scores, counts = _random_problem(seed)
    lam = fit_softmax(scores, counts)
    ref = minimize_scalar(
        lambda a: softmax_nll(a, scores, counts), bounds=(0, 20), method="bounded", options={"xatol": 1e-12}
    )
    assert lam == pytest.approx(ref.x, rel=1e-6)
    assert softmax_nll(lam, scores, counts) <= ref.fun + 1e-9


def test_softmax_estimate_is_zero_when_choices_avoid_high_scores():
    scores = [np.array([0.0, 1.0, 2.0])]
    counts = [np.array([10.0, 2.0, 0.0])]
    assert fit_softmax(scores, counts) == 0.0


def test_softmax_constant_scores_give_zero():
    assert fit_softmax([np.ones(4)], [np.array([1.0, 2.0, 3.0, 4.0])]) == 0.0


def test_perfect_ordering_raises():
    with pytest.raises(SeparationError):
        fit_softmax([np.array([0.0, 1.0])], [np.array([0.0, 5.0])])


@pytest.mark.parametrize("seed", range(5))
def test_lapse_estimate_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    bases = [rng.dirichlet(np.ones(n) * 0.3) for n in (8, 16)]
    counts = [rng.multinomial(60, 0.6 * b + 0.4 / b.size).astype(float) for b in bases]

    def nll(e):
        return -sum(float(c @ np.log((1 - e) * b + e / b.size)) for b, c in zip(bases, counts, strict=True))

    eps = fit_lapse(bases, counts)
    ref = minimize_scalar(nll, bounds=(1e-12, 1.0), method="bounded", options={"xatol": 1e-12})
    assert eps == pytest.approx(ref.x, abs=1e-6)


def test_lapse_is_positive_when_a_chosen_option_has_zero_base_probability():
    bases = [np.array([1.0, 0.0, 0.0])]
    counts = [np.array([8.0, 1.0, 1.0])]
    eps = fit_lapse(bases, counts)
    assert 0.0 < eps < 1.0
    # closed form for this case: maximize 8 log(1 - 2e/3) + 2 log(e/3)
    assert eps == pytest.approx(0.3, rel=1e-9)
