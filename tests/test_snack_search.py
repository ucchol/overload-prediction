"""The search model on rating levels against the item level implementation, and parameter recovery. No real data."""

from __future__ import annotations

import numpy as np
import pytest

from overload import snack_search as ss
from overload.search import race_probabilities, sample_until_return, sample_until_return_length


def _displays(rng: np.random.Generator, trials: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    values = rng.integers(-3, 4, size=(trials, n)).astype(float)
    choice = rng.integers(0, n, size=trials)
    return values, choice


@pytest.mark.parametrize("n", [9, 16, 25, 36])
def test_levels_match_the_item_level_model(n):
    rng = np.random.default_rng(n)
    values, choice = _displays(rng, 40, n)
    trials = ss.LevelTrials.from_values(values, choice)
    for lam in (0.0, 0.4, 1.3, 3.0):
        P = ss.item_probabilities(lam, trials, values)
        looks = ss.expected_looks(lam, trials)
        for t in range(values.shape[0]):
            assert np.allclose(P[t], sample_until_return(lam, values[t]), rtol=1e-10, atol=1e-14)
            assert looks[t] == pytest.approx(sample_until_return_length(lam, values[t]), rel=1e-10)


def test_equal_ratings_give_the_birthday_problem():
    values = np.zeros((1, 36))
    trials = ss.LevelTrials.from_values(values, np.array([0]))
    expected = sum(np.prod([(36 - i) / 36 for i in range(k)]) for k in range(37))
    assert ss.expected_looks(2.0, trials)[0] == pytest.approx(expected, rel=1e-12)
    assert np.allclose(ss.item_probabilities(2.0, trials, values), 1 / 36)


def test_chosen_log_probability_matches_item_probabilities():
    rng = np.random.default_rng(7)
    values, choice = _displays(rng, 30, 16)
    trials = ss.LevelTrials.from_values(values, choice)
    P = ss.item_probabilities(0.9, trials, values)
    assert np.allclose(ss.chosen_log_probability(0.9, trials), np.log(P[np.arange(30), choice]))


def test_fit_recovers_the_sensitivity_of_simulated_choices():
    rng = np.random.default_rng(11)
    lam_true = 0.8
    parts = []
    for n in (9, 16, 25, 36):
        values, _ = _displays(rng, 1500, n)
        choice = np.array(
            [rng.choice(n, p=race_probabilities(np.exp(lam_true * v) / np.exp(lam_true * v).sum())) for v in values]
        )
        parts.append((ss.LevelTrials.from_values(values, choice), np.ones(values.shape[0])))
    lam = ss.fit_search(parts)
    assert lam == pytest.approx(lam_true, abs=0.08)
    # the fit is a maximum of the weighted likelihood
    nll = lambda g: -sum(w @ ss.chosen_log_probability(g, t) for t, w in parts)  # noqa: E731
    assert nll(lam) <= min(nll(lam * 0.98), nll(lam * 1.02))


def test_weights_act_as_repeated_trials():
    rng = np.random.default_rng(3)
    values, choice = _displays(rng, 200, 9)
    choice = values.argmax(axis=1)  # choices follow the ratings, so the fit is interior only with some noise
    choice[::3] = rng.integers(0, 9, size=choice[::3].size)
    trials = ss.LevelTrials.from_values(values, choice)
    w = rng.integers(0, 3, size=200).astype(float)
    rows = np.repeat(np.arange(200), w.astype(int))
    a = ss.fit_search([(trials, w)])
    b = ss.fit_search([(trials.subset(rows), np.ones(rows.size))])
    assert a == pytest.approx(b, rel=1e-6)
