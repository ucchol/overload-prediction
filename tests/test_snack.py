"""Unit tests of the snack data module (ANALYSIS_PLAN.md, Amendment 3), on synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from overload import snack
from overload.data import Trial
from overload.fit import SeparationError, fit_softmax
from overload.models import score_dbs_avg, score_mlba_avg, set_size_factor


def _random_part(rng, T=60, n=7, levels=7, mask_share=None):
    values = rng.integers(-3, 4, size=(T, n)).astype(float)
    S = values * 0.7
    lam_true = 0.9
    mask = None
    if mask_share is not None:
        mask = rng.random((T, n)) > mask_share
        mask[np.arange(T), rng.integers(0, n, T)] = True
    q = snack.probabilities(lam_true, S, mask)
    choice = np.array([rng.choice(n, p=row) for row in q])
    return S, choice, mask


def test_fit_matches_package_softmax_fit():
    rng = np.random.default_rng(1)
    S, choice, _ = _random_part(rng)
    w = np.ones(choice.size)
    lam = snack.fit_sensitivity([snack.make_part(S, choice, None, w)])
    counts = [np.bincount([c], minlength=S.shape[1]).astype(float) for c in choice]
    assert lam == pytest.approx(fit_softmax(list(S), counts), rel=1e-10)


def test_integer_weights_equal_repeated_trials():
    rng = np.random.default_rng(2)
    S, choice, _ = _random_part(rng, T=40)
    w = rng.integers(0, 4, size=choice.size).astype(float)
    lam_w = snack.fit_sensitivity([snack.make_part(S, choice, None, w)])
    rep = np.repeat(np.arange(choice.size), w.astype(int))
    lam_r = snack.fit_sensitivity([snack.make_part(S[rep], choice[rep], None, np.ones(rep.size))])
    assert lam_w == pytest.approx(lam_r, rel=1e-10)


def test_masked_fit_matches_fit_on_available_items():
    rng = np.random.default_rng(3)
    S, choice, mask = _random_part(rng, T=50, n=9, mask_share=0.5)
    lam = snack.fit_sensitivity([snack.make_part(S, choice, mask, np.ones(choice.size))])
    scores, counts = [], []
    for t in range(choice.size):
        avail = np.flatnonzero(mask[t])
        scores.append(S[t, avail])
        counts.append((avail == choice[t]).astype(float))
    assert lam == pytest.approx(fit_softmax(scores, counts), rel=1e-10)


def test_probabilities_are_zero_outside_the_mask():
    S = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    mask = np.array([[True, False, True], [False, True, True]])
    q = snack.probabilities(1.3, S, mask)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.all(q[~mask] == 0.0)


def test_comparison_scores_equal_the_package_scores_with_one_attribute():
    rng = np.random.default_rng(4)
    values = rng.integers(-3, 4, size=(5, 9)).astype(float)
    r = np.array([1.0])
    for t in range(values.shape[0]):
        trial = Trial("t", "low", 9, tuple("abcdefghi"), values[t][:, None], np.zeros(9), "c")
        assert np.allclose(snack.scores(values[t : t + 1], "DbS_avg")[0], score_dbs_avg(trial, r))
        assert np.allclose(snack.scores(values[t : t + 1], "MLBA_avg")[0], score_mlba_avg(trial, r))


def test_separation_is_detected():
    S = np.array([[0.0, 1.0], [2.0, 1.0]])
    choice = np.array([1, 0])
    with pytest.raises(SeparationError):
        snack.fit_sensitivity([snack.make_part(S, choice, None, np.ones(2))])


def _synthetic_data(rng, beta_true, lam0=0.8, trials=400, seen_share=None):
    blocks = []
    for n in snack.SET_SIZES:
        values = rng.integers(-3, 4, size=(trials, n)).astype(float)
        lam = lam0 * set_size_factor(n, beta_true)
        seen = np.ones((trials, n), dtype=bool) if seen_share is None else rng.random((trials, n)) < seen_share
        q = snack.probabilities(lam, values, None if seen_share is None else seen)
        choice = np.array([rng.choice(n, p=row) for row in q])
        blocks.append(snack.Block(n, values, choice, seen, rng.integers(0, 20, size=trials)))
    return snack.SnackData(tuple(blocks), tuple(range(20)))


def test_elasticity_recovers_a_known_beta():
    rng = np.random.default_rng(5)
    data = _synthetic_data(rng, beta_true=-0.5)
    w = data.weights()
    beta, lam = snack.elasticity(data, w, "all", snack.included_rows(data, "all"))
    assert beta == pytest.approx(-0.5, abs=0.15)
    assert lam == pytest.approx(0.8, rel=0.15)


def test_held_out_uses_the_other_set_sizes():
    rng = np.random.default_rng(6)
    data = _synthetic_data(rng, beta_true=0.0, trials=150)
    w = data.weights()
    ho = snack.held_out(data, w, "CL", 0.0, "all")
    others = snack._parts(data, w, "CL", 0.0, "all", [1, 2, 3], snack.included_rows(data, "all"))
    assert ho["lambda"][0] == pytest.approx(snack.fit_sensitivity(others), rel=1e-12)
    assert ho["L"] == pytest.approx(np.mean(ho["by_size"]))


def test_weights_follow_the_participant_draw():
    rng = np.random.default_rng(7)
    data = _synthetic_data(rng, beta_true=0.0, trials=30)
    draw = np.array([0, 0, 3] + [5] * 17)
    w = data.weights(draw)
    for b, wk in zip(data.blocks, w, strict=True):
        expected = np.where(b.participant == 0, 2.0, 0.0) + (b.participant == 3) + 17.0 * (b.participant == 5)
        assert np.array_equal(wk, expected)


def test_loader_filters_trials_and_reads_gaze(tmp_path):
    for n in snack.SET_SIZES:
        rows = []
        for t, (subject, choice, missing, gaze_nan) in enumerate(
            [(4, 0, False, False), (2, n - 1, False, True), (2, n, False, False), (9, 1, True, False)]
        ):
            row = {"setsize": n, "subject": subject, "trial": t, "choice": choice}
            for i in range(n):
                row[f"item_value_{i}"] = np.nan if (missing and i == 2) else float(i % 7 - 3)
                row[f"cumulative_gaze_{i}"] = np.nan if (gaze_nan and i == 0) else (0.1 if i < 3 else 0.0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(tmp_path / f"setsize-{n}_desc-data.csv", index=False)
    data = snack.load_snack(tmp_path)
    assert data.participants == (2, 4, 9)
    for b in data.blocks:
        assert b.trials == 2  # choice out of range and missing rating are dropped
        assert list(b.choice) == [0, b.n - 1]
        assert list(b.participant) == [1, 0]
        assert b.seen[0, :3].all() and not b.seen[0, 3:].any()
        assert not b.seen[1, 0]  # a missing gaze value counts as not looked at


def test_warm_start_reaches_the_same_root():
    rng = np.random.default_rng(8)
    data = _synthetic_data(rng, beta_true=-0.3, trials=200, seen_share=0.6)
    w = data.weights(rng.integers(0, 20, size=20))
    rows = snack.included_rows(data, "seen")
    for family in snack.FAMILIES:
        parts = snack._parts(data, w, family, 0.5, "seen", [0, 2, 3], rows)
        cold = snack.fit_sensitivity(parts)
        for start in (cold * 0.01, cold * 0.9, cold * 3.0, cold * 100.0):
            assert snack.fit_sensitivity(parts, start) == pytest.approx(cold, rel=1e-12)


def test_compressed_likelihood_equals_the_item_level_likelihood():
    rng = np.random.default_rng(9)
    for family in snack.FAMILIES:
        values = rng.integers(-3, 4, size=(80, 16)).astype(float)
        S = snack.scores(values, family)
        mask = rng.random(S.shape) < 0.5
        choice = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in mask])
        mask[np.arange(80), choice] = True
        part = snack.make_part(S, choice, mask, rng.integers(0, 3, 80).astype(float), c=1.7)
        q = snack.probabilities(0.8 * 1.7, S, mask)
        direct = np.log2(16 * q[np.arange(80), choice])
        assert np.allclose(snack.log_scores(0.8, part), direct, rtol=1e-12, atol=1e-12)
        loglik = float(part.w @ np.log(q[np.arange(80), choice]))
        assert snack.weighted_loglik(0.8, [part]) == pytest.approx(loglik, rel=1e-12)


def test_solver_needs_few_evaluations_even_from_the_root(monkeypatch):
    rng = np.random.default_rng(10)
    data = _synthetic_data(rng, beta_true=0.2, trials=150)
    parts = snack._parts(data, data.weights(), "MLBA_avg", 0.5, "all", [0, 1, 3])
    root = snack.fit_sensitivity(parts)
    calls = {"n": 0}
    original = snack._grad_hess

    def counting(lam, items):
        calls["n"] += 1
        return original(lam, items)

    monkeypatch.setattr(snack, "_grad_hess", counting)
    for start in (None, root, root * 0.5, root * 2.0):
        calls["n"] = 0
        assert snack.fit_sensitivity(parts, start) == pytest.approx(root, rel=1e-12)
        assert calls["n"] <= 15
