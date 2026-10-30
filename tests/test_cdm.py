"""The Caplin, Dean and Martin models and the Amendment 7 analysis on simulated data. No real data are used."""

from __future__ import annotations

import numpy as np
import pytest

from overload import cdm
from overload.experiments import stopping_cdm as sc
from overload.search import race_length, race_probabilities
from overload.snack import scores as snack_scores


def _displays(rng: np.random.Generator, rounds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = rng.choice(cdm.SIZES, size=rounds)
    c = rng.choice(cdm.COMPLEXITIES, size=rounds)
    values = np.full((rounds, cdm.MAX_N), np.nan)
    for r in range(rounds):
        values[r, : n[r]] = np.minimum(np.ceil(rng.exponential(4.0, n[r])), 35.0)
    return values, n, c


def test_sample_until_return_on_dollar_levels_matches_the_item_formulas():
    rng = np.random.default_rng(1)
    values, n, _ = _displays(rng, 30)
    choice = np.zeros(30, int)
    lv = cdm.levels(values, choice)
    for lam in (0.0, 0.2, 1.3):
        logp, looks = cdm.sr_levels(lam, lv)
        for r in range(30):
            v = values[r, : n[r]]
            q = np.exp(lam * (v - v.max()))
            p = race_probabilities(q / q.sum())
            level = np.rint(v - cdm.LEVELS[0]).astype(int)
            assert np.allclose(np.exp(logp[r, level]), p, rtol=1e-9)
            assert looks[r] == pytest.approx(race_length(q / q.sum()), rel=1e-9)


def test_display_fit_equals_the_round_fit():
    rng = np.random.default_rng(2)
    values, n, _ = _displays(rng, 12)
    values = np.repeat(values, 20, axis=0)
    n = np.repeat(n, 20)
    lam_true = 0.4
    lv0 = cdm.levels(values, np.zeros(values.shape[0], int))
    logp, _ = cdm.sr_levels(lam_true, lv0)
    choice = np.zeros(values.shape[0], int)
    for r in range(values.shape[0]):
        v = values[r, : n[r]]
        p = np.exp(logp[r, np.rint(v - 1).astype(int)])
        choice[r] = rng.choice(n[r], p=p / p.sum())
    lv = cdm.levels(values, choice)
    w = rng.integers(0, 3, size=values.shape[0]).astype(float)
    a = cdm.fit_sr_displays(cdm.Displays.of(lv), w)
    b = cdm.fit_sr(lv, w)
    assert a == pytest.approx(b, rel=1e-7)
    assert a == pytest.approx(lam_true, abs=0.15)


def test_static_scores_match_the_snack_scores_on_full_sets():
    rng = np.random.default_rng(3)
    v = rng.integers(1, 30, size=(5, 10)).astype(float)
    full = np.full((5, cdm.MAX_N), np.nan)
    full[:, :10] = v
    for fam in ("DbS_avg", "MLBA_avg"):
        ours = cdm.static_scores(full, fam, mlba_scale=1.0)[:, :10]
        assert np.allclose(ours, snack_scores(v, fam))


def test_logistic_fit_is_the_penalized_maximum():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(400, 2)) * [3.0, 0.5] + [10.0, 2.0]
    y = rng.random(400) < 1.0 / (1.0 + np.exp(-(0.8 * (x[:, 0] - 10.0))))
    w = rng.integers(1, 3, size=400).astype(float)
    beta, mu, sd = cdm.fit_logistic(x, y, w)
    Z = np.column_stack([np.ones(400), (x - mu) / sd])

    def objective(b):
        eta = Z @ b
        return w @ (y * eta - np.logaddexp(0.0, eta)) - 0.5 * np.sum(b[1:] ** 2)

    best = objective(beta)
    for _ in range(50):
        assert objective(beta + rng.normal(scale=1e-3, size=3)) <= best + 1e-9
    assert np.all(np.isfinite(cdm.logistic_log2(beta, mu, sd, x, y)))


def test_logistic_fit_with_one_outcome_takes_its_limit():
    x = np.array([[1.0], [2.0], [3.0], [5.0]])
    w = np.array([1.0, 2.0, 0.0, 1.0])
    for outcome in (True, False):
        beta, mu, sd = cdm.fit_logistic(x, np.full(4, outcome), w)
        assert np.isinf(beta[0]) and (beta[0] > 0) == outcome and beta[1] == 0.0
        scores = cdm.logistic_log2(beta, mu, sd, np.array([[2.0], [4.0]]), np.array([outcome, not outcome]))
        assert scores[0] == 0.0
        assert scores[1] == pytest.approx(np.log2(1e-300))


def _synthetic(process: str, seed: int) -> cdm.CDMData:
    """Rounds and selections simulated with the design check of scripts/simulate_cdm_tests.py."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "simulate_cdm_tests.py"
    spec = importlib.util.spec_from_file_location("simulate_cdm_tests", path)
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    rng = np.random.default_rng(seed)
    people, per = 40, 18
    rounds = people * per
    values, n, c = _displays(rng, rounds)
    base = cdm.CDMData(
        experiment=np.repeat(np.arange(people) % 2, per),
        n=n,
        complexity=c,
        participant=np.repeat(np.arange(people), per),
        participants=tuple(range(people)),
        values=values,
        choice=np.zeros(rounds, int),
        final_time=np.ones(rounds),
        sel_round=np.zeros(0, int),
        sel_value=np.zeros(0),
        sel_time=np.zeros(0),
        sel_position=np.zeros(0, int),
        sel_better=np.zeros(0, int),
        sel_stop=np.zeros(0, bool),
        sel_first=np.zeros(0, bool),
    )
    return sim.simulate(base, process, seed)


@pytest.mark.parametrize(
    ("process", "absolute", "order"),
    [("satisficing_in_order", True, True), ("first_return_uniform", False, False)],
)
def test_tests_tell_the_processes_apart(process, absolute, order):
    sc._STATE.update(data=_synthetic(process, 11))
    point = sc.run_replicate(-1)
    reps = [sc.run_replicate(b) for b in range(30)]
    s = sc.summarize_all(point, reps)
    stop = s["pooled"]["stop"]["value_minus_share"]["point"]
    assert (stop > 0.03) == absolute
    assert s["pooled"]["C2"]["order_matters"] == order
    assert {"C1", "C2", "C3"} <= set(s["pooled"])


def test_folds_split_participants_evenly():
    f = sc.folds(37)
    assert set(np.bincount(f)) <= {3, 4}
    assert np.array_equal(f, sc.folds(37))
