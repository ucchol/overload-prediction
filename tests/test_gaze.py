"""The gaze dataset models and the Amendment 6 analysis on simulated data. No real data are used."""

from __future__ import annotations

import numpy as np
import pytest

from overload import gaze
from overload.experiments import two_look_gaze as tg
from overload.search import race_length, race_probabilities, two_look_satisficing


def test_small_set_race_formulas_match_the_general_ones():
    rng = np.random.default_rng(1)
    for n in (2, 3, 5):
        p = rng.dirichlet(np.ones(n), size=50)
        P = gaze.race_probabilities(p)
        E = gaze.race_length(p)
        for t in range(50):
            assert np.allclose(P[t], race_probabilities(p[t]), rtol=1e-12)
            assert E[t] == pytest.approx(race_length(p[t]), rel=1e-12)


def test_two_look_looks_match_the_item_model():
    rng = np.random.default_rng(2)
    values = rng.uniform(0, 10, size=(40, 3))
    for lam in (0.0, 0.3, 1.1):
        looks = gaze.two_look_looks(lam, values)
        for t in range(40):
            assert looks[t] == pytest.approx(two_look_satisficing(lam, values[t], 10.0)[1], rel=1e-12)


def _simulate(rng: np.random.Generator, name: str, n: int, participants: int, trials: int, lam: float) -> gaze.GazeSet:
    T = participants * trials
    values = rng.integers(0, 11, size=(T, n)).astype(float)
    choice, looks = np.zeros(T, int), np.zeros(T)
    for t in range(T):
        q = np.exp(0.5 * lam * values[t])
        q /= q.sum()
        h = np.exp(0.5 * lam * (values[t] - 10.0))
        hits, k = np.zeros(n, int), 0
        while True:
            i = int(rng.choice(n, p=q))
            k += 1
            if rng.random() < h[i]:
                hits[i] += 1
                if hits[i] == 2:
                    choice[t] = i
                    break
        looks[t] = k
    person = np.repeat(np.arange(participants), trials)
    speed = rng.uniform(0.7, 1.4, participants)[person]
    seconds = speed * (0.4 + 0.25 * looks) * np.exp(rng.normal(0, 0.15, T))
    confidence = -looks + rng.normal(0, 1.0, T) if name == "folke2016" else None
    return gaze.GazeSet(name, values, choice, seconds, person, tuple(range(participants)), confidence)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(7)
    sets = {name: _simulate(rng, name, n, 20, 40, 0.8) for name, n in zip(gaze.DATASETS, (2, 3, 3), strict=True)}
    tg._STATE.update(sets=sets)
    return tg.run_replicate(-1)


def test_sensitivities_are_recovered(synthetic):
    for name in gaze.DATASETS:
        assert np.allclose(synthetic["datasets"][name]["lambda"]["SR"], 0.8, atol=0.25)


def test_time_from_the_model_is_found(synthetic):
    t = synthetic["pooled"]["time"]
    assert t["TLS"] < t["static"] and t["TLS"] < t["SR"]
    s = synthetic["pooled"]["slow"]
    assert s["looks_TLS"] > s["looks_SR"] and s["looks_TLS"] > 0.1
    assert synthetic["datasets"]["folke2016"]["confidence"]["looks_TLS"] < 0


def test_law_fitted_to_everyone_recovers_the_time_per_look(synthetic):
    # simulated seconds = speed (0.4 + 0.25 looks) with speed averaging about 1.05
    for name in gaze.DATASETS:
        t0, tau = synthetic["datasets"][name]["law"]["TLS"]
        assert 0.15 < tau < 0.4 and 0.1 < t0 < 0.9


def test_static_choice_models_are_fitted_and_lose_to_the_search_model(synthetic):
    for name in gaze.DATASETS:
        D = synthetic["datasets"][name]
        assert set(D["L"]) == {"CL", "DbS", "MLBA", "SR"}
        assert all(len(v) == tg.FOLDS for v in D["lambda"].values())
        assert D["L"]["SR"] > D["L"]["DbS"]


def test_results_do_not_depend_on_the_order_of_the_items_on_screen():
    rng = np.random.default_rng(11)
    gs = _simulate(rng, "krajbich2011", 3, 12, 30, 0.8)
    perm = np.array([rng.permutation(3) for _ in range(gs.choice.size)])
    values = np.take_along_axis(gs.values, perm, axis=1)
    choice = np.argmax(perm == gs.choice[:, None], axis=1)
    shuffled = gaze.GazeSet(gs.name, values, choice, gs.seconds, gs.participant, gs.participants)
    a, b = tg._dataset(gs, None), tg._dataset(shuffled, None)
    assert a["slow"] == b["slow"]  # exact, since equal displays tie exactly
    for key in ("L", "time"):
        for m in a[key]:
            assert a[key][m] == pytest.approx(b[key][m], abs=1e-9)


def test_dbs_scores_of_two_items_are_the_sign_of_the_difference():
    from overload.snack import scores

    v = np.array([[3.0, 7.0], [5.0, 5.0], [9.0, 1.0]])
    assert np.array_equal(scores(v, "DbS_avg"), np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]))


def test_summary_and_table(synthetic):
    reps = [tg.run_replicate(b) for b in (0, 1)]
    summary = tg.summarize_all(synthetic, reps)
    assert {"G1", "G2", "G3"} <= set(summary["pooled"])
    assert {"TLS_plus_top_value", "TLS_plus_mean_value", "TLS_plus_top_gap"} <= set(summary["pooled"]["slow"])
    assert {"TLS_minus_CL_percent", "TLS_minus_DbS_percent", "TLS_minus_MLBA_percent"} <= set(
        summary["pooled"]["choices"]
    )
    for name in gaze.DATASETS:
        assert set(summary[name]["law"]["TLS"]) == {"t0", "tau"}
        assert summary[name]["looks"]["TLS"]["point"] > summary[name]["looks"]["SR"]["point"]
    text = tg.table(summary)
    assert "G1, time held out" in text and "seconds per look" in text


def test_folds_split_participants_evenly():
    f = tg.folds(37)
    assert set(np.bincount(f)) <= {3, 4}
    assert np.array_equal(f, tg.folds(37))
