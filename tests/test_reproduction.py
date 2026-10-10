"""Reproduction of the earlier (AAMAS) pipeline from the raw study data.

These tests need the study's CSV file, which is not part of the repository. Point the environment
variable OVERLOAD_DATA at it, otherwise the tests are skipped.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from overload.data import load_study
from overload.engine import precompute_bases, run_replicate, subset_completeness
from overload.models import baseline_models

FIX = Path(__file__).parent / "fixtures"
DATA = os.environ.get("OVERLOAD_DATA")
pytestmark = pytest.mark.skipif(not DATA or not Path(DATA).exists(), reason="set OVERLOAD_DATA to the study CSV")

OLD = {"QR": "qra", "CL": "cl", "DbS_sum": "dbs", "MLBA_sum": "mlba"}


@pytest.fixture(scope="module")
def study():
    # The earlier pipeline used the laptop definitions that Amendment 4 corrected, so the reproduction
    # test runs on those definitions.
    tasks = load_study(DATA, laptop="aamas")
    models = baseline_models()
    return tasks, models, precompute_bases(tasks, models)


def test_trials_as_displayed(study):
    tasks, _, _ = study
    assert [t.n for t in tasks["file"].trials] == [8, 8, 16, 16, 32, 32]
    assert [t.n for t in tasks["laptop"].trials] == [8, 8, 16, 16, 32, 30]
    for task in tasks.values():
        for t in task.trials:
            assert len(set(o.lower() for o in t.options)) == t.n


def test_rubric_shares(study):
    tasks, _, _ = study
    ref = json.loads((FIX / "rubric_weights_reference.json").read_text())
    for name, task in tasks.items():
        k = len(ref[name]["share"])
        np.testing.assert_allclose(task.shares[:k], ref[name]["share"], atol=1e-12)
        assert task.shares.sum() == pytest.approx(1.0)


def test_point_estimates_match_the_earlier_pipeline(study):
    tasks, models, bases = study
    ref = json.loads((FIX / "aamas_reference.json").read_text())
    rec = run_replicate(tasks, models, bases, -1)
    for name in ("file", "laptop"):
        np.testing.assert_allclose(rec[name]["Lstar"], ref[name]["Lmm"], atol=1e-12)
        for new, old in OLD.items():
            np.testing.assert_allclose(rec[name][new]["ho"], ref[name][old]["ho"], atol=1e-6)
            np.testing.assert_allclose(rec[name][new]["fit"], ref[name][old]["fit"], atol=1e-6)
            np.testing.assert_allclose(rec[name][new]["O_ho"], ref[name][old]["Oho"], atol=1e-6)


def test_published_completeness_values(study):
    tasks, models, bases = study
    rec = run_replicate(tasks, models, bases, -1)
    published = {
        ("file", "QR"): 21.9, ("file", "CL"): 29.5, ("file", "DbS_sum"): 23.9, ("file", "MLBA_sum"): 22.6,
        ("laptop", "QR"): 47.2, ("laptop", "CL"): 50.3, ("laptop", "DbS_sum"): 49.6, ("laptop", "MLBA_sum"): 50.0,
    }  # fmt: skip
    for (name, model), value in published.items():
        assert round(subset_completeness(rec[name], model, "ho", range(6)), 1) == value


@pytest.mark.parametrize("b", [0, 1, 499, 999])
def test_bootstrap_replicates_match_the_earlier_pipeline(study, b):
    tasks, models, bases = study
    ref = json.loads((FIX / "aamas_bootstrap_reference.json").read_text())[str(b)]
    rec = run_replicate(tasks, models, bases, b)
    for name in ("file", "laptop"):
        np.testing.assert_allclose(rec[name]["Lstar"], ref[name]["Lmm"], atol=1e-12)
        for new, old in OLD.items():
            np.testing.assert_allclose(rec[name][new]["ho"], ref[name][old]["ho"], atol=1e-6)
            np.testing.assert_allclose(rec[name][new]["O_ho"], ref[name][old]["Oho"], atol=1e-6)
