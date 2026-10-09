"""End to end run of the confirmatory H1 experiment on synthetic data in the format of the snack files."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from overload import snack
from overload.experiments import h1_confirmatory


def _write_synthetic(folder, rng, participants=6, trials=12):
    for n in snack.SET_SIZES:
        rows = []
        for s in range(participants):
            for t in range(trials):
                values = rng.integers(-3, 4, size=n).astype(float)
                seen = rng.random(n) < 0.6
                q = np.exp(0.8 * values) * np.where(seen, 1.0, 0.05)
                choice = int(rng.choice(n, p=q / q.sum()))
                row = {"setsize": n, "subject": 100 + s, "trial": t, "choice": choice}
                for i in range(n):
                    row[f"item_value_{i}"] = values[i]
                    row[f"cumulative_gaze_{i}"] = float(rng.random()) if seen[i] else 0.0
                rows.append(row)
        pd.DataFrame(rows).to_csv(folder / f"setsize-{n}_desc-data.csv", index=False)


def test_confirmatory_runs_end_to_end(tmp_path):
    rng = np.random.default_rng(11)
    data_dir, out = tmp_path / "snack", tmp_path / "out"
    data_dir.mkdir()
    _write_synthetic(data_dir, rng)
    summary = h1_confirmatory.run(data_dir, out, replicates=4, jobs=1)
    assert set(summary) >= {"C1", "C2", "C3", "descriptives", "provenance"}
    assert len(summary["C1"]["per_size"]) == 4
    assert set(summary["C2"]["primary"]) == set(h1_confirmatory.PRIMARY)
    assert isinstance(summary["C3"]["narrowing_supported"], bool)
    assert (out / "table_h1_confirmatory.md").exists()
    reps = [json.loads(line) for line in (out / "replicates.jsonl").open()]
    assert [r["b"] for r in reps] == [0, 1, 2, 3]


def test_replicates_are_reproducible(tmp_path):
    rng = np.random.default_rng(12)
    data_dir = tmp_path / "snack"
    data_dir.mkdir()
    _write_synthetic(data_dir, rng, participants=5, trials=8)
    data = snack.load_snack(data_dir)
    a = h1_confirmatory.run_replicate(data, 3)
    b = h1_confirmatory.run_replicate(data, 3)
    assert a == b
    assert h1_confirmatory.run_replicate(data, 4) != a
