"""One replicate of an experiment: resample, refit every model, record per trial measures.

A replicate's record holds, for every task, the attainable log scores of its trials and, for every
model, the held out, fitted and alone log scores, the held out and fitted overlaps and the parameter
estimates. Only aggregate values are recorded, never individual choices.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import scipy

from . import __version__
from .bootstrap import resample_indices
from .data import TASKS, Task
from .fit import evaluate
from .metrics import (
    attainable,
    completeness,
    effective_number,
    log_score,
    overlap,
    sampled_effective_number,
    shares,
)
from .models import Model

PLAN = Path(__file__).resolve().parents[2] / "ANALYSIS_PLAN.md"
CONCENTRATION_DRAWS = 2000


def concentration_seed(task_index: int, trial_index: int, b: int, role: int) -> int:
    """Seed of a sampled effective number (plan, Amendment 2).

    ``role`` is 1 for a model's prediction and 2 for the observed shares of the original sample, which
    stand in for the true probabilities of the bootstrap world. Replicate b = -1 is the original sample.
    """
    return 1_000_000 * role + 1_000 * (b + 1) + 10 * trial_index + task_index


def precompute_bases(tasks: Mapping[str, Task], models: Sequence[Model]) -> dict[str, dict[str, list[np.ndarray]]]:
    """Scores or base distributions of every model in every trial. They do not depend on choices."""
    return {name: {m.name: m.bases(tasks[name]) for m in models} for name in TASKS}


def run_replicate(
    tasks: Mapping[str, Task],
    models: Sequence[Model],
    bases: Mapping[str, Mapping[str, list[np.ndarray]]],
    b: int,
    alone: bool = False,
    concentration: Sequence[str] = (),
) -> dict:
    """Record of replicate b (b = -1 is the original sample).

    For every model named in ``concentration`` the record also holds the effective number of the observed
    shares and the sampled effective number of the model's held out predictions at the observed number of
    choices. For the original sample it adds the sampled effective number of the observed shares.
    """
    idx = resample_indices(tasks, b)
    record: dict = {"b": b}
    for task_index, name in enumerate(TASKS):
        task = tasks[name]
        counts = task.counts(idx[name])
        observed = [shares(c) for c in counts]
        rec: dict = {
            "Lstar": [attainable(c) for c in counts],
            "Lplug": [attainable(c, corrected=False) for c in counts],
        }
        if concentration:
            rec["E_obs"] = [effective_number(p) for p in observed]
            rec["n"] = [int(c.size) for c in counts]
            if b < 0:
                rec["E_samp_obs"] = [
                    sampled_effective_number(
                        p, int(c.sum()), CONCENTRATION_DRAWS, concentration_seed(task_index, j, b, role=2)
                    )
                    for j, (p, c) in enumerate(zip(observed, counts, strict=True))
                ]
        for m in models:
            ev = evaluate(m, task, counts, bases[name][m.name], alone=alone)
            entry = {
                "ho": [log_score(c, q) for c, q in zip(counts, ev.held_out, strict=True)],
                "fit": [log_score(c, q) for c, q in zip(counts, ev.fitted, strict=True)],
                "O_ho": [overlap(p, q) for p, q in zip(observed, ev.held_out, strict=True)],
                "O_fit": [overlap(p, q) for p, q in zip(observed, ev.fitted, strict=True)],
                "theta": ev.params["pooled"],
                "theta_ho": ev.params["held_out"],
            }
            if alone:
                entry["alone"] = [log_score(c, q) for c, q in zip(counts, ev.alone, strict=True)]
                entry["theta_alone"] = ev.params["alone"]
            if m.name in concentration:
                entry["E_samp_ho"] = [
                    sampled_effective_number(
                        q, int(c.sum()), CONCENTRATION_DRAWS, concentration_seed(task_index, j, b, role=1)
                    )
                    for j, (q, c) in enumerate(zip(ev.held_out, counts, strict=True))
                ]
            rec[m.name] = entry
        record[name] = rec
    return record


def subset_completeness(
    rec: Mapping,
    model: str,
    which: str,
    trials: Sequence[int],
    benchmark: Sequence[float] | None = None,
) -> float:
    """Completeness of ``which`` ("ho", "fit" or "alone") over the given trial indices.

    ``benchmark`` gives the attainable log score of every trial. Bootstrap replicates pass the full
    sample values (plan, Amendment 1), otherwise the record's own values are used.
    """
    bench = rec["Lstar"] if benchmark is None else benchmark
    return completeness([rec[model][which][t] for t in trials], [bench[t] for t in trials])


def subset_overlap(rec: Mapping, model: str, which: str, trials: Sequence[int]) -> float:
    return float(np.mean([rec[model][which][t] for t in trials]))


def plan_digest() -> str:
    """SHA-256 of the analysis plan the results were produced under."""
    if not PLAN.exists():
        return "plan file not found, install the package in editable mode from the repository"
    return hashlib.sha256(PLAN.read_bytes()).hexdigest()


def provenance(extra: Mapping | None = None) -> dict:
    """Plan digest, versions and time, written next to every result."""
    info = {
        "plan_sha256": plan_digest(),
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": metadata.version("pandas"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        info.update(extra)
    return info


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
