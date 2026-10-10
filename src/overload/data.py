"""Loading the original study into tasks, trials and a choice panel.

A task holds its six trials, the rubric shares of its coded attributes and a panel with one entry per
respondent, listing that respondent's matched choices as (trial index, option index) pairs. Counts
for any resample of respondents are rebuilt from the panel, so resampling never touches the raw file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .coding import FILE_ATTRIBUTES, LAPTOP_ATTRIBUTES, code_file_trial, code_laptop_trial

TASKS: tuple[str, ...] = ("file", "laptop")

# Attributes entering the rubric regression. Laptop price is coded but constant within every trial,
# so it carries no information about differences between options and is left out (share zero).
_SHARE_ATTRIBUTES = {"file": 5, "laptop": 6}


@dataclass(frozen=True)
class Trial:
    """One choice set as displayed to participants."""

    key: str
    information: str  # "low" or "high"
    intended_options: int
    options: tuple[str, ...]
    X: np.ndarray  # coded attribute levels, options by attributes
    U: np.ndarray  # rubric scores
    column: str

    @property
    def n(self) -> int:
        return len(self.options)


@dataclass
class Task:
    """One task of the study, its trials and its choice panel."""

    name: str
    attributes: tuple[str, ...]
    trials: tuple[Trial, ...]
    shares: np.ndarray
    share_r2: float
    panel: list[list[tuple[int, int]]] = field(repr=False)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(t.key for t in self.trials)

    def counts(self, respondents: Sequence[int] | None = None) -> list[np.ndarray]:
        """Choice counts per trial, for all respondents or for a (re)sample of panel indices."""
        out = [np.zeros(t.n) for t in self.trials]
        rows = range(len(self.panel)) if respondents is None else respondents
        for r in rows:
            for trial_index, option_index in self.panel[r]:
                out[trial_index][option_index] += 1.0
        return out

    def trials_at(self, information: str) -> list[int]:
        return [j for j, t in enumerate(self.trials) if t.information == information]


def _stimuli(name: str) -> dict:
    return json.loads(resources.files("overload.stimuli").joinpath(name).read_text())


def rubric_shares(trials: Sequence[Trial], n_attributes: int) -> tuple[np.ndarray, float]:
    """Nonnegative rubric shares from within trial centered rubric scores on coded attributes.

    Returns the shares over all coded attributes (zero for attributes left out of the regression)
    and the within trial variance of the rubric scores that the nonnegative fit explains.
    """
    Xc = np.vstack([t.X[:, :n_attributes] - t.X[:, :n_attributes].mean(axis=0) for t in trials])
    yc = np.concatenate([t.U - t.U.mean() for t in trials])
    w, _ = nnls(Xc, yc)
    r2 = 1.0 - float(((yc - Xc @ w) ** 2).sum()) / float((yc**2).sum())
    shares = np.zeros(trials[0].X.shape[1])
    shares[:n_attributes] = w / w.sum()
    return shares, r2


def _match(value: object, index: dict[str, int]) -> int | None:
    if isinstance(value, str):
        return index.get(value.strip().lower())
    return None


def load_task(df: pd.DataFrame, name: str, stimuli: str | None = None) -> Task:
    """Build one task from the data frame of the original study.

    ``stimuli`` names the option definitions in ``overload.stimuli``, by default ``{name}_trials.json``.
    """
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}")
    spec = _stimuli(stimuli or f"{name}_trials.json")
    rubric = _stimuli("rubric.json")[name]
    coder = code_file_trial if name == "file" else code_laptop_trial
    attributes = FILE_ATTRIBUTES if name == "file" else LAPTOP_ATTRIBUTES

    trials = []
    for key, s in spec.items():
        options = s["options"]
        U = np.array([rubric[s["rubric_key"]][o] for o in options], dtype=float)
        trials.append(
            Trial(
                key=key,
                information=s["information"],
                intended_options=s["intended_options"],
                options=tuple(options),
                X=coder(options),
                U=U,
                column=s["column"],
            )
        )

    indices = [{o.lower(): i for i, o in enumerate(t.options)} for t in trials]
    panel: list[list[tuple[int, int]]] = []
    for _, row in df.iterrows():
        obs = []
        for j, t in enumerate(trials):
            i = _match(row[t.column], indices[j])
            if i is not None:
                obs.append((j, i))
        if obs:
            panel.append(obs)

    shares, r2 = rubric_shares(trials, _SHARE_ATTRIBUTES[name])
    return Task(
        name=name,
        attributes=attributes,
        trials=tuple(trials),
        shares=shares,
        share_r2=r2,
        panel=panel,
    )


LAPTOP_STIMULI = {
    # The study's storyboard, which the designers' rubric scores follow (plan, Amendment 4).
    "storyboard": "laptop_trials.json",
    # The definitions of the earlier (AAMAS) pipeline, which differ in laptop trials 5 and 6. Kept only to
    # test that the rebuilt pipeline reproduces the earlier one.
    "aamas": "laptop_trials_aamas.json",
}


def load_study(csv_path: str | Path, laptop: str = "storyboard") -> dict[str, Task]:
    """Load both tasks of the original study from its data file.

    ``laptop`` selects the laptop option definitions, "storyboard" (the study's displays) or "aamas".
    """
    if laptop not in LAPTOP_STIMULI:
        raise ValueError(f"unknown laptop definitions {laptop!r}")
    df = pd.read_csv(csv_path)
    return {
        "file": load_task(df, "file"),
        "laptop": load_task(df, "laptop", LAPTOP_STIMULI[laptop]),
    }
