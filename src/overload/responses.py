"""What participants reported after each choice, and how long they took to choose.

Choice overload is a feeling as much as a choice. After every trial the study asked six questions, each
answered from 1 to 5, the classic measures of overload in the literature. Together with the regret of the
chosen option and the time spent on the choice page, they say how hard a trial felt and how well it went.
This module turns the raw survey export into one tidy row per participant and trial. It never keeps an
identifier, and participants are numbered in the order of the data file.

The six questions, in the order asked.

1. How satisfied are you with your decision? (1 very dissatisfied, 5 very satisfied)
2. How certain are you that you made the best decision? (1 very uncertain, 5 very certain)
3. How confused did you feel while performing this task? (1 not confused at all, 5 very confused)
4. How frustrated did you feel while performing this task? (1 not frustrated at all, 5 very frustrated)
5. How complex was this trial to choose the best option? (1 very little complexity, 5 very high complexity)
6. How uncertain are you about choosing the best option? (printed anchors 1 very uncertain, 5 very certain)

Question 6 asks about uncertainty, but its printed anchors run the other way. Participants answered it as an
uncertainty question. Their answers fall with satisfaction and certainty and rise with confusion,
frustration and complexity (``overload_index`` reports these correlations). It is therefore kept as worded,
a higher answer meaning more uncertain. The overload index aligns all six so that a higher value always means
more overloaded, reversing satisfaction and certainty, and averages them in scale points from 1 to 5.

Regret is 100 minus the designers' score of the chosen option, so 0 means a best option was chosen and 80
the worst. Decision time is the time on the choice page in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import LAPTOP_STIMULI, _stimuli

PREFIX = {"file": "ci_co_fl_", "laptop": "em_co_lp_"}

# name, direction (+1 when a higher answer means more overloaded), short wording
QUESTIONS: tuple[tuple[str, int, str], ...] = (
    ("satisfied", -1, "How satisfied are you with your decision?"),
    ("certain", -1, "How certain are you that you made the best decision?"),
    ("confused", +1, "How confused did you feel while performing this task?"),
    ("frustrated", +1, "How frustrated did you feel while performing this task?"),
    ("complex", +1, "How complex was this trial to choose the best option?"),
    ("uncertain", +1, "How uncertain are you about choosing the best option?"),
)
ITEMS: tuple[str, ...] = tuple(q[0] for q in QUESTIONS)


def aligned(frame: pd.DataFrame) -> pd.DataFrame:
    """The six answers turned so that a higher value always means more overloaded (6 - answer when reversed)."""
    return pd.DataFrame({name: frame[name] if sign > 0 else 6 - frame[name] for name, sign, _ in QUESTIONS})


def load_responses(csv_path: str | Path, laptop: str = "storyboard") -> pd.DataFrame:
    """One row per participant and trial of both tasks, with the six answers, regret, time and the choice.

    Columns. ``person`` (0, 1, ... in file order), ``task``, ``trial`` (0 to 5), ``key``, ``options``,
    ``information``, ``choice`` (option index in display order, -1 if the answer matched no option),
    ``position`` (1 for the first option listed), ``regret``, ``seconds``, the six answers under the names
    of ``ITEMS`` and ``overload``, the mean of the six aligned answers.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    files = {"file": "file_trials.json", "laptop": LAPTOP_STIMULI[laptop]}
    rows = []
    for task, prefix in PREFIX.items():
        spec = _stimuli(files[task])
        for j, (key, s) in enumerate(spec.items()):
            options = [o.lower() for o in s["options"]]
            index = {o: i for i, o in enumerate(options)}
            col = s["column"]
            assert col == f"{prefix}{j + 1}", (col, prefix, j)
            choice = df[col].map(lambda x, ix=index: ix.get(x.strip().lower(), -1) if isinstance(x, str) else -1)
            block = pd.DataFrame(
                {
                    "person": np.arange(len(df)),
                    "task": task,
                    "trial": j,
                    "key": key,
                    "options": len(options),
                    "information": s["information"],
                    "choice": choice.to_numpy(dtype=int),
                    "regret": pd.to_numeric(df[f"{col}_regret"], errors="coerce"),
                    "seconds": pd.to_numeric(df[f"{col}_timer"], errors="coerce"),
                }
            )
            block["position"] = np.where(block["choice"] >= 0, block["choice"] + 1, -1)
            for k, name in enumerate(ITEMS, start=1):
                block[name] = pd.to_numeric(df[f"{col}_survey_{k}"], errors="coerce")
            rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    out["overload"] = aligned(out).mean(axis=1)
    return out
