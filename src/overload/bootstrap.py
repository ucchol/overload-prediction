"""Participant bootstrap and interval summaries of Section 4 of ANALYSIS_PLAN.md.

Replicate b seeds a generator with 10000 + b and draws file selection participants before laptop
participants, each with replacement and of the original size. Every model is refitted in every fold of
every replicate. Replicate b = -1 denotes the original sample.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .data import TASKS, Task

SEED_OFFSET = 10000


def resample_indices(tasks: Mapping[str, Task], b: int) -> dict[str, np.ndarray | None]:
    """Panel indices of replicate b for every task, None for the original sample (b = -1)."""
    if b < 0:
        return {name: None for name in TASKS}
    rng = np.random.default_rng(SEED_OFFSET + b)
    out: dict[str, np.ndarray | None] = {}
    for name in TASKS:
        n = len(tasks[name].panel)
        out[name] = rng.integers(0, n, size=n)
    return out


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(values, [a, 100.0 - a])
    return float(lo), float(hi)


def basic_interval(point: float, values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = percentile_interval(values, level)
    return 2.0 * point - hi, 2.0 * point - lo


def two_sided_p(values: np.ndarray) -> float:
    """Twice the smaller share of replicates at or below and at or above zero, capped at one."""
    below = float(np.mean(values <= 0.0))
    above = float(np.mean(values >= 0.0))
    return min(1.0, 2.0 * min(below, above))


def summarize(point: float, values: Sequence[float]) -> dict[str, float | bool | list[float]]:
    """Point estimate, percentile and basic intervals, p value and whether zero is clearly excluded."""
    v = np.asarray(values, dtype=float)
    pct = percentile_interval(v)
    bas = basic_interval(point, v)
    clear = (pct[0] > 0 and bas[0] > 0) or (pct[1] < 0 and bas[1] < 0)
    return {
        "point": float(point),
        "percentile": [pct[0], pct[1]],
        "basic": [bas[0], bas[1]],
        "p": two_sided_p(v),
        "clear": bool(clear),
    }


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm adjusted p values for one family."""
    names = sorted(pvalues, key=pvalues.get)
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def chunks(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
