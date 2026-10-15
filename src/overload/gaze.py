"""Three open gaze datasets of value based choice among two or three foods, for the confirmatory test of Amendment 6.

Krajbich, Armel and Rangel (2010) offered two foods, Krajbich and Rangel (2011) three foods and Folke, Jacobsen, Fleming
and De Martino (2016) three snacks, each rated beforehand by the person choosing (Folke as willingness to pay). The
files are those prepared by Thomas, Molter, Krajbich, Heekeren and Mohr (2019, MIT license), where every value runs
from 0 or 1 to 10, so the top of every scale is 10. Folke's file also carries how confident people felt after each
choice, attached from the original file by row, after checking that the two files agree trial by trial.

The models are those of the snack data, written for a handful of items. With p a probability vector and e_k its k-th
elementary symmetric polynomial, sample until return chooses item i with probability p_i^2 sum_k (k + 1)! e_k(p without
i) and takes sum_k k! e_k(p) looks. Two look satisficing chooses as sample until return at the same sensitivity, and
takes the sample until return length on rho / P divided by P looks, with rho = softmax(lam v / 2) exp(lam (v - 10) / 2)
and P = sum rho.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import logsumexp, softmax

DATASETS: tuple[str, ...] = ("krajbich2010", "krajbich2011", "folke2016")
COMMIT = "4a4f230fb5aa2500d81bfc16b44da3478519ca4f"
V_TOP = 10.0
_GRID = np.concatenate([[0.0], np.geomspace(1e-2, 1e2, 21)])


@dataclass(frozen=True)
class GazeSet:
    """The included trials of one dataset."""

    name: str
    values: np.ndarray  # trials by items
    choice: np.ndarray  # chosen item index
    seconds: np.ndarray  # response time
    participant: np.ndarray  # index into participants
    participants: tuple[int, ...]
    confidence: np.ndarray | None = None  # Folke only

    @property
    def n(self) -> int:
        return int(self.values.shape[1])


def load_gaze(folder: str | Path) -> dict[str, GazeSet]:
    """Read the three prepared files and keep trials with finite values, a valid choice and a positive time."""
    folder = Path(folder)
    out = {}
    for name in DATASETS:
        df = pd.read_csv(folder / f"{name}.csv")
        confidence = None
        if name == "folke2016":
            orig = pd.read_csv(folder / "folke2016_original.csv")
            if len(orig) != len(df) or not np.allclose(df["rt"].to_numpy(float), orig["Choice_RT"].to_numpy(float)):
                raise ValueError("the Folke files do not agree trial by trial, so confidence cannot be attached")
            confidence = pd.to_numeric(orig["Conf"], errors="coerce").to_numpy(float)
        cols = [c for c in df.columns if c.startswith("item_value_")]
        values = df[cols].to_numpy(float)
        choice = pd.to_numeric(df["choice"], errors="coerce").to_numpy(float)
        rt = pd.to_numeric(df["rt"], errors="coerce").to_numpy(float)
        ok = (
            np.isfinite(values).all(axis=1)
            & np.isfinite(choice)
            & (choice >= 0)
            & (choice < len(cols))
            & (choice == np.round(choice))
            & np.isfinite(rt)
            & (rt > 0)
        )
        if np.any(values[ok] > V_TOP):
            raise ValueError(f"{name} has values above the top of the scale")
        ids = sorted({int(s) for s in df["subject"].to_numpy()[ok]})
        index = {s: i for i, s in enumerate(ids)}
        out[name] = GazeSet(
            name=name,
            values=values[ok],
            choice=choice[ok].astype(int),
            seconds=rt[ok] / 1000.0,
            participant=np.array([index[int(s)] for s in df["subject"].to_numpy()[ok]], dtype=int),
            participants=tuple(ids),
            confidence=None if confidence is None else confidence[ok],
        )
    return out


# ----------------------------------------------------------------------------- race formulas for a few items
def _elementary(p: np.ndarray) -> np.ndarray:
    """e_0 .. e_n of every row of p, trials by n + 1."""
    T, n = p.shape
    e = np.zeros((T, n + 1))
    e[:, 0] = 1.0
    for j in range(n):
        e[:, 1:] = e[:, 1:] + p[:, j : j + 1] * e[:, :-1]
    return e


def _factorials(m: int) -> np.ndarray:
    return np.cumprod(np.concatenate([[1.0], np.arange(1.0, m + 1)]))


def race_length(p: np.ndarray) -> np.ndarray:
    """Expected looks of sample until return for every row of p (rows sum to one)."""
    e = _elementary(p)
    return e @ _factorials(p.shape[1])


def race_probabilities(p: np.ndarray) -> np.ndarray:
    """Choice probabilities of sample until return for every row of p (rows sum to one)."""
    T, n = p.shape
    f = _factorials(n)[1:]  # (k + 1)! for k = 0 .. n - 1
    out = np.zeros((T, n))
    for i in range(n):
        e = _elementary(np.delete(p, i, axis=1))
        out[:, i] = p[:, i] ** 2 * (e @ f)
    return out / out.sum(axis=1, keepdims=True)


def two_look_looks(lam: float, values: np.ndarray, v_top: float = V_TOP) -> np.ndarray:
    """Expected looks of two look satisficing for every trial."""
    q = softmax(0.5 * lam * values, axis=1)
    rho = q * np.exp(0.5 * lam * (values - v_top))
    P = rho.sum(axis=1)
    return race_length(rho / P[:, None]) / P


def sr_looks(lam: float, values: np.ndarray) -> np.ndarray:
    return race_length(softmax(lam * values, axis=1))


# ----------------------------------------------------------------------------- fits
def chosen_log_cl(lam: float, values: np.ndarray, choice: np.ndarray) -> np.ndarray:
    z = lam * values
    return z[np.arange(choice.size), choice] - logsumexp(z, axis=1)


def chosen_log_sr(lam: float, values: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """log probability of the chosen item, floored at 1e-300 as in ``overload.metrics`` for vanishing probabilities."""
    P = race_probabilities(softmax(lam * values, axis=1))
    return np.log(np.maximum(P[np.arange(choice.size), choice], 1e-300))


def fit_cl(values: np.ndarray, choice: np.ndarray, w: np.ndarray) -> float:
    """Weighted maximum likelihood sensitivity of CL, lambda >= 0 (concave likelihood)."""
    keep = w > 0
    v, c, w = values[keep], choice[keep], w[keep]
    chosen = v[np.arange(c.size), c]

    def grad(lam: float) -> float:
        q = softmax(lam * v, axis=1)
        return float(w @ (chosen - (q * v).sum(axis=1)))

    if grad(0.0) <= 0.0:
        return 0.0
    hi = 1.0
    while grad(hi) > 0.0:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("choices are perfectly ordered by the values")
    return float(brentq(grad, 0.0, hi, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500))


def fit_sr(values: np.ndarray, choice: np.ndarray, w: np.ndarray) -> float:
    """Weighted maximum likelihood sensitivity of sample until return, grid then bounded Brent as in Amendment 5."""
    keep = w > 0
    v, c, w = values[keep], choice[keep], w[keep]

    def nll(lam: float) -> float:
        return -float(w @ chosen_log_sr(lam, v, c))

    vals = np.array([nll(g) for g in _GRID])
    k = int(np.argmin(vals))
    if k == _GRID.size - 1:
        raise RuntimeError("the search likelihood keeps rising at the largest sensitivity of the grid")
    lo, hi = _GRID[max(k - 1, 0)], _GRID[k + 1]
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    return float(res.x) if res.fun <= vals[k] else float(_GRID[k])
