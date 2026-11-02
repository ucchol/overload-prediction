"""H1 within one study, the cost of judging and the growth of misses, ANALYSIS_PLAN.md Amendment 10.

Data of Aguiar, Boccardi, Kashaev and Kim (2023), three cost files (1, 3 or 5 operations per prize).

Usage
    python -I -m overload.experiments.h1_within --data data/aguiar --out results/h1_within
    python -I -m overload.experiments.h1_within --data data/aguiar --simulate {fixed,falling} --out ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..fit import fit_search, fit_softmax
from ..search import sample_until_return

COSTS = ("low", "medium", "high")
PRIZES = np.array([50, 48, 30, 14, 12, 10, 0], dtype=float)
_A = np.array([0, 2 / 5, 0, 3 / 10, 0, 0, 3 / 10])
_L1 = np.array([1 / 2, 0, 0, 0, 0, 0, 1 / 2])
_L2 = np.array([0, 0, 1 / 2, 0, 0, 1 / 2, 0])
LOTTERIES = np.array([[0, 0, 0, 0, 1, 0, 0], _L1, _L2, (_L1 + _L2) / 2, (_L1 + _A) / 2, (_L2 + _A) / 2])
EV = LOTTERIES @ PRIZES  # option 0 is the sure 12 tokens
MENUS = [c for k in range(6) for c in itertools.combinations(range(1, 6), k)]  # Julia powerset(1:5) order
SEED_OFFSET = 100000
TOL = 1e-9


# ----------------------------------------------------------------------------- data
def load(folder: Path) -> dict[str, dict[int, np.ndarray]]:
    """Counts of the chosen option (index into the menu's options) per menu, for each cost level."""
    out = {}
    for c in COSTS:
        df = pd.read_csv(folder / f"menu_choice_{c}.csv")
        counts: dict[int, np.ndarray] = {}
        for m in range(2, len(MENUS) + 1):  # menu 1 is empty
            opts = options(m)
            ch = df.loc[df["x1"] == m, "x2"].to_numpy(int)
            assert np.isin(ch, opts).all(), (c, m)
            counts[m] = np.array([(ch == o).sum() for o in opts], dtype=float)
        out[c] = counts
    return out


def options(m: int) -> list[int]:
    return [0, *MENUS[m - 1]]


def size(m: int) -> int:
    return len(options(m))


def values(m: int, util: np.ndarray = EV) -> np.ndarray:
    return util[options(m)]


def best_mask(v: np.ndarray) -> np.ndarray:
    return v >= v.max() - TOL


# ----------------------------------------------------------------------------- measures
def slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))


def growth(rate_by_menu: dict[int, float], counts: dict[int, np.ndarray]) -> float:
    sizes = sorted({size(m) for m in counts})
    x, y, w = [], [], []
    for n in sizes:
        ms = [m for m in counts if size(m) == n and counts[m].sum() > 0]
        N = np.array([counts[m].sum() for m in ms])
        x.append(np.log2(n))
        y.append(float(np.sum(N * np.array([rate_by_menu[m] for m in ms])) / N.sum()))
        w.append(N.sum())
    return slope(np.array(x), np.array(y), np.array(w))


def observed_miss(counts: dict[int, np.ndarray], util: np.ndarray = EV) -> dict[int, float]:
    return {m: float(c[~best_mask(values(m, util))].sum() / c.sum()) for m, c in counts.items() if c.sum() > 0}


def held_out(counts: dict[int, np.ndarray], model: str) -> tuple[dict[int, float], float]:
    """Each menu size forecast from fits to the other sizes. Returns miss probabilities and mean bits."""
    miss, bits, N = {}, 0.0, 0.0
    for n in sorted({size(m) for m in counts}):
        train = [m for m in counts if size(m) != n]
        S, C = [values(m) for m in train], [counts[m] for m in train]
        lam = fit_softmax(S, C) if model == "CL" else fit_search("race", S, C)
        for m in [m for m in counts if size(m) == n]:
            v = values(m)
            p = softmax(lam * v) if model == "CL" else sample_until_return(lam, v)
            miss[m] = float(1.0 - p[best_mask(v)].sum())
            bits += float(counts[m] @ np.log2(np.maximum(p, 1e-300)))
            N += counts[m].sum()
    return miss, bits / N


def fit_crra(counts: dict[int, np.ndarray]) -> tuple[float, float]:
    """CRRA u(x) = (x + 1)^(1 - r) / (1 - r), r < 1, with a CL sensitivity, by maximum likelihood."""

    def nll(t):
        r, loglam = t
        u = LOTTERIES @ ((PRIZES + 1.0) ** (1 - r) / (1 - r))
        f = 0.0
        for m, c in counts.items():
            z = np.exp(loglam) * u[options(m)]
            f -= float(c @ (z - logsumexp(z)))
        return f

    best = min((minimize(nll, s, method="L-BFGS-B", bounds=[(-2, 0.95), (-8, 4)]) for s in
                ([0.0, -1.0], [0.5, 0.0], [-0.5, -2.0])), key=lambda r: r.fun)
    return float(best.x[0]), float(np.exp(best.x[1]))


def crra_util(r: float) -> np.ndarray:
    return LOTTERIES @ ((PRIZES + 1.0) ** (1 - r) / (1 - r))


def measures(data: dict[str, dict[int, np.ndarray]], r_crra: float) -> dict[str, float]:
    out: dict[str, float] = {}
    u = crra_util(r_crra)
    for c in COSTS:
        out[f"{c}|growth|observed"] = growth(observed_miss(data[c]), data[c])
        out[f"{c}|growth|observed_crra"] = growth(observed_miss(data[c], u), data[c])
        for model in ("CL", "SR"):
            miss, bits = held_out(data[c], model)
            out[f"{c}|growth|{model}"] = growth(miss, data[c])
            out[f"{c}|observed_minus|{model}"] = out[f"{c}|growth|observed"] - out[f"{c}|growth|{model}"]
            out[f"{c}|bits|{model}"] = bits
        for n in range(2, 7):
            ms = [m for m in data[c] if size(m) == n]
            N = sum(data[c][m].sum() for m in ms)
            om = observed_miss(data[c])
            out[f"{c}|miss_rate|n={n}"] = float(sum(om[m] * data[c][m].sum() for m in ms) / N)
    out["H1w"] = out["high|growth|observed"] - out["low|growth|observed"]
    out["H1w_medium_minus_low"] = out["medium|growth|observed"] - out["low|growth|observed"]
    out["H1w_crra"] = out["high|growth|observed_crra"] - out["low|growth|observed_crra"]
    return out


def resample(data: dict[str, dict[int, np.ndarray]], b: int) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(SEED_OFFSET + b)
    out = {}
    for c in COSTS:
        out[c] = {}
        for m in sorted(data[c]):
            p = data[c][m] / data[c][m].sum()
            out[c][m] = rng.multinomial(int(data[c][m].sum()), p).astype(float)
    return out


def simulated(data: dict[str, dict[int, np.ndarray]], kind: str, seed: int = 10) -> dict[str, dict[int, np.ndarray]]:
    """Choices drawn on the real menus with the real numbers of choices, from CL with one sensitivity at every
    cost ('fixed') or a sensitivity that falls with cost ('falling')."""
    rng = np.random.default_rng(seed)
    lam = {"fixed": dict(low=0.3, medium=0.3, high=0.3), "falling": dict(low=0.6, medium=0.3, high=0.1)}[kind]
    return {c: {m: rng.multinomial(int(k.sum()), softmax(lam[c] * values(m))).astype(float)
                for m, k in data[c].items()} for c in COSTS}


# ----------------------------------------------------------------------------- run
def run(folder: Path, out: Path, replicates: int, simulate: str | None = None) -> dict:
    t0 = time.time()
    data = load(folder)
    if simulate:
        data = simulated(data, simulate)
    r_crra, lam_crra = fit_crra(data["low"])
    point = measures(data, r_crra)
    reps = [measures(resample(data, b), r_crra) for b in range(replicates)]
    summary = {k: summarize(v, [r[k] for r in reps]) if reps else {"point": v} for k, v in point.items()}
    summary["crra_r_low_cost"] = r_crra
    summary["crra_lambda_low_cost"] = lam_crra
    h = summary["H1w"]
    summary["H1w_supported"] = bool(h.get("clear") and h["point"] > 0)
    summary["provenance"] = provenance({"replicates": replicates, "seconds": round(time.time() - t0, 1),
                                        "simulated": simulate})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "point.json", point)
    with (out / "replicates.jsonl").open("w") as fh:
        for b, r in enumerate(reps):
            fh.write(json.dumps({"b": b, **r}) + "\n")
    write_json(out / "summary.json", summary)
    (out / "table_h1_within.md").write_text(table(summary))
    return summary


def _f(e: dict, d: int = 3) -> str:
    if "percentile" not in e:
        return f"{e['point']:+.{d}f}"
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" + (" *" if e["clear"] else "")


def table(s: dict) -> str:
    L = ["# H1 within one study (Amendment 10), Aguiar, Boccardi, Kashaev and Kim (2023)", "",
         "| Measure | Low cost | Medium cost | High cost |", "|---|---|---|---|"]
    rows = [("Growth of misses per doubling, people", "growth|observed"),
            ("Growth, CL held out", "growth|CL"), ("Growth, sample until return held out", "growth|SR"),
            ("People minus CL", "observed_minus|CL"), ("People minus sample until return", "observed_minus|SR"),
            ("Growth, misses against CRRA utility", "growth|observed_crra"),
            ("Bits per choice, CL", "bits|CL"), ("Bits per choice, sample until return", "bits|SR")]
    rows += [(f"Miss rate, {n} options", f"miss_rate|n={n}") for n in range(2, 7)]
    for label, k in rows:
        L.append(f"| {label} | " + " | ".join(_f(s[f'{c}|{k}']) for c in COSTS) + " |")
    L += ["", f"H1w, high minus low cost growth: {_f(s['H1w'])}. Supported: {s['H1w_supported']}.",
          f"Medium minus low: {_f(s['H1w_medium_minus_low'])}. With CRRA utility (r = {s['crra_r_low_cost']:.3f}, "
          f"fitted at low cost): {_f(s['H1w_crra'])}.", "",
          "Brackets give 95% percentile intervals from resampling choices within menu and cost. A star marks a value "
          "whose percentile and basic intervals both exclude zero."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/h1_within"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--simulate", choices=("fixed", "falling"))
    a = ap.parse_args()
    s = run(a.data, a.out, a.replicates, a.simulate)
    print((a.out / "table_h1_within.md").read_text())
    print("seconds", s["provenance"]["seconds"])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
