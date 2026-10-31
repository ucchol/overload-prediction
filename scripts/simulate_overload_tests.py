"""Design check for Amendment 8. On the real displays (values, set sizes and who saw them, never choices), simulate
choices from CL with a fixed sensitivity and from CL whose sensitivity rises with set size (compensation), and run
the frozen tests. O1 should call the first overload and the second compensation.

Usage
    python -I scripts/simulate_overload_tests.py --cdm data/cdm --reutskaja data/reutskaja --boot 30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from overload.cdm import softmax_masked  # noqa: E402
from overload.experiments import overload_premonition as op  # noqa: E402
from overload.overload_forecast import compensation  # noqa: E402

LAMBDA = {"cdm": 0.35, "reutskaja": 0.9}  # sensitivities of the size a fit could give, per dollar or rating point


def simulate(panel, lam: float, beta_on: bool, rng: np.random.Generator):
    f = compensation(panel.n) if beta_on else np.ones(panel.n.size)
    q = softmax_masked(lam * f[:, None] * panel.values)
    u = rng.random(panel.n.size)
    choice = (np.cumsum(q, axis=1) < u[:, None]).sum(axis=1)
    return replace(panel, choice=np.minimum(choice, panel.n - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdm", type=Path)
    ap.add_argument("--reutskaja", type=Path)
    ap.add_argument("--boot", type=int, default=30)
    a = ap.parse_args()
    real = op.groups({"cdm": a.cdm, "reutskaja": a.reutskaja})
    for process, beta_on in (("fixed sensitivity", False), ("compensation", True)):
        rng = np.random.default_rng(5)
        G = {
            g: ([simulate(p, LAMBDA["cdm" if g.startswith("cdm") else g], beta_on, rng) for p in panels], P)
            for g, (panels, P) in real.items()
        }
        op._STATE.update(groups=G, datasets=op.datasets_of(G))
        point = op.run_replicate(-1)
        reps = [op.run_replicate(b) for b in range(a.boot)]
        S = op.summarize_all(point, reps)
        for ds, B in S["datasets"].items():
            d, o = B["observed_minus_comp"], B["growth"]["observed"]
            print(
                f"{process:18s} {ds:9s} growth {o['point']:+.3f}  people minus compensation {d['point']:+.4f} "
                f"[{d['percentile'][0]:+.4f}, {d['percentile'][1]:+.4f}]  O1 {B['O1']['verdict']}  "
                f"SR AUC {B['auc']['SR']['point']:.3f}  O2 {B['O2']['supported']}"
            )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
