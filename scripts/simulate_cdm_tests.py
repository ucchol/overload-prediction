"""Design check for Amendment 7. Can the tests tell a search that stops on its own clock from one that stops at a
good enough option, on the displays people actually saw?

Only the displays of the included rounds are used (every option's value and place, the set size, the complexity and
which participant saw it). Choices, selections and times are simulated, never read, under four processes.

- first_return_uniform. Options are read at random, with replacement, since a value is unknown until worked out. A
  read option that seems better than the current selection is selected. The search ends at the first return to an
  option already read, the stopping rule of sample until return.
- first_return_value. The same, but reading is drawn to valuable options with probability softmax(0.15 v), the
  attention of sample until return.
- satisficing_in_order. Options are read from the top of the list. The search ends once the selection seems worth at
  least a reservation amount, or at the end of the list.
- satisficing_random. The same, read in a random order.

Working out an option takes a lognormal time (median 2 s with 3 operations, 4 s with 7) and is noisy (standard
deviation 1 and 3 dollars). Reservation amounts are those Caplin, Dean and Martin reported for Experiment 2. Every
test of the amendment is run on each simulated data set, with a small participant bootstrap for its intervals.

Usage
    python -I scripts/simulate_cdm_tests.py data/cdm --boot 100 --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from overload.cdm import CDMData, load_cdm  # noqa: E402
from overload.experiments import stopping_cdm as sc  # noqa: E402

TAU = {3: 2.0, 7: 4.0}
NOISE = {3: 1.0, 7: 3.0}
RESERVATION = {(10, 3): 9.54, (20, 3): 11.18, (40, 3): 15.54, (10, 7): 6.36, (20, 7): 9.95, (40, 7): 10.84}
PROCESSES = ("first_return_uniform", "first_return_value", "satisficing_in_order", "satisficing_random")


def simulate_round(v: np.ndarray, c: int, process: str, rng: np.random.Generator):
    n = v.size
    seen = np.zeros(n, bool)
    perceived = v + rng.normal(0.0, NOISE[c], n)
    current = -np.inf
    clock, sels = 0.0, []
    if process.startswith("first_return"):
        q = np.ones(n) / n if process.endswith("uniform") else np.exp(0.15 * (v - v.max()))
        q = q / q.sum()
        while True:
            i = int(rng.choice(n, p=q))
            if seen[i]:
                break
            seen[i] = True
            clock += TAU[c] * np.exp(rng.normal(0.0, 0.4))
            if perceived[i] > current:
                current = perceived[i]
                sels.append((i, max(1.0, np.ceil(clock))))
    else:
        order = np.arange(n) if process.endswith("in_order") else rng.permutation(n)
        for i in order:
            clock += TAU[c] * np.exp(rng.normal(0.0, 0.4))
            if perceived[i] > current:
                current = perceived[i]
                sels.append((i, max(1.0, np.ceil(clock))))
            if current >= RESERVATION[(n, c)]:
                break
    return sels


def simulate(real: CDMData, process: str, seed: int) -> CDMData:
    rng = np.random.default_rng(seed)
    s_round, s_value, s_time, s_pos, s_better, s_stop, s_first = [], [], [], [], [], [], []
    choice, final_time = np.zeros(real.rounds, int), np.zeros(real.rounds)
    for r in range(real.rounds):
        v = real.values[r, : real.n[r]]
        sels = simulate_round(v, int(real.complexity[r]), process, rng)
        for k, (i, t) in enumerate(sels):
            s_round.append(r)
            s_value.append(v[i])
            s_time.append(t)
            s_pos.append(i)
            s_better.append(int((v > v[i]).sum()))
            s_stop.append(k == len(sels) - 1)
            s_first.append(k == 0)
        choice[r], final_time[r] = sels[-1][0], sels[-1][1]
    return CDMData(
        experiment=real.experiment,
        n=real.n,
        complexity=real.complexity,
        participant=real.participant,
        participants=real.participants,
        values=real.values,
        choice=choice,
        final_time=final_time,
        sel_round=np.array(s_round),
        sel_value=np.array(s_value, float),
        sel_time=np.array(s_time, float),
        sel_position=np.array(s_pos),
        sel_better=np.array(s_better),
        sel_stop=np.array(s_stop, bool),
        sel_first=np.array(s_first, bool),
    )


def _f(e: dict, d: int = 4) -> str:
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{' *' if e['clear'] else ''}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cdm", type=Path)
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    real = load_cdm(args.cdm)
    for k, process in enumerate(PROCESSES):
        sc._STATE.update(data=simulate(real, process, args.seed + 1000 * k))
        point = sc.run_replicate(-1)
        reps = [sc.run_replicate(b) for b in range(args.boot)]
        S = sc.summarize_all(point, reps)
        s = S["pooled"]
        print(f"\n{process}")
        print(f"  C1 value minus share, bits per selection   {_f(s['stop']['value_minus_share'])}  {s['C1']}")
        print(f"  C1 the same with log time in both rules     {_f(s['stop']['value_minus_share_with_time'])}")
        o = S["process"]["first"]
        print(
            f"  C2 first selection, relative position {o['observed']['point']:.3f} "
            f"(model {o['predicted']['point']:.3f}), minus one half {_f(o['minus_half'], 3)}, "
            f"minus model {_f(o['minus_predicted'], 3)}  {s['C2']}"
        )
        g = s["growth"]
        print(f"  growth, elasticity of final selection time {_f(g['observed'], 2)}, looks {_f(g['predicted'], 2)}")
        print(f"  C3 SR minus CL percent {_f(s['choices']['SR_minus_CL_percent'], 2)}  {s['C3']}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
