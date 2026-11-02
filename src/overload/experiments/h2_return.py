"""H2 of the paper, the return rule tested on gaze sequences, ANALYSIS_PLAN.md Amendment 11.

Cues per trial from the order of gazes. Return marks the first item gazed at again, dwell is each item's share of
gaze time, last marks the last item gazed at. CL and CL plus one cue, softmax(lambda v + beta cue), each set size
predicted from fits to the others. Snack data (Thomas, Molter and Krajbich, 2021) and Reutskaja and colleagues (2011).

Usage
    python -I -m overload.experiments.h2_return --snack data/snack --snack-gaze data/snack_gaze \
        --reutskaja data/reutskaja --out results/h2_return
    add --simulate {return,dwell,none} to replace every choice by a simulated one (the gaze is kept)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from ..bootstrap import summarize
from ..engine import provenance, write_json

CUES = ("return", "dwell", "last")
DATASETS = ("snack", "reutskaja")
SEED_OFFSET = 110000


@dataclass
class Block:
    """Trials of one set size, padded to n items."""

    n: int
    V: np.ndarray  # trials by items, ratings
    choice: np.ndarray
    person: np.ndarray  # participant index
    cue: dict  # name -> trials by items
    any_return: np.ndarray


def cues_from_sequence(items: list[int], durs: list[float], n: int) -> tuple[dict, bool]:
    """Cues of one trial from its gazes (consecutive fixations on one item already merged)."""
    r, d, last = np.zeros(n), np.zeros(n), np.zeros(n)
    seen: set[int] = set()
    returned = False
    for i, t in zip(items, durs, strict=True):
        if i in seen and not returned:
            r[i] = 1.0
            returned = True
        seen.add(i)
        d[i] += t
    if items:
        last[items[-1]] = 1.0
    tot = d.sum()
    return {"return": r, "dwell": d / tot if tot > 0 else d, "last": last}, returned


def merge(items: list[int], durs: list[float]) -> tuple[list[int], list[float]]:
    out_i, out_d = [], []
    for i, t in zip(items, durs, strict=True):
        if out_i and out_i[-1] == i:
            out_d[-1] += t
        else:
            out_i.append(i)
            out_d.append(t)
    return out_i, out_d


def _blocks(rows: list[tuple]) -> list[Block]:
    """rows: (n, values, choice, person, cues, returned)."""
    out = []
    for n in sorted({r[0] for r in rows}):
        rs = [r for r in rows if r[0] == n]
        out.append(Block(
            n=n, V=np.array([r[1] for r in rs]), choice=np.array([r[2] for r in rs]),
            person=np.array([r[3] for r in rs]),
            cue={c: np.array([r[4][c] for r in rs]) for c in CUES},
            any_return=np.array([r[5] for r in rs]),
        ))
    return out


def load_snack_gaze(snack: Path, gaze: Path) -> tuple[list[Block], int]:
    rows, people = [], {}
    for n in (9, 16, 25, 36):
        agg = pd.read_csv(snack / f"setsize-{n}_desc-data.csv")
        g = pd.concat([pd.read_csv(p) for p in sorted(gaze.glob(f"*_setsize-{n}_desc-gazes.csv"))],
                      ignore_index=True)
        g = g.sort_values(["subject", "trial", "gaze_num"])
        seq = {k: (list(v["item"].astype(int)), list(v["dur"].astype(float)))
               for k, v in g.groupby(["subject", "trial"], sort=False)}
        for _, t in agg.iterrows():
            V = t[[f"item_value_{i}" for i in range(n)]].to_numpy(float)
            c = t["choice"]
            if not np.all(np.isfinite(V)) or not np.isfinite(c):
                continue
            items, durs = seq.get((t["subject"], t["trial"]), ([], []))
            items, durs = merge(items, durs)
            cue, ret = cues_from_sequence(items, durs, n)
            p = people.setdefault(int(t["subject"]), len(people))
            rows.append((n, V, int(c), p, cue, ret))
    return _blocks(rows), len(people)


def load_reutskaja_gaze(folder: Path) -> tuple[list[Block], int]:
    df = pd.read_stata(folder / "fixations_data.dta").sort_values(["subject", "trialid", "fixOrder"])
    rows, people = [], {}
    for (s, _), t in df.groupby(["subject", "trialid"], sort=False):
        n = int(t["numalt"].iloc[0])
        V = t[[f"rating{i}" for i in range(1, n + 1)]].iloc[0].to_numpy(float)
        c = int(t["chosendestring"].iloc[0]) - 1
        items, durs = merge(list(t["Roinum"].astype(int) - 1), list(t["FixDuration"].astype(float)))
        cue, ret = cues_from_sequence(items, durs, n)
        p = people.setdefault(int(s), len(people))
        rows.append((n, V, c, p, cue, ret))
    return _blocks(rows), len(people)


# ----------------------------------------------------------------------------- fitting
def _nll(theta, blocks, w, cue):
    lam, beta = theta
    f, g = 0.0, np.zeros(2)
    for b, wb in zip(blocks, w, strict=True):
        Z = b.cue[cue] if cue else np.zeros_like(b.V)
        U = lam * b.V + beta * Z
        lse = logsumexp(U, axis=1)
        idx = np.arange(b.choice.size)
        f -= float(wb @ (U[idx, b.choice] - lse))
        Q = softmax(U, axis=1)
        g[0] -= float(wb @ (b.V[idx, b.choice] - (Q * b.V).sum(1)))
        g[1] -= float(wb @ (Z[idx, b.choice] - (Q * Z).sum(1)))
    return f, g


def fit(blocks, w, cue):
    starts = [(0.5, 0.0)] if cue is None else [(0.5, 0.0), (0.5, 2.0), (0.1, -1.0)]
    best = None
    for s in starts:
        if cue is None:

            def f1(x):
                f, g = _nll([x[0], 0.0], blocks, w, None)
                return f, g[:1]

            res = minimize(f1, [s[0]], jac=True, method="L-BFGS-B", bounds=[(0, 1e3)],
                           options={"ftol": 1e-14, "gtol": 1e-9})
            x = np.array([res.x[0], 0.0])
        else:
            res = minimize(_nll, np.array(s), args=(blocks, w, cue), jac=True, method="L-BFGS-B",
                           bounds=[(0, 1e3), (-1e3, 1e3)], options={"ftol": 1e-14, "gtol": 1e-9})
            x = res.x
        if best is None or res.fun < best[0]:
            best = (res.fun, x)
    return best[1]


def held_out_bits(blocks, w, cue) -> float:
    """Mean log2 probability of the chosen item, each set size predicted from fits to the others."""
    tot, N = 0.0, 0.0
    for k, b in enumerate(blocks):
        others = [x for j, x in enumerate(blocks) if j != k]
        wo = [x for j, x in enumerate(w) if j != k]
        th = fit(others, wo, cue)
        f, _ = _nll(th, [b], [w[k]], cue)
        tot -= f / np.log(2)
        N += w[k].sum()
    return tot / N


def measures(data: dict, draws: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in DATASETS:
        blocks, _ = data[name]
        cnt = draws[name]
        w = [cnt[b.person].astype(float) for b in blocks]
        base = held_out_bits(blocks, w, None)
        out[f"{name}|bits|CL"] = base
        for c in CUES:
            bits = held_out_bits(blocks, w, c)
            out[f"{name}|bits|CL+{c}"] = bits
            out[f"{name}|gain|{c}"] = bits - base
            out[f"{name}|beta|{c}"] = float(fit(blocks, w, c)[1])
        out[f"{name}|return_minus_dwell"] = out[f"{name}|gain|return"] - out[f"{name}|gain|dwell"]
        for b, wb in zip(blocks, w, strict=True):
            idx = np.arange(b.choice.size)
            m = b.any_return & (wb > 0)
            chose_ret = b.cue["return"][idx, b.choice] == 1.0
            out[f"{name}|share_return_chosen|n={b.n}"] = float(wb[m] @ chose_ret[m] / wb[m].sum()) if m.any() else np.nan
            out[f"{name}|share_no_return|n={b.n}"] = float(wb @ (~b.any_return) / wb.sum())
    return out


def draw_counts(n_people: dict, b: int) -> dict[str, np.ndarray]:
    if b < 0:
        return {k: np.ones(v) for k, v in n_people.items()}
    rng = np.random.default_rng(SEED_OFFSET + b)
    out = {}
    for k in DATASETS:
        P = n_people[k]
        out[k] = np.bincount(rng.integers(0, P, size=P), minlength=P)
    return out


def simulate(data: dict, kind: str, seed: int = 11) -> dict:
    """Replace every choice by a draw from a known model, keeping the gaze. 'return' chooses the first returned
    item on 60 percent of trials with a return, otherwise by CL. 'dwell' chooses by value plus dwell. 'none' by CL."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (blocks, P) in data.items():
        nb = []
        for b in blocks:
            ch = []
            for t in range(b.choice.size):
                v = b.V[t]
                q = softmax(0.8 * v)
                if kind == "return" and b.any_return[t] and rng.random() < 0.6:
                    ch.append(int(np.argmax(b.cue["return"][t])))
                elif kind == "dwell":
                    ch.append(int(rng.choice(v.size, p=softmax(0.8 * v + 4.0 * b.cue["dwell"][t]))))
                else:
                    ch.append(int(rng.choice(v.size, p=q)))
            nb.append(Block(b.n, b.V, np.array(ch), b.person, b.cue, b.any_return))
        out[name] = (nb, P)
    return out


def run(a) -> dict:
    t0 = time.time()
    data = {"snack": load_snack_gaze(a.snack, a.snack_gaze), "reutskaja": load_reutskaja_gaze(a.reutskaja)}
    if a.simulate:
        data = simulate(data, a.simulate)
    n_people = {k: v[1] for k, v in data.items()}
    point = measures(data, draw_counts(n_people, -1))
    out = a.out
    part = out / "partial"
    part.mkdir(parents=True, exist_ok=True)
    for start in range(0, a.replicates, 10):
        f = part / f"{start:05d}.jsonl"
        if f.exists():
            continue
        rs = [{"b": b, **measures(data, draw_counts(n_people, b))} for b in range(start, min(start + 10, a.replicates))]
        tmp = part / f"{start:05d}.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rs))
        tmp.rename(f)
    reps = [json.loads(x) for s in range(0, a.replicates, 10) for x in (part / f"{s:05d}.jsonl").read_text().splitlines()]
    summary = {k: summarize(v, [r[k] for r in reps]) if reps and np.isfinite(v) else {"point": v}
               for k, v in point.items()}
    sup = {}
    for name in DATASETS:
        g, d = summary[f"{name}|gain|return"], summary[f"{name}|return_minus_dwell"]
        sup[name] = bool(g.get("clear") and g["point"] > 0 and d.get("clear") and d["point"] > 0)
    summary["H2_supported_by_dataset"] = sup
    summary["H2_supported"] = all(sup.values())
    summary["provenance"] = provenance({"replicates": a.replicates, "seconds": round(time.time() - t0, 1),
                                        "simulated": a.simulate})
    write_json(out / "point.json", point)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "summary.json", summary)
    (out / "table_h2_return.md").write_text(table(summary))
    import shutil

    shutil.rmtree(part)
    return summary


def _f(e, d=3):
    if "percentile" not in e:
        return f"{e['point']:+.{d}f}"
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" + (" *" if e["clear"] else "")


def table(s):
    L = ["# H2, the return rule on gaze sequences (Amendment 11)", "",
         "| Measure | Snack | Reutskaja |", "|---|---|---|"]
    keys = [("Bits per choice, CL", "bits|CL")]
    for c in CUES:
        keys += [(f"Gain from the {c} cue, bits per choice", f"gain|{c}"), (f"Weight of the {c} cue", f"beta|{c}")]
    keys += [("Return gain minus dwell gain", "return_minus_dwell")]
    for label, k in keys:
        L.append(f"| {label} | " + " | ".join(_f(s[f'{d}|{k}']) for d in DATASETS) + " |")
    L += ["", "| Set size | Snack, first returned item chosen | Snack, no return | Reutskaja, first returned item chosen "
          "| Reutskaja, no return |", "|---|---|---|---|---|"]
    for n in (4, 9, 16, 25, 36):
        cells = []
        for d in DATASETS:
            for k in ("share_return_chosen", "share_no_return"):
                e = s.get(f"{d}|{k}|n={n}")
                cells.append(f"{e['point']:.2f}" if e else "")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", f"H2 supported by dataset: {s['H2_supported_by_dataset']}. Overall (both required): {s['H2_supported']}.",
          "", "Brackets give 95% percentile intervals from the participant bootstrap. A star marks a value whose "
          "percentile and basic intervals both exclude zero."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snack", type=Path, required=True)
    ap.add_argument("--snack-gaze", type=Path, required=True)
    ap.add_argument("--reutskaja", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/h2_return"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--simulate", choices=("return", "dwell", "none"))
    a = ap.parse_args()
    s = run(a)
    print((a.out / "table_h2_return.md").read_text())
    print("seconds", s["provenance"]["seconds"])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
