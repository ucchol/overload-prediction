"""Confirmatory test A3b of H3, inputs over rules, ANALYSIS_PLAN.md Amendment 9.

Each rule of the completeness ladder (CL, DbS_avg, MLBA_avg) is extended by the independent filename cue,
softmax(lambda s + gamma z), and the held out completeness gained by the cue is set against the spread of
held out completeness across the rules, in file selection.

Usage
    python -I -m overload.experiments.h3_cue --data PATH_TO_CSV --out results/h3_cue --jobs 2
    python -I -m overload.experiments.h3_cue --simulate {rules,cue} --out results/h3_cue_sim_rules
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from ..bootstrap import resample_indices, summarize
from ..data import Task, load_study
from ..engine import provenance, write_json
from ..filename_cue import cue, load_vectors
from ..fit import fit_softmax, predict_softmax
from ..metrics import attainable, completeness, log_score
from ..models import score_cl, score_dbs_avg, score_mlba_avg

RULES = {"CL": score_cl, "DbS_avg": score_dbs_avg, "MLBA_avg": score_mlba_avg}
PRIMARY_RULE = "CL"
SUBSETS = ("all", "low", "high")
BOUND = 1e6
_STATE: dict = {}


# ----------------------------------------------------------------------------- two parameter fit
def _nll_grad(theta: np.ndarray, S: list[np.ndarray], Z: list[np.ndarray], C: list[np.ndarray]):
    lam, gam = theta
    f, g = 0.0, np.zeros(2)
    for s, z, c in zip(S, Z, C, strict=True):
        u = lam * s + gam * z
        N = c.sum()
        f += N * logsumexp(u) - float(c @ u)
        q = softmax(u)
        g[0] += N * float(q @ s) - float(c @ s)
        g[1] += N * float(q @ z) - float(c @ z)
    return f, g


def fit_cue(S: list[np.ndarray], Z: list[np.ndarray], C: list[np.ndarray]) -> tuple[float, float]:
    """Maximum likelihood (lambda >= 0, gamma) of softmax(lambda s + gamma z). The likelihood is concave."""
    keep = [(s, z, c) for s, z, c in zip(S, Z, C, strict=True) if c.sum() > 0]
    S, Z, C = [k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep]
    lam0 = fit_softmax(S, C)
    best = None
    for start in ((lam0, 0.0), (0.0, 0.0), (lam0, 10.0), (lam0, -10.0)):
        res = minimize(
            _nll_grad, np.array(start), args=(S, Z, C), jac=True, method="L-BFGS-B",
            bounds=[(0.0, BOUND), (-BOUND, BOUND)], options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000},
        )
        if best is None or res.fun < best.fun:
            best = res
    return float(best.x[0]), float(best.x[1])


# ----------------------------------------------------------------------------- one replicate
def scores(task: Task) -> dict[str, list[np.ndarray]]:
    return {m: [np.asarray(f(t, task.shares), dtype=float) for t in task.trials] for m, f in RULES.items()}


def cues(task: Task) -> list[np.ndarray]:
    vec = load_vectors()
    return [cue(t.options, vec) for t in task.trials]


def replicate(counts: list[np.ndarray], S: dict, Z: list[np.ndarray]) -> dict:
    """Held out log scores of every rule with and without the cue, and the pooled cue weights."""
    T = len(counts)
    rec: dict = {"Lstar": [attainable(c) for c in counts]}
    for m, s in S.items():
        ho, ho_cue = [], []
        for t in range(T):
            o = [u for u in range(T) if u != t]
            lam = fit_softmax([s[u] for u in o], [counts[u] for u in o])
            ho.append(log_score(counts[t], predict_softmax(lam, s[t])))
            lc, gc = fit_cue([s[u] for u in o], [Z[u] for u in o], [counts[u] for u in o])
            ho_cue.append(log_score(counts[t], softmax(lc * s[t] + gc * Z[t])))
        rec[m] = ho
        rec[m + "+cue"] = ho_cue
        rec[m + "+cue|theta"] = list(fit_cue(s, Z, counts))
    return rec


def measures(rec: dict, trials: dict[str, list[int]], bench: Sequence[float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sub, idx in trials.items():
        K = {k: completeness([rec[k][t] for t in idx], [bench[t] for t in idx])
             for k in [*RULES, *(m + "+cue" for m in RULES)]}
        spread = max(K[m] for m in RULES) - min(K[m] for m in RULES)
        out[f"{sub}|spread"] = spread
        for m in RULES:
            out[f"{sub}|K|{m}"] = K[m]
            out[f"{sub}|K|{m}+cue"] = K[m + "+cue"]
            out[f"{sub}|gain|{m}"] = K[m + "+cue"] - K[m]
            out[f"{sub}|A3b|{m}"] = K[m + "+cue"] - K[m] - spread
    for m in RULES:
        out[f"gamma|{m}"] = rec[m + "+cue|theta"][1]
    return out


def _worker(bs: Sequence[int]) -> list[dict]:
    task, tasks, S, Z = _STATE["task"], _STATE["tasks"], _STATE["S"], _STATE["Z"]
    out = []
    for b in bs:
        idx = resample_indices(tasks, b)["file"]
        r = replicate(task.counts(idx), S, Z)
        r["b"] = b
        out.append(r)
    return out


# ----------------------------------------------------------------------------- simulation
def simulated_task(task: Task, S: dict, Z: list[np.ndarray], truth: str, seed: int = 9) -> Task:
    """The file task with every recorded choice replaced by a draw from a known model, CL ('rules') or
    CL with the cue ('cue'), at sensitivities fixed in advance. Only which trials each respondent answered
    is kept from the data."""
    rng = np.random.default_rng(seed)
    probs = []
    for s, z in zip(S["CL"], Z, strict=True):
        u = (s - s.mean()) / max(np.std(s), 1e-9)
        if truth == "cue":
            u = u + 1.5 * (z - z.mean()) / max(np.std(z), 1e-9)
        probs.append(softmax(u))
    panel = [[(j, int(rng.choice(len(probs[j]), p=probs[j]))) for j, _ in row] for row in task.panel]
    return Task(task.name, task.attributes, task.trials, task.shares, task.share_r2, panel)


# ----------------------------------------------------------------------------- run
def run(csv: Path, out: Path, replicates: int, jobs: int, simulate: str | None = None) -> dict:
    tasks = load_study(csv)
    task = tasks["file"]
    S, Z = scores(task), cues(task)
    if simulate:
        task = simulated_task(task, S, Z, simulate)
        tasks = {**tasks, "file": task}
    _STATE.update(task=task, tasks=tasks, S=S, Z=Z)
    trials = {"all": list(range(len(task.trials))), "low": task.trials_at("low"), "high": task.trials_at("high")}

    t0 = time.time()
    point = replicate(task.counts(), S, Z)
    order = list(range(replicates))
    # Finished chunks are saved and skipped on a restart. Every replicate has its own seed.
    out.mkdir(parents=True, exist_ok=True)
    part = out / "partial"
    part.mkdir(exist_ok=True)
    groups = [order[i : i + 10] for i in range(0, len(order), 10)]
    todo = [g for g in groups if not (part / f"{g[0]:05d}.jsonl").exists()]

    def save(g: list[int], res: list[dict]) -> None:
        tmp = part / f"{g[0]:05d}.tmp"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in res))
        tmp.rename(part / f"{g[0]:05d}.jsonl")

    if jobs > 1 and todo:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for g, res in zip(todo, pool.map(_worker, todo), strict=True):
                save(g, res)
    else:
        for g in todo:
            save(g, _worker(g))
    reps = [json.loads(x) for g in groups for x in (part / f"{g[0]:05d}.jsonl").read_text().splitlines()]
    reps.sort(key=lambda r: r["b"])

    bench = point["Lstar"]  # full sample benchmark in every replicate (plan, Amendment 1)
    pm = measures(point, trials, bench)
    bm = [measures(r, trials, bench) for r in reps]
    summary = {k: summarize(v, [m[k] for m in bm]) if reps else {"point": v} for k, v in pm.items()}
    primary = summary[f"all|A3b|{PRIMARY_RULE}"]
    summary["A3b_supported"] = bool(primary.get("clear") and primary["point"] > 0)
    summary["cue"] = {t.key: dict(zip(t.options, map(float, z))) for t, z in zip(task.trials, Z, strict=True)}
    summary["provenance"] = provenance({"replicates": replicates, "seconds": round(time.time() - t0, 1),
                                        "simulated": simulate})
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "point.json", point)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "summary.json", summary)
    write_table(summary, out)
    import shutil

    shutil.rmtree(part)
    return summary


def _fmt(e: dict) -> str:
    if "percentile" not in e:
        return f"{e['point']:+.1f}"
    lo, hi = e["percentile"]
    return f"{e['point']:+.1f} [{lo:+.1f}, {hi:+.1f}]" + (" *" if e["clear"] else "")


def write_table(s: dict, out: Path) -> None:
    lines = ["# H3, test A3b (Amendment 9), file selection", "",
             "| Measure | All trials | Low information | High information |", "|---|---|---|---|"]
    keys = [("Spread of held out completeness across rules, points", "spread")]
    for m in RULES:
        keys += [(f"Held out completeness, {m}, %", f"K|{m}"), (f"Held out completeness, {m} with cue, %", f"K|{m}+cue"),
                 (f"Gain from the cue, {m}, points", f"gain|{m}"), (f"A3b, gain minus spread, {m}", f"A3b|{m}")]
    for label, k in keys:
        lines.append(f"| {label} | " + " | ".join(_fmt(s[f'{sub}|{k}']) for sub in SUBSETS) + " |")
    lines += ["", "| Cue weight gamma, pooled fit | Value |", "|---|---|"]
    lines += [f"| {m} | {_fmt(s[f'gamma|{m}'])} |" for m in RULES]
    lines += ["", f"A3b supported (CL, all trials, clearly positive): {s['A3b_supported']}", "",
              "Brackets give 95% percentile intervals from the participant bootstrap. A star marks a value whose "
              "percentile and basic intervals both exclude zero."]
    (out / "table_h3_cue.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/h3_cue"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--simulate", choices=("rules", "cue"))
    a = ap.parse_args()
    s = run(a.data, a.out, a.replicates, a.jobs, a.simulate)
    print((a.out / "table_h3_cue.md").read_text())
    print("seconds", s["provenance"]["seconds"])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
