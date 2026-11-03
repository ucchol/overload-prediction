"""Simulation checks for the process models of src/overload/time_models.py (docs/time_models_plan.md).

For each process model, times and choices are simulated from known parameters over the real displays of one set
(values only, no recorded choice or time is read), with a speed factor per person. The model is then fitted to the
simulated data and checked in three ways.
1. Its expected log time per trial against the true expected log time (Monte Carlo from the true parameters),
   within person R2 and correlation.
2. The within person R2 it reaches on simulated held out people, against the R2 of the true expected log time
   (the best any law could do on that sample).
3. Its numerical expected log time against a Monte Carlo average under its own fitted parameters.

Usage, python3 -I scripts/time_models_sim.py --dev DEV_CSV --set snack --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

sys.path.insert(0, "src")
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from overload import time_models as tm  # noqa: E402


def sim_race(R: tm.Race, tr: tm.Trials, rng, reps: int = 1):
    """Simulate (choice, relative decision time) from a race with parameters R.theta, t0 in relative units R.t0abs."""
    th = R.theta
    v = R.drifts(tr.x, th)
    ok = tr.mask
    n = len(tr.n)
    T = np.full((reps, n), np.inf); C = np.zeros((reps, n), int)
    for r in range(reps):
        todo = np.ones(n, bool)
        while todo.any():
            idx = np.where(todo)[0]
            vv = v[idx]
            if R.kind in ("lba", "mlba"):
                A = np.exp(th[-3]); b = A + np.exp(th[-2])
                d = vv + rng.normal(0, 1, vv.shape)
                k0 = rng.uniform(0, A, vv.shape)
                t = np.where((d > 0) & ok[idx], (b - k0) / np.where(d > 0, d, 1), np.inf)
            else:
                b = np.exp(th[-2])
                mu = b / np.maximum(vv, 1e-9)
                t = np.where(ok[idx], rng.wald(np.where(ok[idx], mu, 1.0), b ** 2), np.inf)
            fin = np.isfinite(t.min(axis=1))
            T[r, idx[fin]] = t[fin].min(axis=1)
            C[r, idx[fin]] = t[fin].argmin(axis=1)
            todo[idx[fin]] = False
    return C, T + R.t0abs


def sim_sr(m: tm.SRTiming, tr: tm.Trials, rng, reps: int = 1):
    kap, sc = np.exp(m.theta[0]), np.exp(m.theta[1])
    n = len(tr.n)
    C = np.zeros((reps, n), int); T = np.zeros((reps, n))
    for i in range(n):
        q = softmax(m.lam * tr.x[i, : tr.n[i]])
        for r in range(reps):
            seen = set(); L = 0
            while True:
                j = rng.choice(q.size, p=q); L += 1
                if j in seen:
                    break
                seen.add(j)
            C[r, i] = j; T[r, i] = m.t0abs + rng.gamma(L * kap, sc)
    return C, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--set", default="snack")
    ap.add_argument("--max_people", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    rng = np.random.default_rng(777)
    S = load_all(a.dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
    T = tm.from_timeset(S[a.set], SCALE[a.set])
    keep = T.pid < a.max_people
    T = T.take(keep)
    P = int(T.pid.max()) + 1
    speed = np.exp(rng.normal(0, 0.4, P))
    truth = {
        "lba": tm.Race("lba", np.array([0.8, np.log(2.5), np.log(0.6), np.log(0.8), 0.0])),
        "mlba": tm.Race("mlba", np.array([1.0, np.log(3.0), np.log(2.0), np.log(0.6), np.log(0.8), 0.0])),
        "wald": tm.Race("wald", np.array([np.log(0.6), np.log(2.0), np.log(1.5), 0.0])),
        "rwald": tm.Race("rwald", np.array([np.log(2.5), np.log(8.0), np.log(1.5), 0.0])),
    }
    report = {}
    half = T.pid < P // 2
    for kind, R in truth.items():
        R.t0abs = 0.3
        C, Tm = sim_race(R, T, rng, reps=1)
        sim = tm.Trials(T.x, T.n, C[0], np.log(Tm[0] * speed[T.pid]), T.pid)
        # true expected log time per trial (Monte Carlo, 200 draws), person speed added
        _, Tmc = sim_race(R, T, rng, reps=200)
        true_e = np.log(Tmc).mean(axis=0) + np.log(speed[T.pid])
        fit = tm.Race(kind).fit(sim.take(half), starts=3)
        te = sim.take(~half)
        pred = fit.expected_logt(te)
        # own Monte Carlo under fitted parameters (relative units)
        fit.t0abs = fit.t0(fit.theta, fit.tmin)
        _, Tf = sim_race(fit, te, rng, reps=200)
        mc = np.log(Tf).mean(axis=0)
        rep = {
            "R2_fitted_vs_true_expected": tm.r2(true_e[~half], pred, te.pid),
            "R2_on_simulated_times_fitted": tm.r2(te.logt, pred, te.pid),
            "R2_on_simulated_times_true": tm.r2(te.logt, true_e[~half], te.pid),
            "R2_numeric_vs_montecarlo": tm.r2(mc, pred, te.pid),
            "theta_true": R.theta.tolist(), "theta_fit": fit.theta.tolist(),
        }
        report[kind] = rep
        print(kind, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}), flush=True)
    m = tm.SRTiming(lam=6.0, theta=np.array([np.log(1.5), np.log(0.25), 0.0])); m.t0abs = 0.3
    C, Tm = sim_sr(m, T, rng, reps=1)
    sim = tm.Trials(T.x, T.n, C[0], np.log(Tm[0] * speed[T.pid]), T.pid)
    _, Tmc = sim_sr(m, T, rng, reps=100)
    true_e = np.log(Tmc).mean(axis=0) + np.log(speed[T.pid])
    fit = tm.SRTiming().fit(sim.take(half))
    te = sim.take(~half)
    pred = fit.expected_logt(te)
    fit.t0abs = fit.tmin / (1 + np.exp(-fit.theta[2]))
    _, Tf = sim_sr(fit, te, rng, reps=100)
    rep = {"R2_fitted_vs_true_expected": tm.r2(true_e[~half], pred, te.pid),
           "R2_on_simulated_times_fitted": tm.r2(te.logt, pred, te.pid),
           "R2_on_simulated_times_true": tm.r2(te.logt, true_e[~half], te.pid),
           "R2_numeric_vs_montecarlo": tm.r2(np.log(Tf).mean(axis=0), pred, te.pid),
           "lambda_true": 6.0, "lambda_fit": fit.lam}
    report["sr_timing"] = rep
    print("sr_timing", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out / f"sim_{a.set}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
