"""Two look satisficing on the snack data (VERIFY_BRIEF_TWO_LOOK.md): point estimates, bootstrap, summaries.

Usage:
  python -I run_two_look.py --data DIR --out FILE [--B 1000]       (synthetic data)
  python -I run_two_look.py --real --out results.json [--B 1000]   (the real snack data)
Bootstrap arrays (no counts) go next to the output as boot_<name>.npz, fit diagnostics as diag_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import two_look_core as tl  # noqa: E402
from two_look_core import sc, srch  # noqa: E402

srch.assert_local(tl, srch, sc)

E1_MODELS = ("CL", "SR", "TLS")
CORR_KEYS = ("looks_TLS", "looks_SR", "top_rating")
SEED_BASE = 40000
LOGN = srch.LOGN


class Pre:
    """Everything that does not depend on the participant weights."""

    def __init__(self, data: dict):
        self.data = data
        self.T = len(data["choice"])
        self.C, self.n = data["C_all"], data["n"]
        self.size_idx, self.pidx, self.P = data["size_idx"], data["pidx"], data["P"]
        self.held_idx = [np.flatnonzero(self.size_idx == k) for k in range(4)]
        self.train_mask = [self.size_idx != k for k in range(4)]
        self.sr = srch.SRFitter(self.C, data["lc"])
        self.timed, self.log_rt = data["timed"], data["log_rt"]
        self.groups = srch.Groups(data)
        self.top_corr = self.groups.correlations(np.asarray(data["top_rating"], float)[self.groups.trials])
        self.n_looked = data["n_looked"]


def time_fold(eff, pre, k, w):
    """Time law of fold k for one effort: parameters and the held-out error."""
    tr = pre.timed & pre.train_mask[k] & (w > 0)
    ho = pre.timed & (pre.size_idx == k) & (w > 0)
    t0, tau, _ = srch.fit_time_law(eff[tr], pre.log_rt[tr], w[tr])
    pred = srch.time_law_predict(t0, tau, eff[ho])
    return t0, tau, srch.wmean(np.abs(pre.log_rt[ho] - pred), w[ho])


def evaluate(pre: Pre, wp: np.ndarray, with_uniform: bool = False) -> dict:
    w = wp[pre.pidx]
    T, C, n = pre.T, pre.C, pre.n
    lam = np.zeros(4)
    diag = np.zeros(3, int)
    E_sr_ho, E_tls_ho, D_tls_ho = np.zeros(T), np.zeros(T), np.zeros(T)
    err = np.zeros((len(E1_MODELS), 4))
    params = np.zeros((len(E1_MODELS), 4, 2))
    if with_uniform:
        E_u_ho, D_u_ho, err_u = np.zeros(T), np.zeros(T), np.zeros(4)
    for k in range(4):
        lam[k], info = pre.sr.fit(np.where(pre.train_mask[k], w, 0.0))  # SR sensitivity = TLS sensitivity
        diag += [info["n_local"] > 1, not info["refined"], lam[k] == 0]
        h = pre.held_idx[k]
        E_sr = srch.sr_expected_looks(lam[k], C)  # every trial of every set size with fold k's lambda
        logq, logh = tl.tls_log_qh(lam[k], C)
        E_tls, _ = tl.two_look_expectations(logq, logh, C, need_distinct=False)
        _, D_tls_ho[h] = tl.two_look_expectations(logq[h], logh[h], C[h])
        E_sr_ho[h], E_tls_ho[h] = E_sr[h], E_tls[h]
        for i, eff in enumerate((n, E_sr, E_tls)):
            t0, tau, e = time_fold(eff, pre, k, w)
            params[i, k] = (t0, tau)
            err[i, k] = e
        if with_uniform:
            lq, lh = tl.uniform_log_qh(lam[k], C, n)
            E_u, _ = tl.two_look_expectations(lq, lh, C, need_distinct=False)
            _, D_u_ho[h] = tl.two_look_expectations(lq[h], lh[h], C[h])
            E_u_ho[h] = E_u[h]
            err_u[k] = time_fold(E_u, pre, k, w)[2]
    g = pre.groups
    gt = g.trials
    corr = np.array([g.participant_mean(g.correlations(E_tls_ho[gt]), wp),
                     g.participant_mean(g.correlations(E_sr_ho[gt]), wp),
                     g.participant_mean(pre.top_corr, wp)])
    H = pre.held_idx
    obs = np.array([srch.wmean(pre.n_looked[H[k]], w[H[k]]) for k in range(4)])
    tls = np.array([srch.wmean(D_tls_ho[H[k]], w[H[k]]) for k in range(4)])
    srd = np.array([srch.wmean(E_sr_ho[H[k]] - 1.0, w[H[k]]) for k in range(4)])
    out = dict(lam=lam, diag=diag, err=err, params=params, corr=corr, obs=obs, tls=tls, srd=srd,
               e_obs=float(sc.ls_slope(LOGN, np.log(obs))), e_tls=float(sc.ls_slope(LOGN, np.log(tls))))
    if with_uniform:
        du = np.array([srch.wmean(D_u_ho[H[k]], w[H[k]]) for k in range(4)])
        out["uniform"] = {"error": float(err_u.mean()), "correlation": g.participant_mean(g.correlations(E_u_ho[gt]), wp),
                          "distinct": [float(x) for x in du], "elasticity": float(sc.ls_slope(LOGN, np.log(du)))}
    return out


def summarize(pt: dict, bs: dict) -> dict:
    S = srch.summary_S
    res = {"lambda": [float(x) for x in pt["lam"]]}
    E, Eb = pt["err"].mean(axis=1), bs["err"].mean(axis=2)
    error = {}
    for i, m in enumerate(E1_MODELS):
        s = S(E[i], Eb[:, i])
        s["by_size"] = [float(x) for x in pt["err"][i]]
        s["params"] = [[float(a), float(b)] for a, b in pt["params"][i]]
        error[m] = s
    res["E1"] = {"error": error, "TLS_minus": {"CL": S(E[2] - E[0], Eb[:, 2] - Eb[:, 0]), "SR": S(E[2] - E[1], Eb[:, 2] - Eb[:, 1])}}
    c, cb = pt["corr"], bs["corr"]
    res["E2"] = {"correlation": {k: S(c[i], cb[:, i]) for i, k in enumerate(CORR_KEYS)},
                 "TLS_minus_SR": S(c[0] - c[1], cb[:, 0] - cb[:, 1]),
                 "TLS_plus_top_rating": S(c[0] + c[2], cb[:, 0] + cb[:, 2])}
    olo, ohi = np.percentile(bs["obs"], [2.5, 97.5], axis=0)
    plo, phi = np.percentile(bs["tls"], [2.5, 97.5], axis=0)
    res["E3"] = {"observed": [float(x) for x in pt["obs"]], "predicted": [float(x) for x in pt["tls"]],
                 "predicted_SR": [float(x) for x in pt["srd"]],
                 "observed_over_predicted": [float(a / b) for a, b in zip(pt["obs"], pt["tls"])],
                 "observed_interval": [[float(x) for x in olo], [float(x) for x in ohi]],
                 "predicted_interval": [[float(x) for x in plo], [float(x) for x in phi]],
                 "observed_elasticity": S(pt["e_obs"], bs["e_obs"]), "predicted_elasticity": S(pt["e_tls"], bs["e_tls"]),
                 "predicted_minus_observed_elasticity": S(pt["e_tls"] - pt["e_obs"], bs["e_tls"] - bs["e_obs"])}
    res["uniform_attention"] = pt["uniform"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--real", action="store_true", help="analyse the real snack data")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir = srch.REAL_DATA if args.real else args.data
    if data_dir is None:
        raise SystemExit("give --data for test data, or --real for the real snack data")
    t0 = time.time()
    data = srch.load_search_data(data_dir, allow_real=args.real)
    pre = Pre(data)
    print(f"precomputation done in {time.time() - t0:.1f}s", flush=True)
    pt = evaluate(pre, np.ones(pre.P), with_uniform=True)
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    B = args.B
    keys = ("lam", "diag", "err", "corr", "obs", "tls", "srd", "e_obs", "e_tls")
    bs = {k: np.zeros((B,) + np.shape(pt[k])) for k in keys}
    for b in range(B):
        draw = np.random.default_rng(SEED_BASE + b).integers(0, pre.P, size=pre.P)
        ev = evaluate(pre, np.bincount(draw, minlength=pre.P).astype(float))
        for k in keys:
            bs[k][b] = ev[k]
        if (b + 1) % 25 == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    res = summarize(pt, bs)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    outdir = os.path.dirname(os.path.abspath(args.out))
    np.savez_compressed(os.path.join(outdir, f"boot_{stem}.npz"), **bs)
    diag = {"replicate_SR_folds_with_several_grid_maxima": int(bs["diag"][:, 0].sum()),
            "replicate_SR_folds_not_refined": int(bs["diag"][:, 1].sum()),
            "replicate_SR_folds_with_lambda_zero": int(bs["diag"][:, 2].sum()),
            "original_SR_diag": [int(x) for x in pt["diag"]],
            "replicate_lambda_range": [float(bs["lam"].min()), float(bs["lam"].max())],
            "seconds": round(time.time() - t0, 1), "B": B}
    with open(os.path.join(outdir, f"diag_{stem}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
