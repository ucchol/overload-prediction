"""Amendment 5 analysis (VERIFY_BRIEF_SEARCH.md): point estimates, participant bootstrap, summaries.

Usage:
  python -I run_search.py --data DIR --out FILE [--B 1000]            (synthetic or test data)
  python -I run_search.py --real --out results.json [--B 1000]         (the real snack data; only when told)
The real data folder (verify/snack) is refused unless --real is given.
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
import search_core as srch  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, sc)

MODELS = srch.MODELS
STATIC = srch.STATIC
LOGN = srch.LOGN
CORR_KEYS = ("looks_SR", "entropy_CL", "entropy_DbS_avg", "entropy_MLBA_avg", "top_rating", "rating_sd", "top_count")
FIXED_FEATURES = ("top_rating", "rating_sd", "top_count")
SEED_BASE = 40000


# ============================================================================ weight-free precomputation
class Pre:
    def __init__(self, data: dict):
        self.data = data
        self.T = len(data["choice"])
        self.C, self.lc, self.n = data["C_all"], data["lc"], data["n"]
        self.size_idx, self.pidx, self.P = data["size_idx"], data["pidx"], data["P"]
        self.top = data["top_level"]
        self.y_top = data["y_top"]
        self.held_idx = [np.flatnonzero(self.size_idx == k) for k in range(4)]
        self.train_mask = [self.size_idx != k for k in range(4)]
        self.train, self.held = {}, {}
        for fam in STATIC:
            P = sc.Problem(data, fam, "all", np.arange(self.T))
            self.train[fam] = [P.subset(np.flatnonzero(self.size_idx != k)) for k in range(4)]
            self.held[fam] = [P.subset(self.held_idx[k]) for k in range(4)]
        self.sr = srch.SRFitter(self.C, self.lc)
        self.timed = data["timed"]
        self.log_rt = data["log_rt"]
        self.groups = srch.Groups(data)
        gt = self.groups.trials
        self.fixed_corr = {f: self.groups.correlations(np.asarray(data[f], float)[gt]) for f in FIXED_FEATURES}
        self.effort_fixed = {"CL": self.n, "DbS_avg": self.n * (self.n - 1), "MLBA_avg": self.n * (self.n - 1)}


def bernoulli_score(y, p, w):
    """Weighted mean of y log2 p + (1 - y) log2(1 - p), p clipped to [1e-15, 1 - 1e-15]."""
    p = np.clip(p, srch.P_CLIP, 1 - srch.P_CLIP)
    return srch.wmean(y * np.log2(p) + (1 - y) * np.log2(1 - p), w)


# ============================================================================ one sample
def evaluate(pre: Pre, wp: np.ndarray, init: dict | None = None) -> dict:
    """Every fit and statistic for participant weights wp (1 for every participant in the original sample)."""
    w = wp[pre.pidx]
    T, C, lc, n = pre.T, pre.C, pre.lc, pre.n
    r = np.arange(T)
    lam = {m: np.zeros(4) for m in MODELS}
    Q = {m: np.zeros((T, 7)) for m in MODELS}
    # ---------------------------------------------------------------- fits, leave one set size out
    for fam in STATIC:
        for k in range(4):
            Pt = pre.train[fam][k]
            u = w[Pt.trials][None]
            lam[fam][k] = sc.fit_lambda(Pt, np.ones_like(u), u, init=None if init is None else init[fam][k:k + 1])[0]
            Ph = pre.held[fam][k]
            Q[fam][Ph.trials] = srch.static_level_probs(Ph, lam[fam][k])
    E_ho = np.zeros(T)
    diag = np.zeros(3, int)  # folds with several grid maxima, unrefined optima, lambda = 0
    for k in range(4):
        lam["SR"][k], info = pre.sr.fit(np.where(pre.train_mask[k], w, 0.0))
        diag += [info["n_local"] > 1, not info["refined"], lam["SR"][k] == 0]
        h = pre.held_idx[k]
        Q["SR"][h], E_ho[h] = srch.sr_probs(lam["SR"][k], C[h])
    # ---------------------------------------------------------------- per trial held-out quantities
    bits = {m: np.log2(n * Q[m][r, lc]) for m in MODELS}
    ptop = {m: C[r, pre.top] * Q[m][r, pre.top] for m in MODELS}
    ent = {fam: srch.entropy_bits(C, Q[fam]) for fam in STATIC}
    H = pre.held_idx
    # ---------------------------------------------------------------- choices (L) and S5
    L_by = np.array([[srch.wmean(bits[m][H[k]], w[H[k]]) for k in range(4)] for m in MODELS])
    # ---------------------------------------------------------------- S1
    obs_by = np.array([srch.wmean(pre.data["n_looked"][H[k]], w[H[k]]) for k in range(4)])
    pred_by = np.array([srch.wmean(E_ho[H[k]] - 1.0, w[H[k]]) for k in range(4)])
    e_obs = float(sc.ls_slope(LOGN, np.log(obs_by)))
    e_pred = float(sc.ls_slope(LOGN, np.log(pred_by)))
    # ---------------------------------------------------------------- S2
    err = np.zeros((len(MODELS), 4))
    for k in range(4):
        tr = pre.timed & pre.train_mask[k] & (w > 0)
        ho = pre.timed & (pre.size_idx == k) & (w > 0)
        y_tr, w_tr, y_ho, w_ho = pre.log_rt[tr], w[tr], pre.log_rt[ho], w[ho]
        for i, m in enumerate(MODELS):
            if m == "SR":
                idx = np.flatnonzero(tr | ho)
                E_k = np.zeros(T)
                E_k[idx] = srch.sr_expected_looks(lam["SR"][k], C[idx])  # fold k's lambda for every trial
                eff = E_k
            else:
                eff = pre.effort_fixed[m]
            t0, tau, _ = srch.fit_time_law(eff[tr], y_tr, w_tr)
            pred = srch.time_law_predict(t0, tau, eff[ho])
            err[i, k] = srch.wmean(np.abs(y_ho - pred), w_ho)
    # ---------------------------------------------------------------- S3
    g = pre.groups
    gt = g.trials
    corr = np.zeros(len(CORR_KEYS))
    model_forecasts = {"looks_SR": E_ho, "entropy_CL": ent["CL"], "entropy_DbS_avg": ent["DbS_avg"], "entropy_MLBA_avg": ent["MLBA_avg"]}
    for i, key in enumerate(CORR_KEYS):
        rg = g.correlations(model_forecasts[key][gt]) if key in model_forecasts else pre.fixed_corr[key]
        corr[i] = g.participant_mean(rg, wp)
    # ---------------------------------------------------------------- S4
    scores = np.zeros((len(MODELS) + 1, 4))
    for k in range(4):
        h = H[k]
        for i, m in enumerate(MODELS):
            scores[i, k] = bernoulli_score(pre.y_top[h], ptop[m][h], w[h])
        other = pre.train_mask[k]
        base = srch.wmean(pre.y_top[other], w[other])
        scores[-1, k] = bernoulli_score(pre.y_top[h], np.full(len(h), base), w[h])
    # ---------------------------------------------------------------- implied CL sensitivity of SR's choices
    impl = np.zeros(4)
    for k in range(4):
        Ph = pre.held["CL"][k]
        Ps = srch.soft_problem(Ph, Q["SR"][Ph.trials])
        u = w[Ph.trials][None]
        impl[k] = sc.fit_lambda(Ps, np.ones_like(u), u, init=None if init is None else init["impl"][k:k + 1])[0]
    out = dict(lam=lam, L_by=L_by, obs_by=obs_by, pred_by=pred_by, e_obs=e_obs, e_pred=e_pred, err=err, corr=corr,
               scores=scores, impl=impl, e_impl=float(sc.ls_slope(LOGN, np.log(impl))), diag=diag, E_ho=E_ho)
    return out


def per_step(pre: Pre, ev: dict):
    """Original sample only, unweighted: median seconds per set size and per-step seconds."""
    med = np.zeros(4)
    sps = {"CL": np.zeros(4), "DbS_avg": np.zeros(4), "SR": np.zeros(4)}
    for k, nk in enumerate(srch.NS):
        tk = pre.timed & (pre.size_idx == k)
        med[k] = np.median(pre.data["rt_ms"][tk]) / 1000.0
        sps["CL"][k] = med[k] / nk
        sps["DbS_avg"][k] = med[k] / (nk * (nk - 1))
        sps["SR"][k] = med[k] / ev["E_ho"][tk].mean()
    return med, sps


# ============================================================================ summaries
def clear_pos(s):
    return bool(s["clear"] and s["percentile"][0] > 0 and s["basic"][0] > 0)


def clear_neg(s):
    return bool(s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0)


def summarize(pre: Pre, pt: dict, bs: dict) -> dict:
    S = srch.summary_S
    mi = {m: i for i, m in enumerate(MODELS)}
    res = {"lambda": {m: [float(x) for x in pt["lam"][m]] for m in MODELS}}
    # ---------------------------------------------------------------- choices
    L = pt["L_by"].mean(axis=1)
    Lb = bs["L_by"].mean(axis=2)  # (B, models)
    tc = {}
    for m in MODELS:
        lo, hi = np.percentile(Lb[:, mi[m]], [2.5, 97.5])
        tc[m] = {"point": float(2 ** L[mi[m]]), "percentile": [float(2 ** lo), float(2 ** hi)],
                 "by_size": [float(2 ** x) for x in pt["L_by"][mi[m]]]}
    res["choices"] = {"times_chance": tc,
                      "SR_minus_percent": {m: srch.summary_percent(L[mi["SR"]] - L[mi[m]], Lb[:, mi["SR"]] - Lb[:, mi[m]]) for m in STATIC}}
    # ---------------------------------------------------------------- S1
    pmo = S(pt["e_pred"] - pt["e_obs"], bs["e_pred"] - bs["e_obs"])
    omo = S(1 - pt["e_obs"], 1 - bs["e_obs"])
    within = bool(pmo["percentile"][0] >= -0.1 and pmo["percentile"][1] <= 0.1)
    res["S1"] = {"observed_by_size": [float(x) for x in pt["obs_by"]], "predicted_by_size": [float(x) for x in pt["pred_by"]],
                 "observed_over_predicted_by_size": [float(a / b) for a, b in zip(pt["obs_by"], pt["pred_by"])],
                 "observed_elasticity": S(pt["e_obs"], bs["e_obs"]), "predicted_elasticity": S(pt["e_pred"], bs["e_pred"]),
                 "one_minus_observed": omo, "predicted_minus_observed": pmo, "within_margin": within,
                 "supported": bool(clear_pos(omo) and within)}
    # ---------------------------------------------------------------- S2
    E = pt["err"].mean(axis=1)
    Eb = bs["err"].mean(axis=2)
    errs = {}
    for m in MODELS:
        errs[m] = S(E[mi[m]], Eb[:, mi[m]])
        errs[m]["by_size"] = [float(x) for x in pt["err"][mi[m]]]
    srm = {m: S(E[mi["SR"]] - E[mi[m]], Eb[:, mi["SR"]] - Eb[:, mi[m]]) for m in STATIC}
    res["S2"] = {"error": errs, "SR_minus": srm, "supported": bool(clear_neg(srm["CL"]) and clear_neg(srm["DbS_avg"]))}
    # ---------------------------------------------------------------- S3
    ci = {k: i for i, k in enumerate(CORR_KEYS)}
    corr = {k: S(pt["corr"][ci[k]], bs["corr"][:, ci[k]]) for k in CORR_KEYS}
    res["S3"] = {"correlation": corr,
                 "SR_minus_CL": S(pt["corr"][ci["looks_SR"]] - pt["corr"][ci["entropy_CL"]],
                                  bs["corr"][:, ci["looks_SR"]] - bs["corr"][:, ci["entropy_CL"]]),
                 "supported": clear_pos(corr["looks_SR"])}
    # ---------------------------------------------------------------- S4
    sc_pt = pt["scores"].mean(axis=1)
    sc_b = bs["scores"].mean(axis=2)
    names = list(MODELS) + ["constant"]
    res["S4"] = {"score": {m: S(sc_pt[i], sc_b[:, i]) for i, m in enumerate(names)},
                 "minus_constant": {m: S(sc_pt[i] - sc_pt[-1], sc_b[:, i] - sc_b[:, -1]) for i, m in enumerate(MODELS)},
                 "SR_minus_CL": S(sc_pt[mi["SR"]] - sc_pt[mi["CL"]], sc_b[:, mi["SR"]] - sc_b[:, mi["CL"]])}
    # ---------------------------------------------------------------- S5
    s5 = srch.summary_percent(L[mi["SR"]] - L[mi["CL"]], Lb[:, mi["SR"]] - Lb[:, mi["CL"]])
    noninf = bool(s5["percentile"][0] > -2.0 and s5["basic"][0] > -2.0)
    res["S5"] = {"SR_minus_CL_percent": s5, "margin": 2.0, "noninferior": noninf,
                 "better": bool(s5["percentile"][0] > 0 and s5["basic"][0] > 0),
                 "worse": bool(s5["percentile"][1] < 0 and s5["basic"][1] < 0), "supported": noninf}
    # ---------------------------------------------------------------- secondary
    res["implied_elasticity"] = S(pt["e_impl"], bs["e_impl"])
    res["implied_cl_sensitivity"] = [float(x) for x in pt["impl"]]
    med, sps = per_step(pre, pt)
    res["median_seconds"] = [float(x) for x in med]
    res["per_step_seconds"] = {m: [float(x) for x in v] for m, v in sps.items()}
    return res


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with setsize-{9,16,25,36}_desc-data.csv")
    ap.add_argument("--real", action="store_true", help="analyse the real snack data (only when the coordinator says so)")
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
    pt = evaluate(pre, np.ones(pre.P))
    init = {**{fam: pt["lam"][fam] for fam in STATIC}, "impl": pt["impl"]}
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    B = args.B
    keys = ("L_by", "obs_by", "pred_by", "e_obs", "e_pred", "err", "corr", "scores", "impl", "e_impl", "diag")
    bs = {k: np.zeros((B,) + np.shape(pt[k])) for k in keys}
    bs["lam"] = np.zeros((B, len(MODELS), 4))
    for b in range(B):
        draw = np.random.default_rng(SEED_BASE + b).integers(0, pre.P, size=pre.P)
        wp = np.bincount(draw, minlength=pre.P).astype(float)
        ev = evaluate(pre, wp, init)
        for k in keys:
            bs[k][b] = ev[k]
        bs["lam"][b] = np.array([ev["lam"][m] for m in MODELS])
        if (b + 1) % 25 == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    res = summarize(pre, pt, bs)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    outdir = os.path.dirname(os.path.abspath(args.out))
    np.savez_compressed(os.path.join(outdir, f"boot_{stem}.npz"), **bs)
    diag = {"replicate_SR_folds_with_several_grid_maxima": int(bs["diag"][:, 0].sum()),
            "replicate_SR_folds_not_refined": int(bs["diag"][:, 1].sum()),
            "replicate_SR_folds_with_lambda_zero": int(bs["diag"][:, 2].sum()),
            "original_SR_diag": [int(x) for x in pt["diag"]],
            "replicate_lambda_ranges": {m: [float(bs["lam"][:, i].min()), float(bs["lam"][:, i].max())] for i, m in enumerate(MODELS)},
            "seconds": round(time.time() - t0, 1), "B": B}
    with open(os.path.join(outdir, f"diag_{stem}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
