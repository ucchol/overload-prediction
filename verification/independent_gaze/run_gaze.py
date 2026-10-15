"""Amendment 6 (VERIFY_BRIEF_GAZE.md): point estimates, participant bootstrap, summaries.

Usage:
  python -I run_gaze.py --data DIR --out FILE [--B 1000]       (synthetic files in the input format)
  python -I run_gaze.py --real --out results.json [--B 1000]   (verify/gaze; only when the coordinator says so)
Bootstrap arrays (no counts) go next to the output as boot_<name>.npz, diagnostics as diag_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gaze_core as gc  # noqa: E402
from gaze_core import sc, srch  # noqa: E402

srch.assert_local(gc, srch, sc)

TIME_KEYS = ("static", "SR", "TLS")
FEATURES = ("top_value", "mean_value", "top_gap")
SLOW_KEYS = ("looks_TLS", "looks_SR") + FEATURES
CHOICE_MODELS = ("CL", "DbS", "MLBA", "SR")  # SR's choices are TLS's
CONF_KEYS = ("looks_TLS", "looks_SR", "entropy_CL")
BOOT_SEED = 60000


class DataPre:
    """Everything of one dataset that does not depend on the participant weights."""

    def __init__(self, d: dict):
        self.d = d
        self.T = T = len(d["choice"])
        self.V, self.choice, self.n, self.y = d["V"], d["choice"], d["n"], d["y"]
        self.S = gc.static_scores(self.V)
        self.pidx, self.P = d["pidx"], d["P"]
        self.train = [d["fold"] != f for f in range(gc.N_FOLDS)]
        self.held = [np.flatnonzero(d["fold"] == f) for f in range(gc.N_FOLDS)]
        self.sr = gc.SRFit(self.V, self.choice)
        base = {"pidx": self.pidx, "size_idx": np.zeros(T, int), "P": self.P}
        self.g_rt = srch.Groups({**base, "timed": np.ones(T, bool), "rt_ms": d["rt_ms"]})  # participants, >= 3 trials
        gt = self.g_rt.trials
        self.fixed = {k: self.g_rt.correlations(gc.round10(np.asarray(d[k], float)[gt])) for k in FEATURES}  # forecasts rounded (.10g)
        self.g_conf = None
        if d["conf"] is not None:
            has = np.isfinite(d["conf"])
            self.g_conf = srch.Groups({**base, "timed": has, "rt_ms": np.where(has, d["conf"], 0.0)})


def evaluate_dataset(pre: DataPre, wp: np.ndarray, init: dict | None = None) -> dict:
    w = wp[pre.pidx]
    on = w > 0
    T, V, ch, y = pre.T, pre.V, pre.choice, pre.y
    r = np.arange(T)
    lam = {m: np.zeros(gc.N_FOLDS) for m in CHOICE_MODELS}
    dense_excess = -np.inf
    logq = {m: np.zeros((T, pre.n)) for m in gc.STATIC}
    lsr = np.zeros(T)
    E_sr_ho, E_tls_ho = np.zeros(T), np.zeros(T)
    pred = {k: np.zeros(T) for k in TIME_KEYS}
    for f in range(gc.N_FOLDS):
        h = pre.held[f]
        wt = np.where(pre.train[f], w, 0.0)
        for m in gc.STATIC:
            lam[m][f] = gc.fit_softmax(pre.S[m], ch, wt, init=None if init is None else init[m][f])
            logq[m][h] = gc.log_softmax(lam[m][f], pre.S[m][h])
        lam["SR"][f], info = pre.sr.fit(wt)
        dense_excess = max(dense_excess, info["dense_excess"])
        lsr[h] = gc.sr_log_chosen(lam["SR"][f], V[h], ch[h])
        tr = pre.train[f] & on
        E_sr = gc.sr_expected_looks(lam["SR"][f], V)  # every trial with the fold's sensitivity
        E_tls = gc.tls_expected_looks(lam["SR"][f], V)
        pred["static"][h] = srch.wmean(y[tr], w[tr])  # effort n is constant within a dataset: the law is a constant
        for k, E in (("SR", E_sr), ("TLS", E_tls)):
            t0, tau, _ = srch.fit_time_law(E[tr], y[tr], w[tr])
            pred[k][h] = srch.time_law_predict(t0, tau, E[h])
        E_sr_ho[h], E_tls_ho[h] = E_sr[h], E_tls[h]
    err = np.array([srch.wmean(np.abs(y[on] - pred[k][on]), w[on]) for k in TIME_KEYS])
    g, gt = pre.g_rt, pre.g_rt.trials
    R = gc.round10  # forecasts are rounded to ten significant digits before ranking; response times are not
    corr = np.array([g.participant_mean(g.correlations(R(E_tls_ho[gt])), wp), g.participant_mean(g.correlations(R(E_sr_ho[gt])), wp)]
                    + [g.participant_mean(pre.fixed[k], wp) for k in FEATURES])
    log2n = np.log2(pre.n)
    bits = {m: log2n + np.maximum(logq[m][r, ch], gc.LOG_FLOOR) / gc.LN2 for m in gc.STATIC}  # floored at 1e-300
    bits["SR"] = log2n + lsr / gc.LN2  # already floored
    L = np.array([srch.wmean(bits[m][on], w[on]) for m in CHOICE_MODELS])
    # time per look: SR's sensitivity and the two laws fitted to every trial, no folds
    lam_all, info_all = pre.sr.fit(w)
    dense_excess = max(dense_excess, info_all["dense_excess"])
    law = np.zeros((2, 2))
    for i, E in enumerate((gc.sr_expected_looks(lam_all, V), gc.tls_expected_looks(lam_all, V))):
        t0, tau, _ = srch.fit_time_law(E[on], y[on], w[on])
        law[i] = (t0, tau)
    looks = np.array([srch.wmean(E_sr_ho[on], w[on]), srch.wmean(E_tls_ho[on], w[on])])
    typical = float(np.exp(srch.wmean(y[on], w[on])))
    out = dict(lam=np.vstack([lam[m] for m in CHOICE_MODELS]), err=err, corr=corr, L=L, law=law, looks=looks,
               typical=typical, lam_all=lam_all, dense_excess=dense_excess)
    if pre.g_conf is not None:
        gcf, gtc = pre.g_conf, pre.g_conf.trials
        ent = gc.entropy_bits(logq["CL"])
        out["conf"] = np.array([gcf.participant_mean(gcf.correlations(R(E_tls_ho[gtc])), wp),  # confidence itself not rounded
                                gcf.participant_mean(gcf.correlations(R(E_sr_ho[gtc])), wp),
                                gcf.participant_mean(gcf.correlations(R(ent[gtc])), wp)])
    return out


# ============================================================================ summaries
def clear_pos(s):
    return bool(s["clear"] and s["percentile"][0] > 0 and s["basic"][0] > 0)


def clear_neg(s):
    return bool(s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0)


def block(pt: dict, bs: dict) -> dict:
    S = srch.summary_S
    e, eb = pt["err"], bs["err"]
    tm = {k: S(e[i], eb[:, i]) for i, k in enumerate(TIME_KEYS)}
    tm["TLS_minus_SR"] = S(e[2] - e[1], eb[:, 2] - eb[:, 1])
    tm["TLS_minus_static"] = S(e[2] - e[0], eb[:, 2] - eb[:, 0])
    c, cb = pt["corr"], bs["corr"]
    slow = {k: S(c[i], cb[:, i]) for i, k in enumerate(SLOW_KEYS)}
    slow["TLS_minus_SR"] = S(c[0] - c[1], cb[:, 0] - cb[:, 1])
    for j, k in enumerate(FEATURES, start=2):
        slow[f"TLS_plus_{k}"] = S(c[0] + c[j], cb[:, 0] + cb[:, j])
    L, Lb = pt["L"], bs["L"]
    mi = {m: i for i, m in enumerate(CHOICE_MODELS)}
    choices = {"times_chance": {m: float(2 ** L[mi[m]]) for m in CHOICE_MODELS}}
    for m in gc.STATIC:
        choices[f"TLS_minus_{m}_percent"] = srch.summary_percent(L[mi["SR"]] - L[mi[m]], Lb[:, mi["SR"]] - Lb[:, mi[m]])
    return {"time": tm, "slow": slow, "choices": choices}


def dataset_extras(pt: dict, bs: dict) -> dict:
    S = srch.summary_S
    return {"lambda": {m: [float(x) for x in pt["lam"][i]] for i, m in enumerate(CHOICE_MODELS)},
            "law": {m: {"t0": S(pt["law"][i, 0], bs["law"][:, i, 0]), "tau": S(pt["law"][i, 1], bs["law"][:, i, 1])}
                    for i, m in enumerate(("SR", "TLS"))},
            "looks": {m: S(pt["looks"][i], bs["looks"][:, i]) for i, m in enumerate(("SR", "TLS"))},
            "typical_seconds": S(pt["typical"], bs["typical"])}


def summarize(pts: dict, bss: dict) -> dict:
    res = {}
    pooled_pt = {k: np.mean([pts[n][k] for n in gc.DATASETS], axis=0) for k in ("err", "corr", "L")}
    pooled_bs = {k: np.mean([bss[n][k] for n in gc.DATASETS], axis=0) for k in ("err", "corr", "L")}
    res["pooled"] = block(pooled_pt, pooled_bs)
    t, s = res["pooled"]["time"], res["pooled"]["slow"]
    c3 = res["pooled"]["choices"]["TLS_minus_CL_percent"]
    noninf = bool(min(c3["percentile"][0], c3["basic"][0]) > -2.0)
    res["pooled"]["G1"] = {"supported": bool(clear_neg(t["TLS_minus_SR"]) and clear_neg(t["TLS_minus_static"]))}
    res["pooled"]["G2"] = {"supported": bool(clear_pos(s["looks_TLS"]) and clear_pos(s["TLS_minus_SR"]))}
    res["pooled"]["G3"] = {"margin": 2.0, "noninferior": noninf,
                           "better": bool(c3["percentile"][0] > 0 and c3["basic"][0] > 0),
                           "worse": bool(c3["percentile"][1] < 0 and c3["basic"][1] < 0), "supported": noninf}
    for n in gc.DATASETS:
        blk = block(pts[n], bss[n])
        blk.update(dataset_extras(pts[n], bss[n]))
        if "conf" in pts[n]:
            S = srch.summary_S
            cf, cfb = pts[n]["conf"], bss[n]["conf"]
            blk["confidence"] = {**{k: S(cf[i], cfb[:, i]) for i, k in enumerate(CONF_KEYS)},
                                 "TLS_minus_SR": S(cf[0] - cf[1], cfb[:, 0] - cfb[:, 1])}
        res[n] = blk
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with the four input files")
    ap.add_argument("--real", action="store_true", help="analyse the real gaze data (only when the coordinator says so)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data_dir = gc.GAZE_DIR if args.real else args.data
    if data_dir is None:
        raise SystemExit("give --data for test data, or --real for the real gaze data")
    t0 = time.time()
    pres = {n: DataPre(gc.load_dataset(data_dir, n, allow_real=args.real)) for n in gc.DATASETS}
    print(f"precomputation done in {time.time() - t0:.1f}s", flush=True)
    pts = {n: evaluate_dataset(pres[n], np.ones(pres[n].P)) for n in gc.DATASETS}
    inits = {n: {m: pts[n]["lam"][i] for i, m in enumerate(CHOICE_MODELS) if m in gc.STATIC} for n in gc.DATASETS}
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    B = args.B
    keys = ("lam", "err", "corr", "L", "law", "looks", "typical", "lam_all", "dense_excess")
    bss = {n: {k: np.zeros((B,) + np.shape(pts[n][k])) for k in keys + (("conf",) if "conf" in pts[n] else ())} for n in gc.DATASETS}
    for b in range(B):
        rng = np.random.default_rng(BOOT_SEED + b)
        for n in gc.DATASETS:  # Krajbich 2010, then Krajbich 2011, then Folke, from one generator
            P = pres[n].P
            wp = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
            ev = evaluate_dataset(pres[n], wp, inits[n])
            for k in bss[n]:
                bss[n][k][b] = ev[k]
        if (b + 1) % 25 == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    res = summarize(pts, bss)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    outdir = os.path.dirname(os.path.abspath(args.out))
    np.savez_compressed(os.path.join(outdir, f"boot_{stem}.npz"), **{f"{n}__{k}": v for n, d in bss.items() for k, v in d.items()})
    diag = {n: {"largest_dense_excess_over_SR_fit": float(max(pts[n]["dense_excess"], bss[n]["dense_excess"].max())),
                "largest_item_value": float(pres[n].V.max()),
                "smallest_item_value": float(pres[n].V.min()),
                "original_lambda_all_SR": float(pts[n]["lam_all"]),
                "replicate_lambda_ranges": {m: [float(bss[n]["lam"][:, i].min()), float(bss[n]["lam"][:, i].max())]
                                            for i, m in enumerate(CHOICE_MODELS)}}
            for n in gc.DATASETS}
    diag["seconds"] = round(time.time() - t0, 1)
    diag["B"] = B
    with open(os.path.join(outdir, f"diag_{stem}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
