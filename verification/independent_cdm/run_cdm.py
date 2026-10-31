"""Amendment 7 (VERIFY_BRIEF_CDM.md): point estimates, participant bootstrap, summaries.

Usage:
  python -I run_cdm.py --data FILE --out FILE [--B 1000]       (synthetic click files in the input format)
  python -I run_cdm.py --real --out results.json [--B 1000]    (verify/cdm; only when the coordinator says so)
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
import cdm_core as cc  # noqa: E402
from cdm_core import sc, srch  # noqa: E402

srch.assert_local(cc, srch, sc)

RULES = ("value", "share", "value_time", "share_time")
BOOT_SEED = 70000
NAMES = tuple(name for name, _ in cc.EXPERIMENTS)


class Pre:
    """Everything of one experiment that does not depend on the participant weights."""

    def __init__(self, X: dict):
        self.X = X
        self.R, self.P = X["R"], X["P"]
        self.G = cc.Groups(X)
        self.sr = cc.SRFitGroups(self.G)
        fold = X["fold"]
        self.train = [fold != f for f in range(cc.N_FOLDS)]
        self.held = [np.flatnonzero(fold == f) for f in range(cc.N_FOLDS)]
        self.local = [[np.flatnonzero(fold[idx] == f) for (_, idx, _, _) in self.G.items] for f in range(cc.N_FOLDS)]
        sc_, tok = X["s_scored"], X["s_t_ok"]
        self.sel = {}
        for f in range(cc.N_FOLDS):
            for c in range(6):
                base = sc_ & (X["s_cell"] == c)
                tr, ho = base & (X["s_fold"] != f), base & (X["s_fold"] == f)
                self.sel[f, c] = {False: (np.flatnonzero(tr), np.flatnonzero(ho)), True: (np.flatnonzero(tr & tok), np.flatnonzero(ho & tok))}
        self.scored = {False: np.flatnonzero(sc_), True: np.flatnonzero(sc_ & tok)}
        self.pred = {"value": X["s_value"], "share": X["s_share"]}
        self.logn = np.log(X["n"].astype(float))


def evaluate(pre: Pre, wp: np.ndarray, init: dict | None = None) -> dict:
    X, G = pre.X, pre.G
    w_r, w_s = wp[X["pidx"]], wp[X["s_pidx"]]
    R = pre.R
    lam = {m: np.zeros(cc.N_FOLDS) for m in cc.MODELS}
    lq = {m: np.zeros(R) for m in cc.MODELS}  # held out log probability of the chosen option
    E_ho, place_pred = np.zeros(R), np.zeros(R)
    pred = {"static": np.full(R, np.nan), "SR": np.full(R, np.nan)}
    lp = {r: np.full(len(X["s_value"]), np.nan) for r in RULES}
    dense_excess, at_limit, n_fits, n_empty, n_ho = -np.inf, 0, 0, 0, 0
    y_s = X["s_last"]
    for f in range(cc.N_FOLDS):
        wt = np.where(pre.train[f], w_r, 0.0)
        for m in cc.STATIC:
            lam[m][f] = cc.fit_softmax_groups(G.S[m], G.items, wt, init=None if init is None else init[m][f])
            for S, (_, idx, _, c), loc in zip(G.S[m], G.items, pre.local[f]):
                if len(loc):
                    lq[m][idx[loc]] = cc.log_softmax(lam[m][f], S[loc])[np.arange(len(loc)), c[loc]]
        lam["SR"][f], info = pre.sr.fit(wt)
        dense_excess = max(dense_excess, info["dense_excess"])
        ls = lam["SR"][f]
        E_all = G.per_round(lambda V, c: cc.sr_expected_looks(ls, V))
        for (_, idx, V, c), loc in zip(G.items, pre.local[f]):
            if len(loc):
                lq["SR"][idx[loc]] = cc.sr_log_chosen(ls, V[loc], c[loc])
                place_pred[idx[loc]] = cc.attention_place(ls, V[loc])
        h = pre.held[f]
        E_ho[h] = E_all[h]
        tr = pre.train[f] & (w_r > 0) & X["tf_ok"]
        hk = h[X["tf_ok"][h]]
        for k, eff in (("static", X["n"].astype(float)), ("SR", E_all)):
            t0, tau, _ = srch.fit_time_law(eff[tr], X["logt_final"][tr], w_r[tr])
            pred[k][hk] = srch.time_law_predict(t0, tau, eff[hk])
        for c in range(6):
            for rule in RULES:
                timed = rule.endswith("_time")
                itr, iho = pre.sel[f, c][timed]
                itr = itr[w_s[itr] > 0]
                if len(iho) == 0:
                    continue
                n_ho += 1
                if len(itr) == 0:
                    # No training selection in this cell: neither rule can be fitted. Provisional choice, to be
                    # settled in the amendment: both rules give probability 1/2, so they predict alike here.
                    lp[rule][iho] = -1.0
                    n_empty += 1
                    continue
                x = pre.pred[rule.split("_")[0]]
                m1, s1 = cc.wstandardize(x[itr], w_s[itr])
                Ztr, Zho = [cc.zscore(x[itr], m1, s1)], [cc.zscore(x[iho], m1, s1)]
                if timed:
                    m2, s2 = cc.wstandardize(X["s_logt"][itr], w_s[itr])
                    Ztr.append(cc.zscore(X["s_logt"][itr], m2, s2))
                    Zho.append(cc.zscore(X["s_logt"][iho], m2, s2))
                beta = cc.fit_logit(np.column_stack(Ztr), y_s[itr], w_s[itr])
                n_fits += 1
                at_limit += int(not np.isfinite(beta[0]))
                lp[rule][iho] = cc.logit_log2_prob(beta, np.column_stack(Zho), y_s[iho])
    on = w_r > 0
    log2n = np.log2(X["n"].astype(float))
    L = np.array([srch.wmean(log2n[on] + np.maximum(lq[m][on], cc.LOG_FLOOR) / cc.LN2, w_r[on]) for m in cc.MODELS])
    stop = []
    for rule in RULES:
        idx = pre.scored[rule.endswith("_time")]
        idx = idx[w_s[idx] > 0]
        stop.append(srch.wmean(lp[rule][idx], w_s[idx]))
    first = np.array([srch.wmean(X["place"][on], w_r[on]), srch.wmean(place_pred[on], w_r[on])])
    ok = on & X["tf_ok"]
    err = np.array([srch.wmean(np.abs(X["logt_final"][ok] - pred[k][ok]), w_r[ok]) for k in ("static", "SR")])
    # growth: per complexity, slope of log weighted geometric means over the three set sizes against log n
    xs = np.log(np.array(cc.SIZES, float))
    slopes = np.zeros((2, 2))
    for ci in range(2):
        for j, (vals, rows) in enumerate(((X["logt_final"], ok), (np.log(E_ho), on))):
            lg = np.array([srch.wmean(vals[rows & (X["cell"] == si * 2 + ci)], w_r[rows & (X["cell"] == si * 2 + ci)]) for si in range(3)])
            slopes[j, ci] = np.polyfit(xs, lg, 1)[0]
    growth = slopes.mean(axis=1)  # observed, predicted
    by_size = np.array([[np.exp(srch.wmean(X["logt_final"][ok & (X["size_idx"] == si)], w_r[ok & (X["size_idx"] == si)])) for si in range(3)],
                        [srch.wmean(E_ho[on & (X["size_idx"] == si)], w_r[on & (X["size_idx"] == si)]) for si in range(3)]])
    return dict(lam=np.vstack([lam[m] for m in cc.MODELS]), L=L, stop=np.array(stop), first=first, err=err, growth=growth,
                by_size=by_size, dense_excess=dense_excess, limit_share=at_limit / max(n_fits, 1),
                empty_training_share=n_empty / max(n_ho, 1))


# ============================================================================ summaries
S = srch.summary_S


def clear_pos(s):
    return bool(s["clear"] and s["percentile"][0] > 0 and s["basic"][0] > 0)


def clear_neg(s):
    return bool(s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0)


def block(pt: dict, bs: dict) -> dict:
    st, sb = pt["stop"], bs["stop"]
    stop = {r: S(st[i], sb[:, i]) for i, r in enumerate(RULES)}
    stop["value_minus_share"] = S(st[0] - st[1], sb[:, 0] - sb[:, 1])
    stop["value_minus_share_with_time"] = S(st[2] - st[3], sb[:, 2] - sb[:, 3])
    g, gb = pt["growth"], bs["growth"]
    growth = {"observed": S(g[0], gb[:, 0]), "predicted": S(g[1], gb[:, 1]), "one_minus_observed": S(1 - g[0], 1 - gb[:, 0]),
              "predicted_minus_observed": S(g[1] - g[0], gb[:, 1] - gb[:, 0])}
    L, Lb = pt["L"], bs["L"]
    mi = {m: i for i, m in enumerate(cc.MODELS)}
    choices = {"times_chance": {m: float(2 ** L[mi[m]]) for m in cc.MODELS}}
    for m in cc.STATIC:
        choices[f"SR_minus_{m}_percent"] = srch.summary_percent(L[mi["SR"]] - L[mi[m]], Lb[:, mi["SR"]] - Lb[:, mi[m]])
    e, eb = pt["err"], bs["err"]
    tm = {"static": S(e[0], eb[:, 0]), "SR": S(e[1], eb[:, 1]), "SR_minus_static": S(e[1] - e[0], eb[:, 1] - eb[:, 0])}
    return {"stop": stop, "growth": growth, "choices": choices, "time": tm}


def experiment_extras(pt: dict, bs: dict) -> dict:
    fo, fb = pt["first"], bs["first"]
    return {"first": {"observed": S(fo[0], fb[:, 0]), "predicted": S(fo[1], fb[:, 1]), "minus_half": S(fo[0] - 0.5, fb[:, 0] - 0.5),
                      "minus_predicted": S(fo[0] - fo[1], fb[:, 0] - fb[:, 1])},
            "lambda": {m: [float(x) for x in pt["lam"][i]] for i, m in enumerate(cc.MODELS)}}


def summarize(pts: dict, bss: dict) -> dict:
    keys = ("stop", "growth", "L", "err")
    pooled_pt = {k: np.mean([pts[n][k] for n in NAMES], axis=0) for k in keys}
    pooled_bs = {k: np.mean([bss[n][k] for n in NAMES], axis=0) for k in keys}
    res = {"pooled": block(pooled_pt, pooled_bs)}
    p = res["pooled"]
    vms = p["stop"]["value_minus_share"]
    c3 = p["choices"]["SR_minus_CL_percent"]
    noninf = bool(min(c3["percentile"][0], c3["basic"][0]) > -2.0)
    for n in NAMES:
        blk = block(pts[n], bss[n])
        ex = experiment_extras(pts[n], bss[n])
        blk["first"] = ex["first"]
        blk["lambda"] = ex["lambda"]
        blk["growth"]["observed_by_size"] = [float(x) for x in pts[n]["by_size"][0]]
        blk["growth"]["looks_by_size"] = [float(x) for x in pts[n]["by_size"][1]]
        res[n] = blk
    fp = res["process"]["first"]
    p["C1"] = {"absolute": clear_pos(vms), "relative": clear_neg(vms)}
    p["C2"] = {"order_matters": bool(clear_neg(fp["minus_half"]) and clear_neg(fp["minus_predicted"]))}
    p["C3"] = {"margin": 2.0, "noninferior": noninf, "better": bool(c3["percentile"][0] > 0 and c3["basic"][0] > 0),
               "worse": bool(c3["percentile"][1] < 0 and c3["basic"][1] < 0), "supported": noninf}
    return res


def precheck_draws(pres: dict, B: int):
    """Before any fitting: replay every replicate's draws (the same generators, Experiment 1 first) and stop if a
    cell of set size and complexity would hold no drawn participant's timed round, where the growth elasticities
    and the cell's rules are undefined and the amendment says nothing. Prints no counts."""
    for b in range(B):
        rng = np.random.default_rng(BOOT_SEED + b)
        for n in NAMES:
            X, P = pres[n].X, pres[n].P
            w_r = np.bincount(rng.integers(0, P, size=P), minlength=P)[X["pidx"]]
            filled = np.bincount(X["cell"][(w_r > 0) & X["tf_ok"]], minlength=6) > 0
            if not filled.all():
                raise SystemExit(f"{n}: bootstrap replicate {b} leaves a cell of set size and complexity without any drawn "
                                 "participant's timed round; the amendment does not say how to treat this, so nothing was fitted")
    print("draw pre-check passed: every replicate keeps a drawn participant's timed round in every cell", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="a click file in the input format (synthetic)")
    ap.add_argument("--real", action="store_true", help="analyse verify/cdm (only when the coordinator says so)")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = os.path.join(cc.CDM_DIR, cc.CLICK_FILE) if args.real else args.data
    if path is None:
        raise SystemExit("give --data for a synthetic click file, or --real for the real one")
    t0 = time.time()
    data = cc.load_cdm(path, allow_real=args.real)
    pres = {n: Pre(data[n]) for n in NAMES}
    print(f"precomputation done in {time.time() - t0:.1f}s", flush=True)
    precheck_draws(pres, args.B)
    pts = {n: evaluate(pres[n], np.ones(pres[n].P)) for n in NAMES}
    inits = {n: {m: pts[n]["lam"][i] for i, m in enumerate(cc.MODELS) if m in cc.STATIC} for n in NAMES}
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    B = args.B
    keys = ("lam", "L", "stop", "first", "err", "growth", "dense_excess", "limit_share", "empty_training_share")
    bss = {n: {k: np.zeros((B,) + np.shape(pts[n][k])) for k in keys} for n in NAMES}
    for b in range(B):
        rng = np.random.default_rng(BOOT_SEED + b)
        for n in NAMES:  # Experiment 1 first, then Experiment 2, from one generator
            P = pres[n].P
            wp = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
            ev = evaluate(pres[n], wp, inits[n])
            for k in keys:
                bss[n][k][b] = ev[k]
        if (b + 1) % 25 == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    res = summarize(pts, bss)
    res["unsubmitted"] = {n: data[n]["unsubmitted"] for n in NAMES}  # descriptive, point values only
    stem = os.path.splitext(os.path.basename(args.out))[0]
    outdir = os.path.dirname(os.path.abspath(args.out))
    np.savez_compressed(os.path.join(outdir, f"boot_{stem}.npz"), **{f"{n}__{k}": v for n, d in bss.items() for k, v in d.items()})
    diag = {n: {**data[n]["diag"],
                "largest_dense_excess_over_SR_fit": float(max(pts[n]["dense_excess"], bss[n]["dense_excess"].max())),
                "share_of_logistic_fits_at_the_all_equal_limit": float(max(pts[n]["limit_share"], bss[n]["limit_share"].max())),
                "share_of_rule_cells_without_training_selections_original": float(pts[n]["empty_training_share"]),
                "share_of_rule_cells_without_training_selections_largest_replicate": float(bss[n]["empty_training_share"].max()),
                "replicate_lambda_ranges": {m: [float(bss[n]["lam"][:, i].min()), float(bss[n]["lam"][:, i].max())]
                                            for i, m in enumerate(cc.MODELS)}}
            for n in NAMES}
    diag["seconds"] = round(time.time() - t0, 1)
    diag["B"] = B
    with open(os.path.join(outdir, f"diag_{stem}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
