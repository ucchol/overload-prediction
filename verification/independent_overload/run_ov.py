"""Amendment 8: point estimates, participant bootstrap, summaries.

Usage:
  python -I run_ov.py --cdm CLICKS.csv --reut TRIALS.dta --out FILE [--B 1000]   (synthetic files)
  python -I run_ov.py --real --out results.json [--B 1000]                       (only when the coordinator says so)
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
import ov_core as oc  # noqa: E402
from ov_core import cc, srch  # noqa: E402

srch.assert_local(oc, cc, srch)
S = srch.summary_S


def draws(rng, Pg):
    """Draw counts of the three groups from one generator, in the brief's order."""
    return {g: np.bincount(rng.integers(0, Pg[g], size=Pg[g]), minlength=Pg[g]).astype(float) for g in oc.GROUPS}


def precheck(panels, Pg, B):
    for b in range(B):
        wg = draws(np.random.default_rng(oc.BOOT_SEED + b), Pg)
        for P in panels:
            w = wg[P["group"]][P["pidx"]]
            if not all((w[P["size_idx"] == k] > 0).any() for k in range(3)):
                raise SystemExit(f"{P['name']}: bootstrap replicate {b} leaves a set size without trials; nothing was fitted")
    print("draw pre-check passed: every replicate keeps trials at every set size of every panel", flush=True)


def evaluate_all(fits, wg, init=None):
    return [oc.evaluate_panel(pf, wg[pf.P["group"]][pf.P["pidx"]], None if init is None else init[i]) for i, pf in enumerate(fits)]


def dataset_vector(evs, names):
    """Per dataset: growth (observed, CL, CL_comp, SR), error (CL, CL_comp, SR), auc (five forecasts), means over panels."""
    out = {}
    for ds, members in (("cdm", [i for i, n in enumerate(names) if n.startswith("cdm")]), ("reutskaja", [names.index("reutskaja")])):
        g = np.array([[evs[i]["growth"][k] for k in ("observed",) + oc.MODELS] for i in members])
        err = np.abs(g[:, 1:] - g[:, :1])
        a = np.array([[evs[i]["auc"][f] for f in oc.FORECASTS] for i in members])
        out[ds] = np.concatenate([g.mean(axis=0), err.mean(axis=0), a.mean(axis=0)])
    return out


IDX = {"growth": slice(0, 4), "error": slice(4, 7), "auc": slice(7, 12)}


def summarize_ds(pt, bs):
    gk = ("observed",) + oc.MODELS
    res = {"growth": {k: S(pt[i], bs[:, i]) for i, k in enumerate(gk)},
           "error": {k: S(pt[4 + i], bs[:, 4 + i]) for i, k in enumerate(oc.MODELS)},
           "auc": {k: S(pt[7 + i], bs[:, 7 + i]) for i, k in enumerate(oc.FORECASTS)}}
    res["comp_minus_cl_error"] = S(pt[5] - pt[4], bs[:, 5] - bs[:, 4])
    res["observed_minus_comp"] = S(pt[0] - pt[2], bs[:, 0] - bs[:, 2])
    res["observed_minus_cl"] = S(pt[0] - pt[1], bs[:, 0] - bs[:, 1])
    res["sr_auc_minus_half"] = S(pt[9] - 0.5, bs[:, 9] - 0.5)
    res["sr_auc_minus_near_best"] = S(pt[9] - pt[10], bs[:, 9] - bs[:, 10])

    def pos(d):
        return bool(d["clear"] and d["percentile"][0] > 0 and d["basic"][0] > 0)

    def neg(d):
        return bool(d["clear"] and d["percentile"][1] < 0 and d["basic"][1] < 0)

    over, comp = pos(res["observed_minus_comp"]), neg(res["observed_minus_cl"])
    res["O1"] = {"verdict": "partial compensation" if (over and comp) else "overload" if over else "compensation" if comp else "undecided"}
    res["O2"] = {"supported": pos(res["sr_auc_minus_near_best"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdm")
    ap.add_argument("--reut")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.real:
        cdm_path, reut_path = os.path.join(cc.CDM_DIR, cc.CLICK_FILE), os.path.join(oc.REUT_DIR, oc.REUT_FILE)
    else:
        cdm_path, reut_path = args.cdm, args.reut
    t0 = time.time()
    panels, Pg = oc.build(cdm_path, reut_path, allow_real=args.real)
    fits = [oc.PanelFits(P) for P in panels]
    names = [P["name"] for P in panels]
    assert tuple(names) == oc.PANELS
    print(f"precomputation done in {time.time() - t0:.1f}s", flush=True)
    precheck(panels, Pg, args.B)
    ones = {g: np.ones(Pg[g]) for g in oc.GROUPS}
    pts = evaluate_all(fits, ones)
    init = [{m: ev["lam"][m] for m in ("CL", "CL_comp")} | {"SR": [None] * 3} for ev in pts]
    pt_ds = dataset_vector(pts, names)
    B = args.B
    bs_ds = {ds: np.zeros((B, 12)) for ds in pt_ds}
    lam_b = np.zeros((B, len(panels), 3, 3))
    dense = max(ev["dense_excess"] for ev in pts)
    for b in range(B):
        wg = draws(np.random.default_rng(oc.BOOT_SEED + b), Pg)
        evs = evaluate_all(fits, wg, init)
        for ds, v in dataset_vector(evs, names).items():
            bs_ds[ds][b] = v
        lam_b[b] = [[ev["lam"][m] for m in oc.MODELS] for ev in evs]
        dense = max(dense, max(ev["dense_excess"] for ev in evs))
        if (b + 1) % 50 == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    res = {"datasets": {ds: summarize_ds(pt_ds[ds], bs_ds[ds]) for ds in pt_ds}, "panels": {}}
    for P, ev in zip(panels, pts):
        entry = {"sizes": [int(s) for s in P["sizes"]],
                 "rate": {k: [float(x) for x in ev["rates"][k]] for k in ("observed",) + oc.MODELS},
                 "lambda": {m: [float(x) for x in ev["lam"][m]] for m in oc.MODELS}}
        if P["name"] == "reutskaja":
            m = P["miss"] > 0.5
            entry["miss_best_unseen"] = [float(P["best_unseen"][m & (P["size_idx"] == k)].mean()) if (m & (P["size_idx"] == k)).any()
                                         else float("nan") for k in range(3)]
        res["panels"][P["name"]] = entry
    stem = os.path.splitext(os.path.basename(args.out))[0]
    outdir = os.path.dirname(os.path.abspath(args.out))
    np.savez_compressed(os.path.join(outdir, f"boot_{stem}.npz"), lam=lam_b, **{f"ds_{k}": v for k, v in bs_ds.items()})
    diag = {"seconds": round(time.time() - t0, 1), "B": B, "largest_dense_excess_over_SR_fit": float(dense),
            "replicate_lambda_ranges": {n: {m: [float(lam_b[:, i, j].min()), float(lam_b[:, i, j].max())] for j, m in enumerate(oc.MODELS)}
                                        for i, n in enumerate(names)}}
    with open(os.path.join(outdir, f"diag_{stem}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
