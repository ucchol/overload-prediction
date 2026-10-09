"""Confirmatory H1 on the snack data: point estimates, participant bootstrap, summaries.

Usage: python -I run_snack.py [--B 1000] [--out results.json]
Writes results.json in the layout of VERIFY_BRIEF_SNACK.md and extras.json with diagnostics.
The bootstrap arrays (log scores, lambdas, betas, effective numbers, shares; no counts) go to
boot_snack_B{B}.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import snack_core as sc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B0 = sc.BETAS.index(0.0)
NB, NF = len(sc.BETAS), 4
PROB_BETAS = np.repeat(np.array(sc.BETAS), NF)  # problem k = (beta index, held-out set size)
PROB_FOLDS = np.tile(np.arange(NF), NB)
LOG2N = np.log2(np.array(sc.SET_SIZES, float))


ALONE = ("all_items_all_trials", "all_items_C3_trials", "seen_items_C3_trials")


def build_problems(data: dict) -> dict:
    """Everything that does not depend on the bootstrap weights, including every trial subset used."""
    T = len(data["choice"])
    every = np.arange(T)
    seen = np.where(data["seen_chosen"])[0]
    probs = {}
    for fam in sc.FAMILIES:
        probs[(fam, "all")] = sc.Problem(data, fam, "all", every)
        probs[(fam, "seen")] = sc.Problem(data, fam, "seen", seen)
    probs["C3_all"] = sc.Problem(data, "CL", "all", seen)  # all items, trials whose chosen item was looked at
    probs["C3_seen"] = probs[("CL", "seen")]
    probs["train"], probs["held"] = {}, {}
    for fam in sc.FAMILIES:
        for ver in sc.VERSIONS:
            P = probs[(fam, ver)]
            probs["train"][(fam, ver)] = [P.subset(np.where(P.size_idx != h)[0]) for h in range(NF)]
            probs["held"][(fam, ver)] = [P.subset(np.where(P.size_idx == h)[0]) for h in range(NF)]
    probs["alone"] = {
        name: [P.subset(np.where(P.size_idx == s)[0]) for s in range(NF)]
        for name, P in zip(ALONE, (probs[("CL", "all")], probs["C3_all"], probs["C3_seen"]))
    }
    return probs


def evaluate_sample(data: dict, probs: dict, w: np.ndarray, b: int, init: dict | None = None) -> dict:
    """Every fit and statistic of one sample. w are trial weights; b = -1 is the original sample."""
    out: dict = {"lam": {}, "L_by": {}}
    betas = np.array(sc.BETAS)
    # ---------------------------------------------------------------- C2 cells, leave one set size out
    for fam in sc.FAMILIES:
        for ver in sc.VERSIONS:
            lam = np.zeros((NB, NF))
            L_by = np.zeros((NB, NF))
            for h in range(NF):
                Pt = probs["train"][(fam, ver)][h]
                Ph = probs["held"][(fam, ver)][h]
                mult_t = np.exp(betas[:, None] * Pt.logA[None])
                u_t = np.broadcast_to(w[Pt.trials][None], mult_t.shape)
                lam[:, h] = sc.fit_lambda(Pt, mult_t, u_t, init=None if init is None else init["lam"][(fam, ver)][:, h])
                bits = sc.log2_scores(Ph, lam[:, h], np.exp(betas[:, None] * Ph.logA[None]))
                wh = w[Ph.trials]
                L_by[:, h] = (bits * wh[None]).sum(axis=1) / wh.sum()
            out["lam"][(fam, ver)] = lam
            out["L_by"][(fam, ver)] = L_by
    # ---------------------------------------------------------------- C1 by position
    lam_cl = out["lam"][("CL", "all")][B0]  # held-out CL, beta = 0, all items
    e_obs, e_mod, p_list, q_list, N_list = np.zeros(4), np.zeros(4), [], [], []
    start = 0
    for k, n in enumerate(sc.SET_SIZES):
        V = data["V_by_size"][k]
        wk = w[start:start + len(V)]
        start += len(V)
        z = lam_cl[k] * V
        q = np.exp(z - z.max(axis=1, keepdims=True))
        q /= q.sum(axis=1, keepdims=True)
        qbar = (wk[:, None] * q).sum(axis=0) / wk.sum()
        p = np.bincount(data["choice"][data["size_idx"] == k], weights=wk, minlength=n) / wk.sum()
        N = int(round(wk.sum()))
        e_obs[k] = sc.effective_number(p)
        e_mod[k] = sc.effective_number_sampled(qbar, N, (1, 2, k, b + 1))
        p_list.append(p)
        q_list.append(qbar)
        N_list.append(N)
    out.update(e_obs=e_obs, e_mod=e_mod, p_pos=p_list, q_pos=q_list, N_pos=N_list)
    # ---------------------------------------------------------------- C3 joint lambda and beta
    for key in ("C3_all", "C3_seen"):
        P = probs[key]
        start = None if init is None else (init[key][0], init[key][1])
        out[key] = sc.fit_joint(P, w[P.trials], start=start)
    # ---------------------------------------------------------------- descriptives
    sel = [np.where(data["size_idx"] == k)[0] for k in range(4)]
    out["share_looked"] = np.array([np.average(data["frac_looked"][s], weights=w[s]) for s in sel])
    out["share_chosen_looked"] = np.array([np.average(data["seen_chosen"][s], weights=w[s]) for s in sel])
    alone = {}
    for name in ALONE:
        vals = np.zeros(NF)
        for s in range(NF):
            Ps = probs["alone"][name][s]
            vals[s] = sc.fit_lambda(Ps, np.ones((1, len(Ps.trials))), w[Ps.trials][None],
                                    init=None if init is None else init["alone"][name][s:s + 1])[0]
        alone[name] = vals
    out["alone"] = alone
    return out


def run_bootstrap(data, probs, point, B, log_every=50):
    cells = [(f, v) for f in sc.FAMILIES for v in sc.VERSIONS]
    st = {
        "L_by": np.zeros((B, len(cells), NB, NF)),
        "lam": np.zeros((B, len(cells), NB, NF)),
        "e_obs": np.zeros((B, 4)),
        "e_mod": np.zeros((B, 4)),
        "c3": np.zeros((B, 2, 3)),
        "share_looked": np.zeros((B, 4)),
        "share_chosen_looked": np.zeros((B, 4)),
        "alone": np.zeros((B, 3, 4)),
    }
    init = {"lam": point["lam"], "C3_all": point["C3_all"], "C3_seen": point["C3_seen"], "alone": point["alone"]}
    t0 = time.time()
    for b in range(B):
        draw = np.random.default_rng(20000 + b).integers(0, data["P"], size=data["P"])
        w = np.bincount(draw, minlength=data["P"]).astype(float)[data["pidx"]]
        ev = evaluate_sample(data, probs, w, b, init=init)
        for i, c in enumerate(cells):
            st["L_by"][b, i] = ev["L_by"][c]
            st["lam"][b, i] = ev["lam"][c]
        st["e_obs"][b], st["e_mod"][b] = ev["e_obs"], ev["e_mod"]
        st["c3"][b, 0], st["c3"][b, 1] = ev["C3_all"], ev["C3_seen"]
        st["share_looked"][b], st["share_chosen_looked"][b] = ev["share_looked"], ev["share_chosen_looked"]
        st["alone"][b] = np.vstack([ev["alone"][k] for k in ("all_items_all_trials", "all_items_C3_trials", "seen_items_C3_trials")])
        if (b + 1) % log_every == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B} done, {time.time() - t0:.1f}s elapsed", flush=True)
    return st


def summarize(point: dict, st: dict) -> tuple[dict, dict]:
    cells = [(f, v) for f in sc.FAMILIES for v in sc.VERSIONS]
    res: dict = {}
    extras: dict = {}
    # ---------------------------------------------------------------- C2
    c2cells = {}
    Lb_all = st["L_by"].mean(axis=3)  # (B, cells, betas): L is the plain mean over the four set sizes
    for i, (fam, ver) in enumerate(cells):
        L_pt = point["L_by"][(fam, ver)].mean(axis=1)
        for j, beta in enumerate(sc.BETAS):
            cell = {
                "L": sc.summary_level(L_pt[j], Lb_all[:, i, j]),
                "by_size": [float(v) for v in point["L_by"][(fam, ver)][j]],
                "lambda_held_out": [float(v) for v in point["lam"][(fam, ver)][j]],
            }
            if beta != 0.0:
                cell["minus_beta0"] = sc.summary_diff(L_pt[j] - L_pt[B0], Lb_all[:, i, j] - Lb_all[:, i, B0])
            c2cells[sc.cell_name(fam, beta, ver)] = cell
    names = (sc.cell_name("CL", 0.5, "all"), sc.cell_name("CL", -0.5, "all"))
    sign = {names[0]: 1, names[1]: -1}
    pv = [c2cells[n]["minus_beta0"]["p"] for n in names]
    ph = sc.holm(pv)
    primary = {}
    for n, p, padj in zip(names, pv, ph):
        d = c2cells[n]["minus_beta0"]
        direction = (d["percentile"][0] > 0 and d["basic"][0] > 0) if sign[n] > 0 else (d["percentile"][1] < 0 and d["basic"][1] < 0)
        primary[n] = {"p": p, "p_holm": padj, "supports_H1": bool(d["clear"] and direction and padj < 0.05)}
    res["C2"] = {"cells": c2cells, "primary": primary, "supported": bool(all(v["supports_H1"] for v in primary.values()))}
    extras["C2_percent_change_vs_beta0"] = {
        n: 100 * (2 ** c2cells[n]["minus_beta0"]["point"] - 1) for n in c2cells if "minus_beta0" in c2cells[n]
    }
    extras["C2_likelihood_vs_chance"] = {n: 2 ** c2cells[n]["L"]["point"] for n in c2cells}
    # ---------------------------------------------------------------- C1
    n_arr = np.array(sc.SET_SIZES, float)
    G = (point["e_obs"] - point["e_mod"]) / n_arr
    world = np.array([sc.effective_number_sampled(point["p_pos"][k], point["N_pos"][k], (2, 2, k, 0)) for k in range(4)])
    G0 = (world - point["e_mod"]) / n_arr
    err = (st["e_obs"] - st["e_mod"]) / n_arr - G0
    mean_gap = sc.summary_centered(G.mean(), err.mean(axis=1))
    slope = sc.summary_centered(float(sc.ls_slope(LOG2N, G)), sc.ls_slope(LOG2N, err))
    neg = lambda s: bool(s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0)  # noqa: E731
    res["C1"] = {
        "per_size": [
            {"set_size": n, "observed_share": float(point["e_obs"][k] / n), "model_share": float(point["e_mod"][k] / n), "gap": float(G[k])}
            for k, n in enumerate(sc.SET_SIZES)
        ],
        "mean_gap": mean_gap,
        "slope": slope,
        "supported": bool(neg(mean_gap) and neg(slope)),
    }
    extras["C1_world_gap"] = [float(g) for g in G0]
    extras["C1_model_position_share_of_uniform"] = [float(sc.effective_number(point["q_pos"][k]) / n) for k, n in enumerate(sc.SET_SIZES)]
    # ---------------------------------------------------------------- C3
    ba, la, _ = point["C3_all"]
    bs, ls_, _ = point["C3_seen"]
    bb_all, bb_seen = st["c3"][:, 0, 0], st["c3"][:, 1, 0]
    diff = sc.summary_diff(ba - bs, bb_all - bb_seen)
    seen_s = sc.summary_diff(bs, bb_seen)
    res["C3"] = {
        "beta_all": sc.summary_diff(ba, bb_all),
        "beta_seen": seen_s,
        "beta_all_minus_seen": diff,
        "lambda_all": float(la),
        "lambda_seen": float(ls_),
        "narrowing_supported": neg(diff),
        "noise_supported": neg(seen_s),
    }
    lo_b, hi_b = sc.BETA_BOUNDS
    extras["C3_replicates_at_beta_bound"] = int(((st["c3"][:, :, 0] <= lo_b) | (st["c3"][:, :, 0] >= hi_b)).sum())
    # ---------------------------------------------------------------- descriptives
    def per_size(pt, bt):
        return [dict(set_size=n, **sc.summary_point_pct(pt[k], bt[:, k])) for k, n in enumerate(sc.SET_SIZES)]

    res["descriptives"] = {
        "share_looked_at": per_size(point["share_looked"], st["share_looked"]),
        "share_chosen_looked_at": per_size(point["share_chosen_looked"], st["share_chosen_looked"]),
        "lambda_alone": {
            name: per_size(point["alone"][name], st["alone"][:, i])
            for i, name in enumerate(("all_items_all_trials", "all_items_C3_trials", "seen_items_C3_trials"))
        },
    }
    return res, extras


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    t0 = time.time()
    data = sc.load_data()
    probs = build_problems(data)
    point = evaluate_sample(data, probs, np.ones(len(data["choice"])), b=-1)
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    st = run_bootstrap(data, probs, point, args.B)
    np.savez_compressed(os.path.join(HERE, f"boot_snack_B{args.B}.npz"), **st)
    res, extras = summarize(point, st)
    extras["seconds"] = round(time.time() - t0, 1)
    extras["B"] = args.B
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(os.path.dirname(args.out), "extras.json" if args.out.endswith("results.json") else "extras_" + os.path.basename(args.out)), "w") as fh:
        json.dump(extras, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
