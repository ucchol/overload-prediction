"""Overload, choices, time and premonition (VERIFY_BRIEF_PROCESS.md, sections A to F).

Usage: python -I run_process.py [--B 1000] [--out results.json]
Bootstrap arrays (no counts) go to boot_process_B{B}.npz, fit diagnostics to diagnostics_B{B}.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovl_answers as ov  # noqa: E402
import process_core as pc  # noqa: E402
from process_core import core  # noqa: E402

pc.assert_local(ov, pc, core)

HERE = os.path.dirname(os.path.abspath(__file__))
MI = {m: i for i, m in enumerate(core.ALL_MODELS)}
SM = {m: i for i, m in enumerate(core.SOFTMAX_MODELS)}
R0_MODELS = {"QR": "QR", "CL": "CL|beta=+0.0", "DbS_avg": "DbS_avg|beta=+0.0", "MLBA_avg": "MLBA_avg|beta=+0.0",
             "DbS_sum": "DbS_sum", "MLBA_sum": "MLBA_sum", "EBA": "EBA"}
CHOICE_MODELS = list(R0_MODELS) + list(pc.SEARCH_MODELS)
DIFFS = [("SR", "CL"), ("SR_DbS", "DbS_avg"), ("SR_MLBA", "MLBA_avg"), ("Accept_read", "CL"),
         ("SR", "DbS_avg"), ("SR", "MLBA_avg"), ("SR_DbS", "CL"), ("SR_DbS", "MLBA_avg")]
TIME_MODELS = ["QR", "CL", "DbS_avg", "MLBA_avg", "EBA", "SR", "SR_DbS", "Accept_read"]
ENTROPY_MODELS = ["QR", "CL", "DbS_avg", "MLBA_avg", "SR", "SR_DbS"]
FORECASTS = (["options", "attributes", "cells", "looks_SR"] + [f"entropy_{m}" for m in ENTROPY_MODELS]
             + [f"effort_{m}" for m in ("CL", "DbS_avg", "EBA", "SR")])
FSTATS = ("spearman", "error", "constant_error", "error_minus_constant")
REGRET_MODELS = ["QR", "CL", "DbS_avg", "MLBA_avg", "EBA", "SR", "SR_DbS"]
RSTATS = ("error", "constant_error", "error_minus_constant", "spearman")
SCOPES = ("file", "laptop", "both")
COLS = {"file": slice(0, 6), "laptop": slice(6, 12)}


# ============================================================================ one task, one sample
def evaluate_task(task, models, fixed, logt, index, regret, w, Ls, init=None):
    """Every choice, time and premonition input of one task for one sample (w: weight per participant)."""
    cnt = core.counts_from_weights(task, w)
    ev = core.evaluate(task, cnt, include_full=False, init=init)
    sh = pc.search_heldout(models, cnt)
    C = cnt.shape[0]
    nvec = task["nvec"]
    # ---------------------------------------------------------------- choices
    Q = {k: ev["Q_ho"][MI[m]] for k, m in R0_MODELS.items()}
    lam = {k: (ev["lam"][SM[m]] if k != "EBA" else ev["eps"][0]) for k, m in R0_MODELS.items()}
    for k in pc.SEARCH_MODELS:
        Q[k] = sh[k]["Q"]
        lam[k] = sh[k]["lam"]
    bits = {k: core.log_scores(Q[k][None], cnt, nvec)[0] for k in CHOICE_MODELS}
    K = {k: float(core.completeness(bits[k], Ls)) for k in CHOICE_MODELS}
    # ---------------------------------------------------------------- effort and time
    m = fixed["m"]
    effort = {k: fixed[k] for k in ("QR", "CL", "DbS_avg", "MLBA_avg", "EBA")}
    effort["SR"] = sh["SR"]["extra"] * m
    effort["SR_DbS"] = sh["SR_DbS"]["extra"] * m
    effort["Accept_read"] = sh["Accept_read"]["extra"] * m
    typical = np.exp((w[:, None] * logt).sum(axis=0) / w.sum())
    pred, logerr = {}, {}
    for k in TIME_MODELS:
        pred[k], logerr[k] = pc.time_law(effort[k], typical)
    pred["constant"], logerr["constant"] = pc.constant_law(typical)
    # ---------------------------------------------------------------- premonition inputs
    ov_target = (w[:, None] * index).sum(axis=0) / w.sum()
    rg_target = (w[:, None] * regret).sum(axis=0) / w.sum()
    fc = {"options": nvec, "attributes": m, "cells": nvec * m, "looks_SR": sh["SR"]["extra"]}
    for k in ENTROPY_MODELS:
        fc[f"entropy_{k}"] = np.array([pc.entropy_bits(Q[k][c, : int(nvec[c])]) for c in range(C)])
    for k in ("CL", "DbS_avg", "EBA", "SR"):
        fc[f"effort_{k}"] = effort[k]
    rfc = {k: np.array([float(np.sum(Q[k][c, : int(nvec[c])] * (100.0 - task["trials"][c]["U"]))) for c in range(C)])
           for k in REGRET_MODELS}
    diag = np.array([[int((sh[k]["n_local"] > 1).sum()), int((~sh[k]["refined"]).sum()), int((sh[k]["lam"] == 0).sum())]
                     for k in pc.SEARCH_MODELS])
    return dict(K=K, bits=bits, lam=lam, effort=effort, typical=typical, pred=pred, logerr=logerr,
                ov_target=ov_target, rg_target=rg_target, fc=fc, rfc=rfc, diag=diag, init=ev["lam"])


def task_flat(r):
    """Fixed-order vectors of one task's replicate quantities, for the bootstrap arrays."""
    return {
        "K": np.array([r["K"][k] for k in CHOICE_MODELS]),
        "typical": r["typical"],
        "sps": np.array([r["typical"] / r["effort"][k] for k in TIME_MODELS]),
        "meanlog": np.array([np.mean(r["logerr"][k]) for k in TIME_MODELS + ["constant"]]),
        "lam_search": np.array([r["lam"][k] for k in pc.SEARCH_MODELS]),
        "diag": r["diag"],
    }


# ============================================================================ scopes (sections E and F)
def scope_inputs(by_task: dict, scope: str):
    parts = [by_task["file"], by_task["laptop"]] if scope == "both" else [by_task[scope]]
    fc = {f: np.concatenate([np.asarray(p["fc"][f], float) for p in parts]) for f in FORECASTS}
    rfc = {k: np.concatenate([p["rfc"][k] for p in parts]) for k in REGRET_MODELS}
    ov_t = np.concatenate([p["ov_target"] for p in parts])
    rg_t = np.concatenate([p["rg_target"] for p in parts])
    return fc, rfc, ov_t, rg_t


def scope_flat(by_task: dict, scope: str):
    fc, rfc, ov_t, rg_t = scope_inputs(by_task, scope)
    fs = {f: pc.forecast_stats(fc[f], ov_t) for f in FORECASTS}
    rs = {k: pc.regret_stats(rfc[k], rg_t) for k in REGRET_MODELS}
    return {
        "fstats": np.array([[fs[f][s] for s in FSTATS] for f in FORECASTS]),
        "rstats": np.array([[rs[k][s] for s in RSTATS] for k in REGRET_MODELS]),
        "fc": np.array([fc[f] for f in FORECASTS]),
        "rfc": np.array([rfc[k] for k in REGRET_MODELS]),
        "ov_target": ov_t,
        "rg_target": rg_t,
    }


# ============================================================================ summaries
def pp(t, v):
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": float(t), "percentile": [float(lo), float(hi)]}


def ppbp(t, v):
    d = pp(t, v)
    d["basic"] = [float(2 * t - d["percentile"][1]), float(2 * t - d["percentile"][0])]
    d["p"] = core.boot_p(v)
    return d


def tdict(names, vals):
    return {n: float(v) for n, v in zip(names, vals)}


def summarize_task(names, pt, bs):
    out_c, out_t = {}, {}
    Kp = task_flat(pt)["K"]
    ci = {k: i for i, k in enumerate(CHOICE_MODELS)}
    out_c["completeness"] = {k: core.summary_diff(Kp[i], bs["K"][:, i]) for i, k in enumerate(CHOICE_MODELS)}
    out_c["differences"] = {f"{a}-{b}": core.summary_diff(Kp[ci[a]] - Kp[ci[b]], bs["K"][:, ci[a]] - bs["K"][:, ci[b]])
                            for a, b in DIFFS}
    out_c["lambda_held_out"] = {k: tdict(names, pt["lam"][k]) for k in CHOICE_MODELS if k != "EBA"}
    out_c["lambda_held_out"]["EBA_lapse"] = tdict(names, pt["lam"]["EBA"])
    out_c["ho_bits"] = {k: tdict(names, pt["bits"][k]) for k in CHOICE_MODELS}
    out_t["typical_seconds"] = {n: pp(pt["typical"][c], bs["typical"][:, c]) for c, n in enumerate(names)}
    mods = {}
    for i, k in enumerate(TIME_MODELS):
        mods[k] = {
            "effort": tdict(names, pt["effort"][k]),
            "seconds_per_step": {n: pp(pt["typical"][c] / pt["effort"][k][c], bs["sps"][:, i, c]) for c, n in enumerate(names)},
            "predicted_seconds": tdict(names, pt["pred"][k]),
            "typical_error": core.summary_diff(pc.typical_error(pt["logerr"][k]), 100 * (np.exp(bs["meanlog"][:, i]) - 1)),
            "log_error_minus_constant": core.summary_diff(np.mean(pt["logerr"][k]) - np.mean(pt["logerr"]["constant"]),
                                                          bs["meanlog"][:, i] - bs["meanlog"][:, -1]),
        }
    mods["constant"] = {
        "predicted_seconds": tdict(names, pt["pred"]["constant"]),
        "typical_error": core.summary_diff(pc.typical_error(pt["logerr"]["constant"]), 100 * (np.exp(bs["meanlog"][:, -1]) - 1)),
    }
    out_t["models"] = mods
    return out_c, out_t


def summarize_scope(names, pt, bs):
    out = {
        "overload": {n: pp(pt["ov_target"][c], bs["ov_target"][:, c]) for c, n in enumerate(names)},
        "regret": {n: pp(pt["rg_target"][c], bs["rg_target"][:, c]) for c, n in enumerate(names)},
        "forecasts": {},
    }
    for i, f in enumerate(FORECASTS):
        entry = {"values": {n: pp(pt["fc"][i, c], bs["fc"][:, i, c]) for c, n in enumerate(names)}}
        for j, s in enumerate(FSTATS):
            entry[s] = core.summary_diff(pt["fstats"][i, j], bs["fstats"][:, i, j])
        out["forecasts"][f] = entry
    out["regret_forecasts"] = {k: {n: pp(pt["rfc"][i, c], bs["rfc"][:, i, c]) for c, n in enumerate(names)}
                               for i, k in enumerate(REGRET_MODELS)}
    for j, s in enumerate(RSTATS):
        out[f"regret_{s}"] = {k: core.summary_diff(pt["rstats"][i, j], bs["rstats"][:, i, j]) for i, k in enumerate(REGRET_MODELS)}
    return out


def summarize_overload(pt, boots, names):
    res = {"measurement": {}}
    for k in ("alpha_within_person", "alpha_raw", "first_component_share"):
        res["measurement"][k] = ppbp(pt[k], np.array([b[k] for b in boots]))
    q6b = np.array([b["q6"] for b in boots])
    res["measurement"]["question6_raw_correlations"] = {f"s{j + 1}": pp(pt["q6"][j], q6b[:, j]) for j in range(5)}
    arr = {k: np.array([b[k] for b in boots]) for k in ("overload", "overload_vs_average", "regret", "best_chosen", "geo_seconds")}
    res["per_trial"] = {n: {k: (ppbp if k == "overload_vs_average" else pp)(pt[k][c], arr[k][:, c]) for k in arr}
                        for c, n in enumerate(names)}
    res["contrasts"] = {}
    for key in pt["contrasts"]:
        task, q, c = key
        qname = "log_seconds_percent" if q == "log_seconds" else q
        res["contrasts"].setdefault(task, {}).setdefault(qname, {})[c] = ppbp(pt["contrasts"][key], np.array([b["contrasts"][key] for b in boots]))
    res["associations"] = {}
    for key in pt["associations"]:
        s, a = key
        res["associations"].setdefault(s, {})[a] = ppbp(pt["associations"][key], np.array([b["associations"][key] for b in boots]))
    return res


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()
    B = args.B
    t0 = time.time()
    tasks = {n: core.build_task(n) for n in core.TASKS}  # storyboard laptop definitions (stimuli/laptop_trials.json)
    S, regret, seconds = ov.load_answers()
    N = S.shape[0]
    if any(tasks[n]["P"] != N for n in core.TASKS):
        raise RuntimeError("an R0 panel differs from the participant rows")
    index = ov.overload_index(S)
    logt = np.log(seconds)
    models = {n: {k: pc.SearchModel(tasks[n], k) for k in pc.SEARCH_MODELS} for n in core.TASKS}
    fixed = {n: pc.fixed_efforts(tasks[n]) for n in core.TASKS}
    Ls = {n: core.lstar(core.counts_from_weights(tasks[n], np.ones(N)), tasks[n]["nvec"]) for n in core.TASKS}
    names = {n: tasks[n]["names"] for n in core.TASKS}
    names["both"] = names["file"] + names["laptop"]

    def one_sample(ws, init=None):
        by_task = {n: evaluate_task(tasks[n], models[n], fixed[n], logt[:, COLS[n]], index[:, COLS[n]],
                                    regret[:, COLS[n]], ws[n], Ls[n], init=None if init is None else init[n])
                   for n in core.TASKS}
        return by_task, {s: scope_flat(by_task, s) for s in SCOPES}

    pts, pscopes = one_sample({n: np.ones(N) for n in core.TASKS})
    init = {n: pts[n]["init"] for n in core.TASKS}
    print(f"point estimates done in {time.time() - t0:.1f}s", flush=True)
    # ---------------------------------------------------------------- R0 bootstrap (choices, time, premonition)
    store = {n: {} for n in core.TASKS}
    sstore = {s: {} for s in SCOPES}
    for b in range(B):
        rng = np.random.default_rng(10000 + b)
        draws = {n: rng.integers(0, N, size=N) for n in core.TASKS}  # file participants first, then laptop
        ws = {n: np.bincount(draws[n], minlength=N).astype(float) for n in core.TASKS}
        by_task, scopes = one_sample(ws, init)
        for n in core.TASKS:
            for k, v in task_flat(by_task[n]).items():
                store[n].setdefault(k, np.zeros((B,) + np.shape(v)))[b] = v
        for s in SCOPES:
            for k, v in scopes[s].items():
                sstore[s].setdefault(k, np.zeros((B,) + np.shape(v)))[b] = v
        if (b + 1) % 25 == 0 or b + 1 == B:
            print(f"R0 replicate {b + 1}/{B}, {time.time() - t0:.1f}s", flush=True)
    # ---------------------------------------------------------------- overload bootstrap
    ov_pt = ov.analyse(S, regret, seconds)
    ov_boot = []
    for b in range(B):
        idx = np.random.default_rng(30000 + b).integers(0, N, size=N)
        ov_boot.append(ov.analyse(S[idx], regret[idx], seconds[idx]))
    print(f"overload bootstrap done, {time.time() - t0:.1f}s", flush=True)
    np.savez_compressed(os.path.join(HERE, f"boot_process_B{B}.npz"),
                        **{f"{n}__{k}": v for n, s in store.items() for k, v in s.items()},
                        **{f"scope_{s}__{k}": v for s, d in sstore.items() for k, v in d.items()})
    # ---------------------------------------------------------------- results
    results = {"overload": summarize_overload(ov_pt, ov_boot, ov.trial_names()), "choices": {}, "time": {}, "premonition": {}}
    for n in core.TASKS:
        results["choices"][n], results["time"][n] = summarize_task(names[n], pts[n], store[n])
    for s in SCOPES:
        results["premonition"][s] = summarize_scope(names[s], pscopes[s], sstore[s])
    diag = {}
    for n in core.TASKS:
        tot = store[n]["diag"].sum(axis=0)  # (search models, 3) summed over replicates
        lam = store[n]["lam_search"]
        diag[n] = {k: {"replicate_folds_with_several_grid_maxima": int(tot[i, 0]),
                       "replicate_folds_not_refined": int(tot[i, 1]),
                       "replicate_folds_with_lambda_zero": int(tot[i, 2]),
                       "replicate_lambda_range": [float(lam[:, i].min()), float(lam[:, i].max())],
                       "original_sample": {"folds_with_several_grid_maxima": int(pts[n]["diag"][i, 0]),
                                           "folds_not_refined": int(pts[n]["diag"][i, 1])}}
                   for i, k in enumerate(pc.SEARCH_MODELS)}
    with open(os.path.join(HERE, f"diagnostics_B{B}.json"), "w") as fh:
        json.dump(diag, fh, indent=1)
    results["_meta"] = {"B": B, "seconds": round(time.time() - t0, 1), "laptop_stimuli": "laptop_trials.json"}
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
