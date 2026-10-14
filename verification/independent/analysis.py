"""Run R0 and H1 for both tasks: point estimates, participant bootstrap, summaries.

Usage: python analysis.py --B 1000 [--out results.json] [--reuse]
The bootstrap arrays (log scores, overlaps, effective numbers; no counts) go to boot_B{B}.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MI = {m: i for i, m in enumerate(core.ALL_MODELS)}
SM = {m: i for i, m in enumerate(core.SOFTMAX_MODELS)}
CL0 = core.beta_name("CL", 0.0)


# ----------------------------------------------------------------------------- sample level
def point_sample(task: dict) -> dict:
    cnt = core.counts_from_weights(task, np.ones(task["P"]))
    ev = core.evaluate(task, cnt, include_full=True)
    q_cl = ev["Q_ho"][MI[CL0]]
    e_obs, e_mod = core.b1_components(task, cnt, q_cl, b=-1)
    e_world = core.b1_world_observed(task, cnt)
    return dict(cnt=cnt, ev=ev, e_obs=e_obs, e_mod=e_mod, e_world=e_world, Lstar=core.lstar(cnt, task["nvec"]))


def run_bootstrap(tasks: dict, points: dict, B: int, log_every: int = 25) -> dict:
    C = 6
    store = {
        name: dict(
            L=np.zeros((B, len(core.ALL_MODELS), C)),
            O=np.zeros((B, len(core.ALL_MODELS), C)),
            lam=np.zeros((B, len(core.SOFTMAX_MODELS), C)),
            eps=np.zeros((B, len(core.EBA_MODELS), C)),
            e_obs=np.zeros((B, C)),
            e_mod=np.zeros((B, C)),
        )
        for name in core.TASKS
    }
    t0 = time.time()
    for b in range(B):
        rng = np.random.default_rng(10000 + b)
        draws = {}
        for name in core.TASKS:  # file panel first, then laptop panel, same generator
            P = tasks[name]["P"]
            draws[name] = rng.integers(0, P, size=P)
        for name in core.TASKS:
            task = tasks[name]
            w = np.bincount(draws[name], minlength=task["P"]).astype(float)
            cnt = core.counts_from_weights(task, w)
            init = points[name]["ev"]["lam"][:, :C]
            ev = core.evaluate(task, cnt, include_full=False, init=init)
            s = store[name]
            s["L"][b] = ev["L_ho"]
            s["O"][b] = ev["O_ho"]
            s["lam"][b] = ev["lam"]
            s["eps"][b] = ev["eps"]
            e_obs, e_mod = core.b1_components(task, cnt, ev["Q_ho"][MI[CL0]], b=b)
            s["e_obs"][b] = e_obs
            s["e_mod"][b] = e_mod
        if (b + 1) % log_every == 0 or b + 1 == B:
            print(f"replicate {b + 1}/{B} done, {time.time() - t0:.1f}s elapsed", flush=True)
    return store


# ----------------------------------------------------------------------------- summaries
def trial_dict(names, values):
    return {n: float(v) for n, v in zip(names, values)}


def summarize_task(task: dict, pt: dict, bs: dict) -> dict:
    names = task["names"]
    nvec = task["nvec"]
    ev = pt["ev"]
    Ls = pt["Lstar"]
    C = len(names)
    low = task["info"] == "low"
    high = task["info"] == "high"
    size_mask = {s: task["sizes"] == s for s in (8, 16, 32)}
    L_ho, O_ho, L_fit = ev["L_ho"], ev["O_ho"], ev["L_fit"]
    Lb, Ob = bs["L"], bs["O"]

    K_ho = core.completeness(L_ho, Ls)  # (models,)
    K_b = core.completeness(Lb, Ls)  # (B, models), benchmark fixed (Amendment 1)
    Om = O_ho.mean(axis=1)
    Om_b = Ob.mean(axis=2)

    out: dict = {}
    out["rubric_shares"] = {a: float(v) for a, v in zip(core.ATTRS[task["task"]], task["r"])}
    cnt = pt["cnt"]
    out["shares"] = {
        t["name"]: {o: float(cnt[c, i] / cnt[c].sum()) for i, o in enumerate(t["names"])}
        for c, t in enumerate(task["trials"])
    }
    out["Lstar"] = trial_dict(names, Ls)

    # ------------------------------------------------------------------ R0
    r0 = {}
    i_cl = MI[CL0]
    for alias, model in core.R0_ALIAS.items():
        m = MI[model]
        sm = SM[model]
        r0[alias] = {
            "lambda": float(ev["lam"][sm, C]),
            "lambda_ho": trial_dict(names, ev["lam"][sm, :C]),
            "ho_bits": trial_dict(names, L_ho[m]),
            "fit_bits": trial_dict(names, L_fit[m]),
            "overlap_ho": trial_dict(names, O_ho[m]),
            "completeness_ho": core.summary_level(K_ho[m], K_b[:, m]),
            "completeness_ho_low": float(core.completeness(L_ho[m], Ls, low)),
            "completeness_ho_high": float(core.completeness(L_ho[m], Ls, high)),
            "completeness_fit": float(core.completeness(L_fit[m], Ls)),
            "overlap_ho_mean": float(Om[m]),
            "minus_CL": core.summary_diff(K_ho[m] - K_ho[i_cl], K_b[:, m] - K_b[:, i_cl]),
            "overlap_minus_CL": core.summary_diff(Om[m] - Om[i_cl], Om_b[:, m] - Om_b[:, i_cl]),
            "replicates": {str(k): float(K_b[k, m]) for k in (0, 1, 499, 999) if k < K_b.shape[0]},
        }
    out["R0"] = r0

    # ------------------------------------------------------------------ H1 A1a
    a1a = {}
    for base in ("CL", "DbS_avg", "MLBA_avg"):
        m0 = MI[core.beta_name(base, 0.0)]
        for beta in core.BETAS:
            name = core.beta_name(base, beta)
            m = MI[name]
            entry = {
                "completeness": core.summary_level(K_ho[m], K_b[:, m]),
                "by_size": {str(s): float(core.completeness(L_ho[m], Ls, mk)) for s, mk in size_mask.items()},
                "overlap": float(Om[m]),
            }
            if beta != 0.0:
                entry["minus_beta0"] = core.summary_diff(K_ho[m] - K_ho[m0], K_b[:, m] - K_b[:, m0])
                entry["overlap_minus_beta0"] = float(Om[m] - Om[m0])
            a1a[name] = entry
    prim_names = (core.beta_name("CL", 0.5), core.beta_name("CL", -0.5))
    predicted = {prim_names[0]: +1, prim_names[1]: -1}
    pv = [a1a[n]["minus_beta0"]["p"] for n in prim_names]
    ph = core.holm(pv)
    primary = {}
    for n, p, padj in zip(prim_names, pv, ph):
        d = a1a[n]["minus_beta0"]
        sign = predicted[n]
        clear_dir = d["clear"] and (
            (sign > 0 and d["percentile"][0] > 0 and d["basic"][0] > 0)
            or (sign < 0 and d["percentile"][1] < 0 and d["basic"][1] < 0)
        )
        overlap_same = bool(np.sign(a1a[n]["overlap_minus_beta0"]) == sign)
        primary[n] = {
            "p": float(p),
            "p_holm": float(padj),
            "supports_H1": bool(clear_dir and padj < 0.05),
            "overlap_same_way": overlap_same,
        }
    a1a["primary"] = primary
    a1a["supported"] = bool(all(v["supports_H1"] and v["overlap_same_way"] for v in primary.values()))

    # ------------------------------------------------------------------ H1 A1b
    a1b = {}
    ssmall, slarge = size_mask[8], size_mask[32]
    for e_i, ename in enumerate(core.EBA_MODELS):
        m = MI[ename]
        delta = L_ho[m] - L_ho[i_cl]  # bits, held out
        delta_b = Lb[:, m] - Lb[:, i_cl]
        contrast = delta[slarge].mean() - delta[ssmall].mean()
        contrast_b = delta_b[:, slarge].mean(axis=1) - delta_b[:, ssmall].mean(axis=1)
        a1b[ename] = {
            "lapse": float(ev["eps"][e_i, C]),
            "lapse_ho": trial_dict(names, ev["eps"][e_i, :C]),
            "completeness": core.summary_level(K_ho[m], K_b[:, m]),
            "minus_CL": core.summary_diff(K_ho[m] - K_ho[i_cl], K_b[:, m] - K_b[:, i_cl]),
            "delta_bits": trial_dict(names, delta),
            "delta_bits_by_size": {str(s): float(delta[mk].mean()) for s, mk in size_mask.items()},
            "large_minus_small_bits": core.summary_diff(contrast, contrast_b),
        }
    lms = a1b["EBA"]["large_minus_small_bits"]
    a1b["supported"] = bool(lms["clear"] and lms["percentile"][0] > 0 and lms["basic"][0] > 0)

    # ------------------------------------------------------------------ H1 B1
    x = np.log2(nvec)
    G = (pt["e_obs"] - pt["e_mod"]) / nvec
    G0 = (pt["e_world"] - pt["e_mod"]) / nvec
    Gs = (bs["e_obs"] - bs["e_mod"]) / nvec
    err = Gs - G0
    mean_gap = core.summary_centered(G.mean(), err.mean(axis=1))
    slope = core.summary_centered(float(core.ls_slope(x, G)), core.ls_slope(x, err))
    neg = lambda s: bool(s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0)  # noqa: E731
    b1 = {
        "per_trial": {
            n: {
                "observed_share": float(pt["e_obs"][c] / nvec[c]),
                "model_share": float(pt["e_mod"][c] / nvec[c]),
                "gap": float(G[c]),
                "world_gap": float(G0[c]),
            }
            for c, n in enumerate(names)
        },
        "mean_gap": mean_gap,
        "slope": slope,
        "supported": bool(neg(mean_gap) and neg(slope)),
    }
    out["H1"] = {"A1a": a1a, "A1b": a1b, "B1": b1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--laptop-stimuli", default="laptop_trials.json", help="laptop definition file in verify/stimuli")
    ap.add_argument("--tag", default="", help="suffix of the bootstrap archive, boot_B{B}{tag}.npz")
    args = ap.parse_args()
    t0 = time.time()
    stim = {"file": None, "laptop": args.laptop_stimuli}
    tasks = {name: core.build_task(name, stimuli_file=stim[name]) for name in core.TASKS}
    points = {name: point_sample(tasks[name]) for name in core.TASKS}
    print(f"point estimates done in {time.time() - t0:.1f}s (laptop definitions {args.laptop_stimuli})", flush=True)
    npz = os.path.join(HERE, f"boot_B{args.B}{args.tag}.npz")
    if args.reuse and os.path.exists(npz):
        raw = np.load(npz)
        store = {n: {k: raw[f"{n}__{k}"] for k in ("L", "O", "lam", "eps", "e_obs", "e_mod")} for n in core.TASKS}
    else:
        store = run_bootstrap(tasks, points, args.B)
        np.savez_compressed(npz, **{f"{n}__{k}": v for n, s in store.items() for k, v in s.items()})
    results = {name: summarize_task(tasks[name], points[name], store[name]) for name in core.TASKS}
    results["_meta"] = {
        "B": args.B,
        "seconds": round(time.time() - t0, 1),
        "laptop_stimuli": args.laptop_stimuli,
        "notes": "Independent reimplementation; completeness in percent, overlap and shares as fractions, log scores in bits.",
    }
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
