"""Runner of the blind MXL check: open-data panels (six models, held out and fitted) and the development data
(MXL only). Writes out.json. Original sample, every trial weighted one. Prints no identifiers or counts."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(VERIFY, "independent_overload"))
sys.path.insert(2, os.path.join(VERIFY, "independent_snack"))
sys.path.insert(3, os.path.join(VERIFY, "independent"))
import mxl_core as mc  # noqa: E402
import ov_core as oc  # noqa: E402
import snack_core as snk  # noqa: E402
import core as dev  # noqa: E402  (R0 implementation: development data, rubric shares, CL scores)

cc = mc.cc
MODELS = ("CL", "CL_comp", "SR", "DbS", "MLBA", "MXL")
MLBA_SCALE = {"cdm": 6.0 / 35.0, "reutskaja": 0.6, "snack": 1.0}


def blocks_of(name, items, scale):
    out = []
    for n, idx, V, c in items:
        out.append(dict(n=int(n), idx=idx, V=V, chosen=np.asarray(c), S_CL=V, S_CL_comp=oc.f_comp(n) * V,
                        S_DbS=cc.dbs_scores(V), S_MLBA=cc.mlba_scores(V, scale=scale)))
    return out


def open_panels():
    panels, _ = oc.build(os.path.join(cc.CDM_DIR, cc.CLICK_FILE), os.path.join(oc.REUT_DIR, oc.REUT_FILE), allow_real=True)
    out = []
    for P in panels:
        ds = "reutskaja" if P["name"] == "reutskaja" else "cdm"
        out.append(dict(name=P["name"], blocks=blocks_of(P["name"], P["items"], MLBA_SCALE[ds]), person=P["pidx"],
                        T=P["T"], P=int(P["pidx"].max()) + 1))
    D = snk.load_data()
    items, start = [], 0
    for k, n in enumerate(snk.SET_SIZES):
        V = D["V_by_size"][k]
        idx = np.arange(start, start + len(V))
        items.append((n, idx, V, D["choice"][idx]))
        start += len(V)
    out.append(dict(name="snack", blocks=blocks_of("snack", items, MLBA_SCALE["snack"]), person=D["pidx"], T=start, P=int(D["P"])))
    return out


def predict(model, lam, b):
    if model == "SR":
        return mc.sr_probs(lam, b["V"])
    if model == "MXL":
        return mc.mxl_probs(lam[0], lam[1], b["S_CL"])
    return mc.softmax_probs(lam, b[f"S_{model}"])


def fit(model, blocks, w, mx, keep):
    if model == "SR":
        lam, excess = mc.fit_sr(blocks, w)
        return lam, {"dense_grid_excess": excess}
    if model == "MXL":
        f = mx.fit(keep)
        return (f["mu"], f["sigma"]), {k: f[k] for k in ("negll", "n_local_minima", "coarse_min", "sigma_at_bound")}
    return fit_key(model, blocks, w), {}


def fit_key(model, blocks, w):
    return mc.fit_softmax(blocks, f"S_{model}", w)


def run_panel(pn):
    blocks, T = pn["blocks"], pn["T"]
    mx = mc.MXL(blocks, pn["person"], pn["P"])
    sizes = [b["n"] for b in blocks]
    x = np.log2(np.array(sizes, float))
    top_mask = [b["V"] == b["V"].max(axis=1, keepdims=True) for b in blocks]
    obs = [float(np.mean(b["V"][np.arange(len(b["chosen"])), b["chosen"]] < b["V"].max(axis=1))) for b in blocks]
    res = {"sizes": sizes, "observed_miss": obs, "observed_growth": float(np.polyfit(x, obs, 1)[0]), "models": {}}
    diag = {}
    for model in MODELS:
        entry = {}
        for mode in ("held_out", "fitted"):
            miss, lp, params, info_all = [], [], [], []
            if mode == "fitted":
                p, info = fit(model, blocks, np.ones(T), mx, [True] * len(blocks))
                params, info_all = [p], [info]
            for k, b in enumerate(blocks):
                if mode == "held_out":
                    w = np.ones(T)
                    w[b["idx"]] = 0.0
                    p, info = fit(model, blocks, w, mx, [j != k for j in range(len(blocks))])
                    params.append(p)
                    info_all.append(info)
                Q = predict(model, p, b)
                miss.append(float(np.mean(1.0 - (Q * top_mask[k]).sum(axis=1))))
                qc = Q[np.arange(len(b["chosen"])), b["chosen"]]
                lp.append(np.log2(np.maximum(qc, 1e-300)))
            allp = np.concatenate(lp)
            e = {"miss_by_size": miss, "growth": float(np.polyfit(x, miss, 1)[0]),
                 "bits": float(allp.mean()), "bits_by_size": [float(v.mean()) for v in lp]}
            if model == "MXL":
                e["mu_sigma"] = [[float(a), float(s)] for a, s in params]
            else:
                e["lambda"] = [float(v) for v in params]
            entry[mode] = e
            diag[f"{model}_{mode}"] = info_all
        res["models"][model] = entry
    return res, diag


def dev_task(tname):
    t = dev.build_task(tname)
    ci = dev.SOFTMAX_MODELS.index("CL|beta=+0.0")
    blocks, person, start = [], [], 0
    for c, tr in enumerate(t["trials"]):
        n = tr["n"]
        rows = np.flatnonzero(t["Y"][:, c, :n].sum(axis=1) > 0)
        chosen = t["Y"][rows, c, :n].argmax(axis=1)
        S = np.tile(t["V"][ci, c, :n], (len(rows), 1))
        idx = np.arange(start, start + len(rows))
        blocks.append(dict(n=n, idx=idx, S_CL=S, chosen=chosen, U=tr["U"]))
        person.append(rows)
        start += len(rows)
    person = np.concatenate(person)
    mx = mc.MXL(blocks, person, t["P"])
    C = len(blocks)
    out = {"displays": t["names"], "rubric_shares": {a: float(v) for a, v in zip(dev.ATTRS[tname], t["r"])}}
    diag = {}
    for mode in ("fitted", "held_out"):
        ms, miss, bits = [], [], []
        f_all = mx.fit([True] * C) if mode == "fitted" else None
        for k, b in enumerate(blocks):
            f = f_all if mode == "fitted" else mx.fit([j != k for j in range(C)])
            if mode == "held_out" or k == 0:
                ms.append([f["mu"], f["sigma"]])
                diag[f"{mode}_{k}"] = {kk: f[kk] for kk in ("negll", "n_local_minima", "coarse_min", "sigma_at_bound")}
            q = mc.mxl_probs(f["mu"], f["sigma"], b["S_CL"][:1])[0]
            top = b["U"] == b["U"].max()
            miss.append(float(1.0 - q[top].sum()))
            bits.append(float(np.mean(np.log2(b["n"] * np.maximum(q[b["chosen"]], 1e-300)))))
        out[mode] = {"mu_sigma": ms, "miss": miss, "bits_above_uniform": bits}
    return out, diag


def main():
    t0 = time.time()
    out = {"open": {}, "development": {}}
    diag = {"open": {}, "development": {}}
    for pn in open_panels():
        out["open"][pn["name"]], diag["open"][pn["name"]] = run_panel(pn)
        print(f"{pn['name']} done, {time.time() - t0:.0f}s", flush=True)
    for tname in ("file", "laptop"):
        out["development"][tname], diag["development"][tname] = dev_task(tname)
        print(f"development {tname} done, {time.time() - t0:.0f}s", flush=True)
    out["_meta"] = {"seconds": round(time.time() - t0, 1)}
    with open(os.path.join(HERE, "out.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(HERE, "diag.json"), "w") as fh:
        json.dump(diag, fh, indent=1, default=float)
    print("wrote out.json", flush=True)


if __name__ == "__main__":
    main()
