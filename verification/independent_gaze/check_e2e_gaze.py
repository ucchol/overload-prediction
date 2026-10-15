"""Brief test 3 on synthetic files in which choices come from two look satisficing, the output layout of
VERIFY_BRIEF_GAZE.md, and the original-sample values recomputed from the files and the reported lambdas
(within-participant correlations with scipy's spearmanr, independently of the grouped rank code).

Usage: python -I check_e2e_gaze.py RESULTS_JSON TRUTH_JSON BOOT_NPZ
(the synthetic files are the folder of TRUTH_JSON, the diagnostics diag_<stem>.json next to RESULTS_JSON)
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gaze_core as gc  # noqa: E402
import run_gaze as rg  # noqa: E402
from gaze_core import sc, srch  # noqa: E402

srch.assert_local(gc, rg, srch, sc)
res = json.load(open(sys.argv[1]))
truth = json.load(open(sys.argv[2]))
DATA = os.path.dirname(os.path.abspath(sys.argv[2]))
assert os.path.realpath(DATA) != gc.GAZE_DIR, "the end-to-end check runs on synthetic files only"
boot = np.load(sys.argv[3])
stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]
diag = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), f"diag_{stem}.json")))
NAMES = gc.DATASETS
MODELS = ("CL", "DbS", "MLBA", "SR")
FEATURES = ("top_value", "mean_value", "top_gap")
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


# ----------------------------------------------------------------------------- layout
S_KEYS = {"point", "percentile", "basic", "p", "clear"}
BLOCK = {"time": {"static", "SR", "TLS", "TLS_minus_SR", "TLS_minus_static"},
         "slow": {"looks_TLS", "looks_SR", *FEATURES, "TLS_minus_SR", *(f"TLS_plus_{f}" for f in FEATURES)},
         "choices": {"times_chance", "TLS_minus_CL_percent", "TLS_minus_DbS_percent", "TLS_minus_MLBA_percent"}}


def is_S(x):
    return (isinstance(x, dict) and set(x) == S_KEYS and len(x["percentile"]) == 2 and len(x["basic"]) == 2
            and isinstance(x["clear"], bool) and 0.0 <= x["p"] <= 1.0)


def block_ok(b):
    ok = all(set(b[k]) == v for k, v in BLOCK.items())
    ok &= all(is_S(b["time"][k]) for k in BLOCK["time"]) and all(is_S(b["slow"][k]) for k in BLOCK["slow"])
    ok &= set(b["choices"]["times_chance"]) == set(MODELS) and all(isinstance(b["choices"]["times_chance"][m], float) for m in MODELS)
    return ok and all(is_S(b["choices"][k]) for k in BLOCK["choices"] - {"times_chance"})


p = res["pooled"]
lay = set(res) == {"pooled", *NAMES} and set(p) == set(BLOCK) | {"G1", "G2", "G3"} and block_ok(p)
lay &= set(p["G1"]) == {"supported"} and set(p["G2"]) == {"supported"}
lay &= set(p["G3"]) == {"margin", "noninferior", "better", "worse", "supported"} and p["G3"]["margin"] == 2.0
for n in NAMES:
    d = res[n]
    extra = {"lambda", "law", "looks", "typical_seconds"} | ({"confidence"} if n == "folke2016" else set())
    lay &= set(d) == set(BLOCK) | extra and block_ok(d)
    lay &= set(d["lambda"]) == set(MODELS) and all(len(d["lambda"][m]) == gc.N_FOLDS for m in MODELS)
    lay &= set(d["law"]) == {"SR", "TLS"} and all(set(d["law"][m]) == {"t0", "tau"} and is_S(d["law"][m]["t0"]) and is_S(d["law"][m]["tau"])
                                                 for m in ("SR", "TLS"))
    lay &= set(d["looks"]) == {"SR", "TLS"} and all(is_S(d["looks"][m]) for m in ("SR", "TLS")) and is_S(d["typical_seconds"])
    if n == "folke2016":
        lay &= set(d["confidence"]) == {"looks_TLS", "looks_SR", "entropy_CL", "TLS_minus_SR"} and all(is_S(v) for v in d["confidence"].values())
check("output layout as in the brief: pooled block with G1 to G3; per dataset the block, lambda by fold for CL, DbS, MLBA and SR, "
      "law (t0, tau) for SR and TLS, looks, typical_seconds; Folke's confidence", lay)

B = boot[f"{NAMES[0]}__err"].shape[0]
shapes = {"lam": (B, 4, gc.N_FOLDS), "err": (B, 3), "corr": (B, 5), "L": (B, 4), "law": (B, 2, 2), "looks": (B, 2),
          "typical": (B,), "lam_all": (B,), "dense_excess": (B,)}
ok = set(boot.files) == {f"{n}__{k}" for n in NAMES for k in shapes} | {"folke2016__conf"} and boot["folke2016__conf"].shape == (B, 3)
ok &= all(boot[f"{n}__{k}"].shape == s for n in NAMES for k, s in shapes.items())
check("bootstrap arrays hold only fitted values and measures, with the expected shapes", ok)

# ----------------------------------------------------------------------------- internal consistency of the reported values
tol = 1e-12
pt = lambda n, sec, k: res[n][sec][k]["point"]  # noqa: E731
ok = all(abs(pt("pooled", sec, k) - np.mean([pt(n, sec, k) for n in NAMES])) < tol for sec in ("time", "slow") for k in BLOCK[sec])
tc = {n: res[n]["choices"]["times_chance"] for n in ("pooled",) + NAMES}
ok &= all(abs(np.log2(tc["pooled"][m]) - np.mean([np.log2(tc[n][m]) for n in NAMES])) < tol for m in MODELS)
for n in ("pooled",) + NAMES:
    ok &= all(abs(res[n]["choices"][f"TLS_minus_{m}_percent"]["point"] - 100 * (tc[n]["SR"] / tc[n][m] - 1)) < 1e-9 for m in ("CL", "DbS", "MLBA"))
    s, t = res[n]["slow"], res[n]["time"]
    ok &= all(abs(s[f"TLS_plus_{f}"]["point"] - s["looks_TLS"]["point"] - s[f]["point"]) < tol for f in FEATURES)
    ok &= abs(s["TLS_minus_SR"]["point"] - s["looks_TLS"]["point"] + s["looks_SR"]["point"]) < tol
    ok &= abs(t["TLS_minus_SR"]["point"] - t["TLS"]["point"] + t["SR"]["point"]) < tol
    ok &= abs(t["TLS_minus_static"]["point"] - t["TLS"]["point"] + t["static"]["point"]) < tol
check("pooled values are the means over the datasets (time errors, correlations, log2 of times chance); "
      "sums, differences and percents agree with their parts", ok)


def clear_pos(s):
    return s["clear"] and s["percentile"][0] > 0 and s["basic"][0] > 0


def clear_neg(s):
    return s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0


t, s, c3 = p["time"], p["slow"], p["choices"]["TLS_minus_CL_percent"]
dec = (p["G1"]["supported"] == (clear_neg(t["TLS_minus_SR"]) and clear_neg(t["TLS_minus_static"]))
       and p["G2"]["supported"] == (clear_pos(s["looks_TLS"]) and clear_pos(s["TLS_minus_SR"]))
       and p["G3"]["noninferior"] == (min(c3["percentile"][0], c3["basic"][0]) > -2.0)
       and p["G3"]["better"] == (c3["percentile"][0] > 0 and c3["basic"][0] > 0)
       and p["G3"]["worse"] == (c3["percentile"][1] < 0 and c3["basic"][1] < 0)
       and p["G3"]["supported"] == p["G3"]["noninferior"])
check("G1 to G3 decisions follow the stated rules from the reported summaries", dec)

# summaries rebuilt from the bootstrap arrays (column orders: err static, SR, TLS; corr TLS, SR, top, mean, gap; L CL, DbS, MLBA, SR)
bt = lambda n, k: boot[f"{n}__{k}"]  # noqa: E731
pool = {k: np.mean([bt(n, k) for n in NAMES], axis=0) for k in ("err", "corr", "L")}


def same(a, b, scale=1.0):
    return (abs(a["point"] - b["point"]) < tol * scale and np.allclose(a["percentile"], b["percentile"], rtol=0, atol=tol * scale)
            and np.allclose(a["basic"], b["basic"], rtol=0, atol=tol * scale) and a["p"] == b["p"] and a["clear"] == b["clear"])


f16 = res["folke2016"]
dL = lambda n, m: float(np.log2(tc[n]["SR"] / tc[n][m]))  # noqa: E731
rebuilt = [
    same(p["time"]["TLS_minus_static"], srch.summary_S(pt("pooled", "time", "TLS") - pt("pooled", "time", "static"), pool["err"][:, 2] - pool["err"][:, 0])),
    same(p["slow"]["TLS_plus_top_gap"], srch.summary_S(pt("pooled", "slow", "looks_TLS") + pt("pooled", "slow", "top_gap"), pool["corr"][:, 0] + pool["corr"][:, 4])),
    same(p["choices"]["TLS_minus_CL_percent"], srch.summary_percent(dL("pooled", "CL"), pool["L"][:, 3] - pool["L"][:, 0]), 1e4),
    same(p["choices"]["TLS_minus_MLBA_percent"], srch.summary_percent(dL("pooled", "MLBA"), pool["L"][:, 3] - pool["L"][:, 2]), 1e4),
    same(res["krajbich2011"]["choices"]["TLS_minus_DbS_percent"], srch.summary_percent(dL("krajbich2011", "DbS"), bt("krajbich2011", "L")[:, 3] - bt("krajbich2011", "L")[:, 1]), 1e4),
    same(f16["law"]["TLS"]["tau"], srch.summary_S(f16["law"]["TLS"]["tau"]["point"], bt("folke2016", "law")[:, 1, 1])),
    same(res["krajbich2010"]["looks"]["TLS"], srch.summary_S(res["krajbich2010"]["looks"]["TLS"]["point"], bt("krajbich2010", "looks")[:, 1])),
    same(f16["confidence"]["TLS_minus_SR"], srch.summary_S(f16["confidence"]["looks_TLS"]["point"] - f16["confidence"]["looks_SR"]["point"],
                                                           bt("folke2016", "conf")[:, 0] - bt("folke2016", "conf")[:, 1])),
]
check(f"summaries rebuilt from the bootstrap arrays match the reported ones ({sum(rebuilt)} of {len(rebuilt)})", all(rebuilt))

# ----------------------------------------------------------------------------- original-sample values recomputed from the files
warnings.simplefilter("ignore")  # spearmanr warns on constant input, which is skipped explicitly below


def within(x, target, rows, pidx, P):
    """Spearman per participant with at least three rows, constant variables skipped, then the plain mean."""
    vals = []
    for i in range(P):
        idx = np.flatnonzero((pidx == i) & rows)
        if len(idx) < 3 or np.ptp(x[idx]) == 0 or np.ptp(target[idx]) == 0:
            continue
        vals.append(spearmanr(x[idx], target[idx])[0])
    return float(np.mean(vals))


def rnd(x):
    """Item 6: every forecast rounded to ten significant digits, element by element, before ranking."""
    return np.array([float(format(v, ".10g")) for v in np.asarray(x, float).tolist()])


unrounded = 0.0  # how far the correlations would move without the rounding (information only)
worst = {"lambda": 0.0, "time": 0.0, "slow": 0.0, "L": 0.0, "looks": 0.0, "typical": 0.0, "law": 0.0, "conf": 0.0}
for n in NAMES:
    d = gc.load_dataset(DATA, n)
    V, ch, y, fold, pidx, P, nn = d["V"], d["choice"], d["y"], d["fold"], d["pidx"], d["P"], d["n"]
    T = len(ch)
    lamj = res[n]["lambda"]
    sr = gc.SRFit(V, ch)
    E_sr, E_tls, ent = np.zeros(T), np.zeros(T), np.zeros(T)
    bits = {m: np.zeros(T) for m in MODELS}
    pred = {k: np.zeros(T) for k in ("static", "SR", "TLS")}
    for f in range(gc.N_FOLDS):
        h, tr = fold == f, fold != f
        k_h = np.arange(h.sum())
        for m, Sm in gc.static_scores(V).items():  # refit on the training participants: the reported lambda
            worst["lambda"] = max(worst["lambda"], abs(gc.fit_softmax(Sm, ch, tr.astype(float)) - lamj[m][f]))
            lq = gc.log_softmax(lamj[m][f], Sm[h])
            bits[m][h] = np.log2(nn) + np.maximum(lq[k_h, ch[h]], gc.LOG_FLOOR) / np.log(2.0)
            if m == "CL":
                ent[h] = -(np.exp(lq) * lq).sum(axis=1) / np.log(2.0)
        lsr = lamj["SR"][f]
        worst["lambda"] = max(worst["lambda"], abs(sr.fit(tr.astype(float))[0] - lsr))
        bits["SR"][h] = np.log2(nn) + gc.sr_log_chosen(lsr, V[h], ch[h]) / np.log(2.0)
        E_sr[h], E_tls[h] = gc.sr_expected_looks(lsr, V[h]), gc.tls_expected_looks(lsr, V[h])
        pred["static"][h] = y[tr].mean()
        for k, fn in (("SR", gc.sr_expected_looks), ("TLS", gc.tls_expected_looks)):
            t0, tau, _ = srch.fit_time_law(fn(lsr, V[tr]), y[tr], np.ones(tr.sum()))
            pred[k][h] = np.log(t0 + tau * fn(lsr, V[h]))
    r = res[n]
    worst["time"] = max(worst["time"], *(abs(np.abs(y - pred[k]).mean() - r["time"][k]["point"]) for k in pred))
    allrows = np.ones(T, bool)
    corr = {"looks_TLS": E_tls, "looks_SR": E_sr, **{f: d[f] for f in FEATURES}}
    worst["slow"] = max(worst["slow"], *(abs(within(rnd(x), d["rt_ms"], allrows, pidx, P) - r["slow"][k]["point"]) for k, x in corr.items()))
    unrounded = max(unrounded, *(abs(within(x, d["rt_ms"], allrows, pidx, P) - r["slow"][k]["point"]) for k, x in corr.items()))
    worst["L"] = max(worst["L"], *(abs(bits[m].mean() - np.log2(r["choices"]["times_chance"][m])) for m in MODELS))
    worst["looks"] = max(worst["looks"], abs(E_sr.mean() - r["looks"]["SR"]["point"]), abs(E_tls.mean() - r["looks"]["TLS"]["point"]))
    worst["typical"] = max(worst["typical"], abs(np.exp(y.mean()) - r["typical_seconds"]["point"]))
    lam_all, _ = sr.fit(np.ones(T))
    for k, fn in (("SR", gc.sr_expected_looks), ("TLS", gc.tls_expected_looks)):
        t0, tau, _ = srch.fit_time_law(fn(lam_all, V), y, np.ones(T))
        worst["law"] = max(worst["law"], abs(t0 - r["law"][k]["t0"]["point"]), abs(tau - r["law"][k]["tau"]["point"]))
    if n == "folke2016":
        has = np.isfinite(d["conf"])
        cf = r["confidence"]
        for k, x in (("looks_TLS", E_tls), ("looks_SR", E_sr), ("entropy_CL", ent)):  # forecasts rounded, confidence not
            worst["conf"] = max(worst["conf"], abs(within(rnd(x), d["conf"], has, pidx, P) - cf[k]["point"]))
            unrounded = max(unrounded, abs(within(x, d["conf"], has, pidx, P) - cf[k]["point"]))
check("original sample recomputed from the files: lambdas refitted by fold, time errors, within-participant correlations "
      "(scipy spearmanr on forecasts rounded by float(format(x, '.10g')), times and confidence unrounded), L, held out looks, "
      "typical seconds, the laws on every trial, Folke's confidence (largest differences "
      + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + ")", max(worst.values()) < 1e-9)
print(f"info: without the rounding, the original-sample correlations would move by up to {unrounded:.1e}")

# ----------------------------------------------------------------------------- brief test 3: recovery and the expected pattern
for n in NAMES:
    lam = np.array(res[n]["lambda"]["SR"])
    sd = bt(n, "lam")[:, 3, :].std(axis=0)
    z = np.abs(lam - truth["lambda"]) / sd
    za = abs(diag[n]["original_lambda_all_SR"] - truth["lambda"]) / bt(n, "lam_all").std()
    check(f"{n}: SR (= TLS) lambda by fold {np.round(lam, 3).tolist()} and on every trial {diag[n]['original_lambda_all_SR']:.3f} "
          f"recover {truth['lambda']} (at most {max(z.max(), za):.1f} bootstrap s.d. away)", max(z.max(), za) < 4)
t = p["time"]
check(f"pooled time errors static {t['static']['point']:.4f}, SR {t['SR']['point']:.4f}, TLS {t['TLS']['point']:.4f}; "
      f"TLS - SR {t['TLS_minus_SR']['point']:+.4f} {np.round(t['TLS_minus_SR']['percentile'], 4).tolist()}, "
      f"TLS - static {t['TLS_minus_static']['point']:+.4f} {np.round(t['TLS_minus_static']['percentile'], 4).tolist()}; G1 {p['G1']['supported']}",
      t["TLS"]["point"] < t["SR"]["point"] and t["TLS"]["point"] < t["static"]["point"] and p["G1"]["supported"])
s = p["slow"]
check(f"pooled slow decisions: TLS {s['looks_TLS']['point']:+.4f} {np.round(s['looks_TLS']['percentile'], 4).tolist()}, SR {s['looks_SR']['point']:+.4f}, "
      f"TLS - SR {s['TLS_minus_SR']['point']:+.4f}; G2 {p['G2']['supported']}",
      s["looks_TLS"]["point"] > 0 and s["TLS_minus_SR"]["point"] > 0 and p["G2"]["supported"])
c = f16["confidence"]
check(f"confidence falls with TLS looks: {c['looks_TLS']['point']:+.4f} {np.round(c['looks_TLS']['percentile'], 4).tolist()} "
      f"(SR {c['looks_SR']['point']:+.4f}, CL entropy {c['entropy_CL']['point']:+.4f}, TLS - SR {c['TLS_minus_SR']['point']:+.4f})",
      c["looks_TLS"]["point"] < 0 and c["looks_TLS"]["clear"])
ok = all(res[n]["looks"]["TLS"]["point"] > res[n]["looks"]["SR"]["point"] for n in NAMES)  # E_TLS = E_SR / P with P <= 1
ok &= all(bt(n, "law")[:, i, 0].std() > 0 for n in NAMES for i in (0, 1)) and all(bt(n, "lam_all").std() > 0 for n in NAMES)
ok &= all(res[n]["law"]["TLS"]["tau"]["point"] > 0 and res[n]["law"]["TLS"]["t0"]["point"] > 0 for n in NAMES)
check("TLS takes more looks than SR on average in every dataset; the laws and the SR sensitivity on every trial vary across replicates "
      "(refitted); TLS's law has a positive start-up and a positive time per look", ok)

print("\ninfo, time per look (point and percentile interval):")
for n in NAMES:
    r = res[n]
    fmt = lambda x: f"{x['point']:.3f} [{x['percentile'][0]:.3f}, {x['percentile'][1]:.3f}]"  # noqa: E731
    print(f"  {n}: SR t0 {fmt(r['law']['SR']['t0'])} s, tau {fmt(r['law']['SR']['tau'])} s/look; TLS t0 {fmt(r['law']['TLS']['t0'])} s, "
          f"tau {fmt(r['law']['TLS']['tau'])} s/look; looks SR {fmt(r['looks']['SR'])}, TLS {fmt(r['looks']['TLS'])}; "
          f"typical {fmt(r['typical_seconds'])} s")
    print(f"     lambda by fold, CL {np.round(r['lambda']['CL'], 3).tolist()}, DbS {np.round(r['lambda']['DbS'], 3).tolist()}, "
          f"MLBA {np.round(r['lambda']['MLBA'], 3).tolist()}")
    print(f"     times chance {r['choices']['times_chance']}; TLS - DbS {r['choices']['TLS_minus_DbS_percent']['point']:+.2f} percent, "
          f"TLS - MLBA {r['choices']['TLS_minus_MLBA_percent']['point']:+.2f} percent; diagnostics: values from {diag[n]['smallest_item_value']:g} to {diag[n]['largest_item_value']:g}, "
          f"largest dense-grid excess {diag[n]['largest_dense_excess_over_SR_fit']:.1e}")
tc3 = p["choices"]["TLS_minus_CL_percent"]
print(f"info G3: TLS - CL {tc3['point']:+.3f} percent, percentile {np.round(tc3['percentile'], 3).tolist()}, basic {np.round(tc3['basic'], 3).tolist()}, "
      f"non-inferior {p['G3']['noninferior']}, better {p['G3']['better']}, worse {p['G3']['worse']}; times chance {p['choices']['times_chance']}")
print("info slow, TLS plus feature: " + ", ".join(f"{f} {s[f'TLS_plus_{f}']['point']:+.4f}" for f in FEATURES))
print(f"info run: {diag['seconds']} s for B = {diag['B']}")
print("\nEND-TO-END CHECKS PASSED" if ok_all else "\nSOME END-TO-END CHECKS FAILED")
