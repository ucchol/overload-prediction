"""Brief test 4 on a synthetic click file: the output layout, internal consistency, the original-sample values
recomputed from the file and the reported lambdas by independent code (pandas for the rows, scipy's BFGS for the
stopping rules, the general race code for SR), and the decisions the scenario should give.

Usage: python -I check_e2e_cdm.py RESULTS_JSON CLICK_FILE BOOT_NPZ {list,random}
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cdm_core as cc  # noqa: E402
import run_cdm as rc  # noqa: E402
from cdm_core import sc, srch  # noqa: E402

srch.assert_local(cc, rc, srch, sc)
res = json.load(open(sys.argv[1]))
CLICKS = sys.argv[2]
assert os.path.realpath(os.path.dirname(os.path.abspath(CLICKS))) != cc.CDM_DIR, "synthetic files only"
boot = np.load(sys.argv[3])
SCEN = sys.argv[4]
stem = os.path.splitext(os.path.basename(sys.argv[1]))[0]
diag = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), f"diag_{stem}.json")))
NAMES = rc.NAMES
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


# ----------------------------------------------------------------------------- layout
SK = {"point", "percentile", "basic", "p", "clear"}


def is_S(x):
    return isinstance(x, dict) and set(x) == SK and len(x["percentile"]) == 2 and len(x["basic"]) == 2 and isinstance(x["clear"], bool)


STOP = {"value", "share", "value_time", "share_time", "value_minus_share", "value_minus_share_with_time"}
GROWTH = {"observed", "predicted", "one_minus_observed", "predicted_minus_observed"}
CHOICES = {"times_chance", "SR_minus_CL_percent", "SR_minus_DbS_percent", "SR_minus_MLBA_percent"}
TIME = {"static", "SR", "SR_minus_static"}


def block_ok(b, exp):
    ok = set(b["stop"]) == STOP and all(is_S(v) for v in b["stop"].values())
    ok &= set(b["growth"]) == GROWTH | ({"observed_by_size", "looks_by_size"} if exp else set())
    ok &= all(is_S(b["growth"][k]) for k in GROWTH)
    if exp:
        ok &= all(len(b["growth"][k]) == 3 and all(isinstance(x, float) for x in b["growth"][k]) for k in ("observed_by_size", "looks_by_size"))
    ok &= set(b["choices"]) == CHOICES and set(b["choices"]["times_chance"]) == set(cc.MODELS)
    ok &= all(is_S(b["choices"][k]) for k in CHOICES - {"times_chance"})
    return ok and set(b["time"]) == TIME and all(is_S(v) for v in b["time"].values())


p = res["pooled"]
lay = set(res) == {"pooled", "standard", "process", "unsubmitted"} and set(res["unsubmitted"]) == set(NAMES) and all(len(v) == 6 and all(isinstance(x, float) for x in v) for v in res["unsubmitted"].values()) and set(p) == {"stop", "growth", "choices", "time", "C1", "C2", "C3"} and block_ok(p, False)
lay &= set(p["C1"]) == {"absolute", "relative"} and set(p["C2"]) == {"order_matters"}
lay &= set(p["C3"]) == {"margin", "noninferior", "better", "worse", "supported"} and p["C3"]["margin"] == 2.0
for n in NAMES:
    e = res[n]
    lay &= set(e) == {"stop", "growth", "choices", "time", "first", "lambda"} and block_ok(e, True)
    lay &= set(e["first"]) == {"observed", "predicted", "minus_half", "minus_predicted"} and all(is_S(v) for v in e["first"].values())
    lay &= set(e["lambda"]) == set(cc.MODELS) and all(len(v) == cc.N_FOLDS for v in e["lambda"].values())
check("output layout as in the brief (pooled block with C1 to C3; per experiment the block, first, lambda by fold, "
      "observed_by_size and looks_by_size inside growth)", lay)
B = boot["standard__L"].shape[0]
shapes = {"lam": (B, 4, cc.N_FOLDS), "L": (B, 4), "stop": (B, 4), "first": (B, 2), "err": (B, 2), "growth": (B, 2),
          "dense_excess": (B,), "limit_share": (B,), "empty_training_share": (B,)}
check("bootstrap arrays hold only fitted values and measures, with the expected shapes",
      set(boot.files) == {f"{n}__{k}" for n in NAMES for k in shapes} and all(boot[f"{n}__{k}"].shape == s for n in NAMES for k, s in shapes.items()))

# ----------------------------------------------------------------------------- internal consistency
tol = 1e-12
pt = lambda n, sec, k: res[n][sec][k]["point"]  # noqa: E731
ok = all(abs(pt("pooled", sec, k) - np.mean([pt(n, sec, k) for n in NAMES])) < tol
         for sec, keys in (("stop", STOP), ("growth", GROWTH), ("time", TIME)) for k in keys)
tc = {n: res[n]["choices"]["times_chance"] for n in ("pooled",) + NAMES}
ok &= all(abs(math.log2(tc["pooled"][m]) - np.mean([math.log2(tc[n][m]) for n in NAMES])) < tol for m in cc.MODELS)
for n in ("pooled",) + NAMES:
    s, g, t = res[n]["stop"], res[n]["growth"], res[n]["time"]
    ok &= abs(s["value_minus_share"]["point"] - s["value"]["point"] + s["share"]["point"]) < tol
    ok &= abs(s["value_minus_share_with_time"]["point"] - s["value_time"]["point"] + s["share_time"]["point"]) < tol
    ok &= abs(g["one_minus_observed"]["point"] - 1 + g["observed"]["point"]) < tol
    ok &= abs(g["predicted_minus_observed"]["point"] - g["predicted"]["point"] + g["observed"]["point"]) < tol
    ok &= abs(t["SR_minus_static"]["point"] - t["SR"]["point"] + t["static"]["point"]) < tol
    ok &= all(abs(res[n]["choices"][f"SR_minus_{m}_percent"]["point"] - 100 * (tc[n]["SR"] / tc[n][m] - 1)) < 1e-9 for m in cc.STATIC)
for n in NAMES:
    f_ = res[n]["first"]
    ok &= abs(f_["minus_half"]["point"] - f_["observed"]["point"] + 0.5) < tol
    ok &= abs(f_["minus_predicted"]["point"] - f_["observed"]["point"] + f_["predicted"]["point"]) < tol
check("pooled values are the means over the two experiments (stopping scores, growth, time errors, log2 times chance); "
      "differences, one minus, and percents agree with their parts", ok)


def clear_pos(s):
    return s["clear"] and s["percentile"][0] > 0 and s["basic"][0] > 0


def clear_neg(s):
    return s["clear"] and s["percentile"][1] < 0 and s["basic"][1] < 0


vms, fp, c3 = p["stop"]["value_minus_share"], res["process"]["first"], p["choices"]["SR_minus_CL_percent"]
dec = (p["C1"]["absolute"] == clear_pos(vms) and p["C1"]["relative"] == clear_neg(vms)
       and p["C2"]["order_matters"] == (clear_neg(fp["minus_half"]) and clear_neg(fp["minus_predicted"]))
       and p["C3"]["noninferior"] == (min(c3["percentile"][0], c3["basic"][0]) > -2) and p["C3"]["supported"] == p["C3"]["noninferior"]
       and p["C3"]["better"] == (c3["percentile"][0] > 0 and c3["basic"][0] > 0) and p["C3"]["worse"] == (c3["percentile"][1] < 0 and c3["basic"][1] < 0))
check("C1 to C3 decisions follow the stated rules from the reported summaries", dec)
bt = lambda n, k: boot[f"{n}__{k}"]  # noqa: E731
pool = {k: np.mean([bt(n, k) for n in NAMES], axis=0) for k in ("stop", "growth", "L", "err")}


def same(a, b, scale=1.0):
    return (abs(a["point"] - b["point"]) < tol * scale and np.allclose(a["percentile"], b["percentile"], rtol=0, atol=tol * scale)
            and np.allclose(a["basic"], b["basic"], rtol=0, atol=tol * scale) and a["p"] == b["p"] and a["clear"] == b["clear"])


dL = lambda n, m: math.log2(tc[n]["SR"] / tc[n][m])  # noqa: E731
rebuilt = [
    same(vms, srch.summary_S(pt("pooled", "stop", "value") - pt("pooled", "stop", "share"), pool["stop"][:, 0] - pool["stop"][:, 1])),
    same(p["stop"]["value_minus_share_with_time"], srch.summary_S(pt("pooled", "stop", "value_time") - pt("pooled", "stop", "share_time"), pool["stop"][:, 2] - pool["stop"][:, 3])),
    same(c3, srch.summary_percent(dL("pooled", "CL"), pool["L"][:, 3] - pool["L"][:, 0]), 1e4),
    same(res["standard"]["choices"]["SR_minus_MLBA_percent"], srch.summary_percent(dL("standard", "MLBA"), bt("standard", "L")[:, 3] - bt("standard", "L")[:, 2]), 1e4),
    same(fp["minus_predicted"], srch.summary_S(fp["minus_predicted"]["point"], bt("process", "first")[:, 0] - bt("process", "first")[:, 1])),
    same(p["growth"]["predicted_minus_observed"], srch.summary_S(pt("pooled", "growth", "predicted_minus_observed"), pool["growth"][:, 1] - pool["growth"][:, 0])),
    same(p["time"]["SR_minus_static"], srch.summary_S(pt("pooled", "time", "SR_minus_static"), pool["err"][:, 1] - pool["err"][:, 0])),
]
check(f"summaries rebuilt from the bootstrap arrays match the reported ones ({sum(rebuilt)} of {len(rebuilt)})", all(rebuilt))

# ----------------------------------------------------------------------------- original sample recomputed independently
raw = pd.read_csv(CLICKS)
raw.columns = [c.strip().upper() for c in raw.columns]
data = cc.load_cdm(CLICKS)


def penalized(beta, X, y, w):
    eta = X @ beta
    return float(w @ (np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(beta[1:] ** 2))


def grad(beta, X, y, w):
    pr = 1 / (1 + np.exp(-(X @ beta)))
    return X.T @ (w * (pr - y)) + np.r_[0.0, beta[1:]]


worst = {"stop": 0.0, "first": 0.0, "L": 0.0, "growth": 0.0, "by_size": 0.0, "time": 0.0}
for name, tracking in cc.EXPERIMENTS:
    X = data[name]
    r = res[name]
    lam = r["lambda"]
    # rows read again with pandas (rules of item 1), rounds with selections, in participant and round order
    D = raw[(raw.TRACKING == tracking) & (raw.PRACTICE == 0) & (raw.SUBMIT == 1) & raw.RNDMAXNUMOPER.isin([3, 7])
            & ~raw.TREATID.isin([21, 24, 25]) & (((raw.NUMOPTS == 10) & raw.RNDCODE.isin(range(1, 7))) | (raw.NUMOPTS.isin([20, 40]) & raw.RNDCODE.notna()))
            & raw.SLCTNUM.notna()].sort_values(["USERID", "RNDNUM", "SLCTNUM"])
    firsts = D[D.SLCTNUM == 1]
    lasts = D[D.SLCTNUM == D.NUMSLCT]
    nr = firsts.NUMOPTS.to_numpy(int)
    # positions 2 .. n + 1 are VAL2 .. VAL(n + 1), columns 1 .. n of VAL1 .. VAL41
    Vr = [row[1:1 + k] for row, k in zip(firsts[[f"VAL{i}" for i in range(1, 42)]].to_numpy(float), nr)]
    if any(np.isnan(v).any() or len(v) != k for v, k in zip(Vr, nr)):
        raise SystemExit("independent reading of the option values failed")
    ids = np.unique(firsts.USERID.to_numpy())
    perm = np.random.default_rng(20261015).permutation(ids)
    fold_id = {u: j % 10 for j, u in enumerate(perm)}
    fold_r = np.array([fold_id[u] for u in firsts.USERID])
    chosen = lasts.OPTORD.to_numpy(int) - 2
    # choices, first places and SR's held out looks with the reported lambdas (general race code for SR)
    lq = {m: np.zeros(len(nr)) for m in cc.MODELS}
    pred_place, E_ho = np.zeros(len(nr)), np.zeros(len(nr))
    for i, (v, c, f) in enumerate(zip(Vr, chosen, fold_r)):
        k = len(v)
        d = v[:, None] - v[None, :]
        scores = {"CL": v, "DbS": np.sign(d).sum(axis=1) / (k - 1), "MLBA": (np.exp(-np.abs(d * 6 / 35)) * d * 6 / 35).sum(axis=1) / (k - 1)}
        for m in cc.STATIC:
            z = lam[m][f] * scores[m]
            lq[m][i] = z[c] - logsumexp(z)
        z = lam["SR"][f] * v
        q = np.exp(z - logsumexp(z))
        Px, Ex = srch.sr_exact(q)
        lq["SR"][i] = math.log(max(Px[c], 1e-300))
        pred_place[i] = q @ np.arange(k) / (k - 1)
        E_ho[i] = Ex
    L = [np.mean(np.log2(nr) + np.maximum(lq[m], math.log(1e-300)) / math.log(2)) for m in cc.MODELS]
    worst["L"] = max(worst["L"], max(abs(L[i] - math.log2(r["choices"]["times_chance"][m])) for i, m in enumerate(cc.MODELS)))
    place = (firsts.OPTORD.to_numpy() - 2) / (nr - 1)
    worst["first"] = max(worst["first"], abs(place.mean() - r["first"]["observed"]["point"]), abs(pred_place.mean() - r["first"]["predicted"]["point"]))
    # growth and time laws from the final selections' times
    tfin = lasts.TIMESLCT.to_numpy(float)
    okt = np.isfinite(tfin) & (tfin > 0)
    lt = np.log(np.where(okt, tfin, 1.0))
    cp = firsts.RNDMAXNUMOPER.to_numpy(int)
    slopes_o, slopes_p = [], []
    for c in (3, 7):
        go = [lt[okt & (nr == k) & (cp == c)].mean() for k in cc.SIZES]
        gp = [np.log(E_ho[(nr == k) & (cp == c)]).mean() for k in cc.SIZES]
        slopes_o.append(np.polyfit(np.log(cc.SIZES), go, 1)[0])
        slopes_p.append(np.polyfit(np.log(cc.SIZES), gp, 1)[0])
    worst["growth"] = max(worst["growth"], abs(np.mean(slopes_o) - r["growth"]["observed"]["point"]), abs(np.mean(slopes_p) - r["growth"]["predicted"]["point"]))
    obs_by = [math.exp(lt[okt & (nr == k)].mean()) for k in cc.SIZES]
    looks_by = [E_ho[nr == k].mean() for k in cc.SIZES]
    worst["by_size"] = max(worst["by_size"], float(np.max(np.abs(np.array(obs_by) - r["growth"]["observed_by_size"]))),
                           float(np.max(np.abs(np.array(looks_by) - r["growth"]["looks_by_size"]))))
    errs = {"static": [], "SR": []}
    pred = {"static": np.full(len(nr), np.nan), "SR": np.full(len(nr), np.nan)}
    for f in range(10):
        tr, ho = (fold_r != f) & okt, (fold_r == f) & okt
        E_f = np.array([srch.sr_exact(np.exp(lam["SR"][f] * v - logsumexp(lam["SR"][f] * v)))[1] for v in Vr])
        for k, eff in (("static", nr.astype(float)), ("SR", E_f)):
            t0, tau, _ = srch.fit_time_law(eff[tr], lt[tr], np.ones(tr.sum()))
            pred[k][ho] = np.log(t0 + tau * eff[ho])
    for i, k in enumerate(("static", "SR")):
        worst["time"] = max(worst["time"], abs(np.mean(np.abs(lt[okt] - pred[k][okt])) - r["time"][k]["point"]))
    # stopping rules: every selection read with pandas, rules fitted by BFGS on the penalized likelihood
    vals = D[[f"VAL{i}" for i in range(2, 42)]].to_numpy(float)
    nn = D.NUMOPTS.to_numpy(int)
    vsel = D.VALNUM.to_numpy(float)
    better = np.array([(row[:k] > v).sum() for row, k, v in zip(vals, nn, vsel)])
    scored = better > 0
    y = (D.SLCTNUM == D.NUMSLCT).to_numpy(float)
    share = better / nn
    ts = D.TIMESLCT.to_numpy(float)
    tok = np.isfinite(ts) & (ts > 0)
    lts = np.log(np.where(tok, ts, 1.0))
    cell = np.searchsorted(cc.SIZES, nn) * 2 + np.searchsorted(cc.CPLX, D.RNDMAXNUMOPER.to_numpy(int))
    fsel = np.array([fold_id[u] for u in D.USERID])
    scores_ = {}
    for rule in rc.RULES:
        timed = rule.endswith("_time")
        x1 = vsel if rule.startswith("value") else share
        use = scored & (tok if timed else True)
        lp = np.full(len(y), np.nan)
        for f in range(10):
            for c in range(6):
                tr = use & (fsel != f) & (cell == c)
                ho = use & (fsel == f) & (cell == c)
                if not ho.any():
                    continue
                cols_tr, cols_ho = [np.ones(tr.sum())], [np.ones(ho.sum())]
                for xx in ([x1, lts] if timed else [x1]):
                    mu = xx[tr].mean()
                    sd = math.sqrt(((xx[tr] - mu) ** 2).mean())
                    cols_tr.append((xx[tr] - mu) / sd)
                    cols_ho.append((xx[ho] - mu) / sd)
                Xtr, Xho = np.column_stack(cols_tr), np.column_stack(cols_ho)
                wtr = np.ones(tr.sum())
                fit = minimize(penalized, np.zeros(Xtr.shape[1]), args=(Xtr, y[tr], wtr), jac=grad, method="BFGS",
                               options={"gtol": 1e-11, "maxiter": 10000})
                eta = Xho @ fit.x
                lpr = np.where(y[ho] > 0.5, -np.logaddexp(0, -eta), -np.logaddexp(0, eta))
                lp[ho] = np.maximum(lpr, math.log(1e-300)) / math.log(2)
        scores_[rule] = np.mean(lp[use])
        worst["stop"] = max(worst["stop"], abs(scores_[rule] - r["stop"][rule]["point"]))
check("original sample recomputed from the rows by independent code: stopping scores (BFGS), first places, L of the four "
      "models (general race code), growth, sizes, time errors (largest differences "
      + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()) + ")", worst["stop"] < 1e-7 and max(v for k, v in worst.items() if k != "stop") < 1e-9)

# ----------------------------------------------------------------------------- the scenario's decisions
fmt = lambda s: f"{s['point']:+.4f} [{s['percentile'][0]:+.4f}, {s['percentile'][1]:+.4f}]"  # noqa: E731
print(f"info: pooled value minus share {fmt(vms)}, with time {fmt(p['stop']['value_minus_share_with_time'])}; "
      f"process first place {fmt(fp['observed'])}, minus half {fmt(fp['minus_half'])}, minus predicted {fmt(fp['minus_predicted'])}")
if SCEN in ("list", "list_final1"):
    check(f"satisficing in list order: C1 absolute ({p['C1']}) and C2 finds the list order ({p['C2']})",
          p["C1"]["absolute"] and not p["C1"]["relative"] and p["C2"]["order_matters"])
    if SCEN == "list_final1":
        # Experiment 1's people click only their final choice: every scored selection there is a last one, every cell
        # holds one outcome, both rules take the limit, and Experiment 1 adds exactly zero to value minus share
        s1, b1 = res["standard"]["stop"], bt("standard", "stop")
        diff_zero = (s1["value_minus_share"]["point"] == 0.0 and s1["value_minus_share_with_time"]["point"] == 0.0
                     and bool(np.all(b1[:, 0] == b1[:, 1])) and bool(np.all(b1[:, 2] == b1[:, 3])))
        scores_zero = all(s1[k]["point"] == 0.0 for k in rc.RULES) and bool(np.all(b1 == 0.0))
        limit_all = bool(np.all(bt("standard", "limit_share") == 1.0))
        vms2 = res["process"]["stop"]["value_minus_share"]
        half = (abs(vms["point"] - vms2["point"] / 2) < 1e-14 and np.allclose(vms["percentile"], np.array(vms2["percentile"]) / 2, rtol=0, atol=1e-14)
                and np.allclose(vms["basic"], np.array(vms2["basic"]) / 2, rtol=0, atol=1e-14) and vms["p"] == vms2["p"] and vms["clear"] == vms2["clear"])
        print(f"info: Experiment 1 rule scores all exactly zero (no cell without training selections) {scores_zero}; "
              f"every replicate's Experiment 1 rules at the one-outcome limit {limit_all}; share of Experiment 1 rule cells "
              f"without training selections, original {diag['standard']['share_of_rule_cells_without_training_selections_original']:.3f}, "
              f"largest replicate {diag['standard']['share_of_rule_cells_without_training_selections_largest_replicate']:.3f}")
        check("Experiment 1 clicks only final choices: its value minus share (with and without time) is exactly zero in the original "
              f"sample and in every replicate ({diff_zero}), and the pooled value minus share is half of Experiment 2's in point, both "
              f"intervals and p, with the same clear flag ({half})", diff_zero and half)
else:
    check(f"random reading with replacement, stop at the first return: C1 not absolute ({p['C1']}) and C2 does not find the "
          f"list order ({p['C2']})", not p["C1"]["absolute"] and not p["C2"]["order_matters"])
for n in NAMES:
    r = res[n]
    print(f"info {n}: stop " + ", ".join(f"{k} {v['point']:+.4f}" for k, v in r["stop"].items())
          + f"; first observed {r['first']['observed']['point']:.3f}, predicted {r['first']['predicted']['point']:.3f}")
    print(f"     growth observed {fmt(r['growth']['observed'])}, predicted {fmt(r['growth']['predicted'])}, by size "
          f"{np.round(r['growth']['observed_by_size'], 2).tolist()} s, looks {np.round(r['growth']['looks_by_size'], 2).tolist()}")
    print(f"     times chance {({m: round(x, 4) for m, x in r['choices']['times_chance'].items()})}; time errors static "
          f"{r['time']['static']['point']:.4f}, SR {r['time']['SR']['point']:.4f}")
    print(f"     lambda SR {np.round(r['lambda']['SR'], 3).tolist()}, CL {np.round(r['lambda']['CL'], 3).tolist()}")
    print(f"     diagnostics {diag[n]}")
print(f"info C3: SR minus CL {fmt(c3)} percent, {p['C3']}; run {diag['seconds']} s for B = {diag['B']}")
print("\nEND-TO-END CHECKS PASSED" if ok_all else "\nSOME END-TO-END CHECKS FAILED")
