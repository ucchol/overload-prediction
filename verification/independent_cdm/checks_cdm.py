"""Checks of the Amendment 7 implementation on simulated data only (brief tests 1, 2, 3 and 5, and the loader).

Usage: python -I checks_cdm.py SIM_DIR      (SIM_DIR holds clicks_list.csv and clicks_random.csv with their truth files)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(VERIFY, "independent_gaze"))
sys.path.insert(2, os.path.join(VERIFY, "independent_search"))
import cdm_core as cc  # noqa: E402
import gaze_core as gc  # noqa: E402  (earlier race code)
import run_cdm as rc  # noqa: E402
from cdm_core import sc, srch  # noqa: E402

srch.assert_local(cc, gc, rc, srch, sc)
SIM_DIR = sys.argv[1]
assert os.path.realpath(SIM_DIR) != cc.CDM_DIR, "checks never touch the real click file"
ok_all = True
rng = np.random.default_rng(23)


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


def dollars(T, n):
    return np.clip(np.round(rng.normal(12.0, 7.0, size=(T, n))), 1, 35)


def read_upper(path):
    """A synthetic click file read directly, column names stripped and upper-cased."""
    d = pd.read_csv(path)
    d.columns = [c.strip().upper() for c in d.columns]
    return d


def sample_rows(logq):
    Q = np.exp(logq)
    return (rng.random((len(Q), 1)) > np.cumsum(Q, axis=1)).sum(axis=1).clip(max=Q.shape[1] - 1)


def simulate_sr(V, lam):
    """Sample until return on every row: looks fall on option j with probability softmax(lam v)_j; the first
    option looked at twice is chosen. Returns choices and numbers of looks."""
    T, n = V.shape
    cum = np.cumsum(np.exp(cc.log_softmax(lam, V)), axis=1)
    seen = np.zeros((T, n), bool)
    choice = np.full(T, -1)
    looks = np.zeros(T, int)
    live = np.arange(T)
    for step in range(1, n + 2):
        u = rng.random(len(live))
        j = np.minimum((u[:, None] > cum[live]).sum(axis=1), n - 1)
        rep = seen[live, j]
        choice[live[rep]] = j[rep]
        looks[live[rep]] = step
        seen[live[~rep], j[~rep]] = True
        live = live[~rep]
        if not len(live):
            break
    return choice, looks


# ----------------------------------------------------------------------------- 1. race formulas, attention, scores
worst_gen, worst_old, worst_sum, worst_esym = 0.0, 0.0, 0.0, 0.0
for n in cc.SIZES:
    V = dollars(12, n)
    for lam in (0.0, 0.05, 0.2, 0.6):
        logP = cc.sr_log_probs(lam, V)
        E = cc.sr_expected_looks(lam, V)
        worst_sum = max(worst_sum, float(np.max(np.abs(np.exp(logP).sum(axis=1) - 1))))
        worst_old = max(worst_old, float(np.max(np.abs(np.exp(logP - gc.sr_log_probs(lam, V)) - 1))),
                        float(np.max(np.abs(E / gc.sr_expected_looks(lam, V) - 1))))
        for t in range(3):
            q = np.exp(cc.log_softmax(lam, V[t:t + 1]))[0]
            Px, Ex = srch.sr_exact(q)
            worst_gen = max(worst_gen, float(np.max(np.abs(np.exp(logP[t]) / Px - 1))), abs(E[t] / Ex - 1))
        p = np.exp(cc.log_softmax(lam, V))
        worst_esym = max(worst_esym, float(np.max(np.abs(cc.esym(p) - gc.esym(p)))))
check(f"SR race probabilities and expected looks on dollar sets of 10, 20 and 40 options equal the earlier general code "
      f"(max relative difference {worst_gen:.1e}) and the earlier vectorized code ({worst_old:.1e}); probabilities sum to one "
      f"({worst_sum:.1e}); the triangular k! e_k recursion equals the full one ({worst_esym:.1e})",
      worst_gen < 1e-11 and worst_old < 1e-11 and worst_sum < 1e-12 and worst_esym < 1e-13)
for n, lam in ((10, 0.25), (20, 0.15), (40, 0.1)):
    v = dollars(1, n)
    S = 200_000
    ch, nl = simulate_sr(np.tile(v, (S, 1)), lam)
    P = np.exp(cc.sr_log_probs(lam, v))[0]
    E = cc.sr_expected_looks(lam, v)[0]
    z = np.max(np.abs(np.bincount(ch, minlength=n) / S - P) / np.sqrt(P * (1 - P) / S + 1e-300))
    zE = abs(nl.mean() - E) / (nl.std() / math.sqrt(S))
    q = np.exp(cc.log_softmax(lam, v))[0]
    place = cc.attention_place(lam, v)[0]
    check(f"simulation of sample until return, {n} dollar options at lambda {lam}: choices within {z:.1f} s.e. of the race "
          f"probabilities, looks {nl.mean():.4f} against {E:.4f} ({zE:.1f} s.e.); attention place {place:.4f} = "
          f"{float(q @ np.arange(n) / (n - 1)):.4f}", max(z, zE) < 4.5 and abs(place - q @ np.arange(n) / (n - 1)) < 1e-14)
worst = 0.0
for n in cc.SIZES:
    for _ in range(20):
        v = rng.integers(-3, 4, size=n).astype(float)
        C = np.bincount((v + 3).astype(int), minlength=7)[None, :].astype(float)
        for fam, mine in (("DbS_avg", cc.dbs_scores(v[None, :])), ("MLBA_avg", cc.mlba_scores(v[None, :], scale=1.0))):
            lev = sc.level_scores(fam, C, np.array([float(n)]))[0]
            worst = max(worst, float(np.max(np.abs(mine[0] - lev[(v + 3).astype(int)]))))
v3 = np.array([[35.0, 0.0, 17.5]])
d = 6.0 / 35.0 * (v3[0][:, None] - v3[0][None, :])
hand = (np.exp(-np.abs(d)) * d).sum(axis=1) / 2
check(f"DbS_avg and MLBA_avg scores for sets of 10, 20 and 40 equal the snack implementation at scale 1 (max difference {worst:.1e}); "
      f"MLBA on dollar values times 6/35 by hand", worst < 1e-13 and np.allclose(cc.mlba_scores(v3)[0], hand)
      and np.allclose(cc.dbs_scores(v3)[0], [1.0, -1.0, 0.0]))


# ----------------------------------------------------------------------------- 2. recovery of every sensitivity
def mixed_groups(T_each, gen):
    """A Groups-like object over dollar rounds of sizes 10, 20 and 40 with choices from gen(V)."""
    X = {"R": 3 * T_each, "groups": {}, "Vlist": [], "chosen": np.zeros(3 * T_each, int)}
    r = 0
    for n in cc.SIZES:
        V = dollars(T_each, n)
        c = gen(V)
        X["groups"][n] = np.arange(r, r + T_each)
        X["Vlist"] += list(V)
        X["chosen"][r:r + T_each] = c
        r += T_each
    return cc.Groups(X)


def softmax_se(G, m, lam_hat, w):
    def ll(x):
        tot = 0.0
        for S, (_, idx, _, c) in zip(G.S[m], G.items):
            tot += float(w[idx] @ cc.log_softmax(x, S)[np.arange(len(idx)), c])
        return tot
    h = 1e-4 * max(lam_hat, 1e-3)
    return 1 / math.sqrt(-(ll(lam_hat + h) - 2 * ll(lam_hat) + ll(lam_hat - h)) / h ** 2)


T_each = 1500
for m, lam_true in (("CL", 0.25), ("DbS", 2.0), ("MLBA", 3.0)):
    fn = {"CL": lambda V: V, "DbS": cc.dbs_scores, "MLBA": cc.mlba_scores}[m]
    G = mixed_groups(T_each, lambda V, fn=fn, lt=lam_true: sample_rows(cc.log_softmax(lt, fn(V))))
    w = np.ones(G.R)
    lam_hat = cc.fit_softmax_groups(G.S[m], G.items, w)
    se = softmax_se(G, m, lam_hat, w)
    check(f"recovery of {m} at lambda {lam_true} on mixed set sizes: {lam_hat:.4f} ({abs(lam_hat - lam_true) / se:.1f} s.e.)",
          abs(lam_hat - lam_true) < 4 * se)
for lam_true in (0.1, 0.3):
    G = mixed_groups(T_each, lambda V, lt=lam_true: simulate_sr(V, lt)[0])
    fit = cc.SRFitGroups(G)
    w = np.ones(G.R)
    lam_hat, info = fit.fit(w)
    h = 1e-4 * lam_hat
    se = 1 / math.sqrt(-(fit.loglik(lam_hat + h, w) - 2 * fit.loglik(lam_hat, w) + fit.loglik(lam_hat - h, w)) / h ** 2)
    check(f"recovery of SR at lambda {lam_true} from simulated searches on mixed set sizes: {lam_hat:.4f} "
          f"({abs(lam_hat - lam_true) / se:.1f} s.e.), dense-grid excess {info['dense_excess']:.1e}",
          abs(lam_hat - lam_true) < 4 * se and info["dense_excess"] <= 1e-9)


# ----------------------------------------------------------------------------- 3. the logistic stopping rules
def objective(beta, Z, y, w):
    return cc.logit_objective(beta, np.column_stack([np.ones(len(y)), Z]), y, w)


worst_obj, worst_grad, pd_ok = -np.inf, 0.0, True
for K in (1, 2):
    for rep in range(4):
        N = 900
        x = rng.normal(size=(N, K)) * rng.uniform(0.5, 3, size=K) + rng.normal(size=K)
        w = rng.integers(0, 4, size=N).astype(float)
        eta = -1.0 + x @ rng.normal(size=K)
        y = (rng.random(N) < 1 / (1 + np.exp(-eta))).astype(float)
        Z = np.column_stack([cc.zscore(x[:, k], *cc.wstandardize(x[:, k], w)) for k in range(K)])
        beta = cc.fit_logit(Z, y, w)
        ref = minimize(objective, np.zeros(K + 1), args=(Z, y, w), method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
        worst_obj = max(worst_obj, objective(beta, Z, y, w) - ref.fun)
        X1 = np.column_stack([np.ones(N), Z])
        p = 1 / (1 + np.exp(-(X1 @ beta)))
        g = X1.T @ (w * (p - y)) + np.r_[0.0, beta[1:]]
        H = (X1.T * (w * p * (1 - p))) @ X1 + np.diag(np.r_[0.0, np.ones(K)])
        worst_grad = max(worst_grad, float(np.max(np.abs(g))))
        pd_ok &= bool(np.all(np.linalg.eigvalsh(H) > 0))
m_, s_ = cc.wstandardize(np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 1.0]))
std_ok = abs(m_ - 2.25) < 1e-15 and abs(s_ - math.sqrt((1.5625 + 2 * 0.0625 + 3.0625) / 4)) < 1e-15
check(f"logistic rules: the Newton fit is the minimum of the penalized objective (at most {worst_obj:.1e} above BFGS, "
      f"largest gradient entry {worst_grad:.1e}, Hessian positive definite); weighted mean and population s.d. by hand",
      worst_obj <= 1e-10 and worst_grad < 1e-7 and pd_ok and std_ok)
N = 400
x = rng.normal(size=N)
y = (x > 0.3).astype(float)  # complete separation
w = np.ones(N)
Z = cc.zscore(x, *cc.wstandardize(x, w))[:, None]
beta_sep = cc.fit_logit(Z, y, w)
xh = rng.normal(size=200)
yh = (xh > 0.3).astype(float)
yh[:5] = 1 - yh[:5]  # held out selections against the separating rule
score_sep = float(np.mean(cc.logit_log2_prob(beta_sep, cc.zscore(xh, *cc.wstandardize(x, w))[:, None], yh)))
beta_lim = cc.fit_logit(Z, np.zeros(N), w)  # every training outcome 0: the intercept's limit
lp_lim = cc.logit_log2_prob(beta_lim, Z[:3], np.array([0.0, 1.0, 0.0]))
Zc = np.zeros((N, 1))  # a constant predictor standardizes to zero, the slope stays at zero
beta_c = cc.fit_logit(Zc, y, w)
check(f"complete separation: the penalized fit stays finite (slope {beta_sep[1]:.3f}) and the held out score is finite "
      f"({score_sep:.3f} bits); with one outcome only the fit takes its limit and the score is floored at 1e-300 "
      f"({lp_lim.tolist()}); a constant predictor gives the intercept logit(mean) and a zero slope",
      np.all(np.isfinite(beta_sep)) and np.isfinite(score_sep) and lp_lim[0] == 0.0 and lp_lim[2] == 0.0
      and abs(lp_lim[1] - np.log2(1e-300)) < 1e-9 and abs(beta_c[1]) < 1e-12 and abs(beta_c[0] - math.log(y.mean() / (1 - y.mean()))) < 1e-10)

# a cell whose training selections share one outcome: the limit keeps every slope at zero, and the value, share,
# value_time and share_time rules give identical held out predictions (probability one for that outcome)
lim_ok = True
for y_all in (0.0, 1.0):
    vtr, str_, ttr = rng.integers(1, 36, size=300).astype(float), rng.random(300), rng.normal(1.5, 0.5, size=300)
    vho, sho, tho = rng.integers(1, 36, size=50).astype(float), rng.random(50), rng.normal(1.5, 0.5, size=50)
    yho = (rng.random(50) < 0.3).astype(float)
    wtr = rng.integers(0, 3, size=300).astype(float)
    preds = []
    for cols_tr, cols_ho in (([vtr], [vho]), ([str_], [sho]), ([vtr, ttr], [vho, tho]), ([str_, ttr], [sho, tho])):
        Ztr = np.column_stack([cc.zscore(x, *cc.wstandardize(x, wtr)) for x in cols_tr])
        Zho = np.column_stack([cc.zscore(xh, *cc.wstandardize(x, wtr)) for x, xh in zip(cols_tr, cols_ho)])
        beta = cc.fit_logit(Ztr, np.full(300, y_all), wtr)
        lim_ok &= bool(np.all(beta[1:] == 0.0) and np.isinf(beta[0]) and np.sign(beta[0]) == (1 if y_all else -1))
        preds.append(cc.logit_log2_prob(beta, Zho, yho))
    lim_ok &= all(np.array_equal(preds[0], p) for p in preds[1:])
    lim_ok &= bool(np.all(preds[0][yho == y_all] == 0.0) and np.allclose(preds[0][yho != y_all], np.log2(1e-300)))
check("one-outcome cell: the fit takes its limit with every slope at zero, and the value, share, value_time and share_time rules "
      "give identical held out predictions (log2 probability 0 for that outcome, the 1e-300 floor for the other)", lim_ok)

# ----------------------------------------------------------------------------- loader
try:
    cc.load_cdm(os.path.join(cc.CDM_DIR, cc.CLICK_FILE))
    refused = False
except PermissionError:
    refused = True
check("the loader refuses the real click file unless explicitly allowed (nothing is read)", refused)
data = {}
for scen in ("list", "random"):
    path = os.path.join(SIM_DIR, f"clicks_{scen}.csv")
    truth = json.load(open(os.path.join(SIM_DIR, f"clicks_{scen}_truth.json")))
    data[scen] = cc.load_cdm(path)
    raw = read_upper(path)
    ok_incl, ok_fold, ok_sel = True, True, True
    for name, X in data[scen].items():
        tracking = dict(cc.EXPERIMENTS)[name]
        want = sorted(tuple(int(x) for x in r) for r in truth["rounds"][name])  # synthetic round keys
        got = sorted((tracking, int(u), int(r)) for u, r in zip(X["ids"][X["pidx"]], X["rndnum"]))
        ok_incl &= got == want
        perm_ids = np.random.default_rng(20261015).permutation(np.unique([u for _, u, _ in want]))
        fold_by_id = {int(u): j % 10 for j, u in enumerate(perm_ids)}
        ok_fold &= all(X["fold_of"][i] == fold_by_id[int(u)] for i, u in enumerate(X["ids"]))
        # selections against a direct reading of the rows
        inc = raw[cc.included(raw.rename(columns=lambda c: c.strip().lower()).apply(pd.to_numeric, errors="coerce"))
                  & (raw["TRACKING"] == tracking) & raw["SLCTNUM"].notna()]
        inc = inc.sort_values(["USERID", "RNDNUM", "SLCTNUM"])
        vals = inc[[f"VAL{i}" for i in range(2, 42)]].to_numpy(float)
        v_sel = inc["VALNUM"].to_numpy(float)
        nn = inc["NUMOPTS"].to_numpy(int)
        better = np.array([(row[:k] > v).sum() for row, k, v in zip(vals, nn, v_sel)])
        best = np.array([row[:k].max() for row, k in zip(vals, nn)])
        ok_sel &= np.allclose(X["s_share"], better / nn) and np.array_equal(X["s_scored"], v_sel < best)
        ok_sel &= np.array_equal(X["s_last"] == 1, (inc["SLCTNUM"] == inc["NUMSLCT"]).to_numpy())
        first = inc[inc["SLCTNUM"] == 1]
        ok_sel &= np.allclose(X["place"], (first["OPTORD"].to_numpy() - 2) / (first["NUMOPTS"].to_numpy() - 1))
        lastr = inc[inc["SLCTNUM"] == inc["NUMSLCT"]]
        ok_sel &= np.array_equal(X["chosen"], lastr["OPTORD"].to_numpy(int) - 2)
    check(f"{scen} file: the analysed rounds are those intended (practice, unsubmitted, round codes, Experiments 3 and 4 "
          f"and rounds without a selection left out)", ok_incl)
    check(f"{scen} file: folds are positions in default_rng(20261015).permutation of the sorted participants modulo 10, "
          f"a fresh generator per experiment", ok_fold)
    check(f"{scen} file: selections, outcomes (last selection), shares of better options (strictly), scored selections "
          f"(not worth as much as the best), first places and choices agree with a direct reading of the rows", ok_sel)

bad = os.path.join(HERE, "sim", "bad_copy")
os.makedirs(bad, exist_ok=True)
raw = read_upper(os.path.join(SIM_DIR, "clicks_list.csv"))
num = raw.rename(columns=lambda c: c.strip().lower()).apply(pd.to_numeric, errors="coerce")
inc_mask = cc.included(num) & raw["SLCTNUM"].notna()
row = raw.index[inc_mask][7]
# the same file with upper-case column names loads to identical arrays
f_up = os.path.join(bad, "clicks_upper.csv")
raw.to_csv(f_up, index=False)
A_low, A_up = data["list"], cc.load_cdm(f_up)
same_case = all(np.array_equal(np.asarray(A_low[n][k]), np.asarray(A_up[n][k]), equal_nan=True)
                for n in A_low for k in ("pidx", "fold", "chosen", "place", "t_final", "s_value", "s_share", "s_last", "s_cell"))
check("upper-case and lower-case column names load to identical arrays", same_case)
# the round of that selection, and one of its option positions that no selection points to
r_key = raw.loc[row, ["TRACKING", "USERID", "RNDNUM"]]
in_round = (raw["TRACKING"] == r_key["TRACKING"]) & (raw["USERID"] == r_key["USERID"]) & (raw["RNDNUM"] == r_key["RNDNUM"])
n_round = int(raw.loc[row, "NUMOPTS"])
free_pos = sorted(set(range(2, n_round + 2)) - set(raw.loc[in_round, "OPTORD"].dropna().astype(int)))[0]
col_free = f"VAL{free_pos}"
cell_rows = inc_mask & (raw["TRACKING"] == 1) & (raw["NUMOPTS"] == 40) & (raw["RNDMAXNUMOPER"] == 7)
tests = {}
for label, change in (("position outside the list", lambda d: d.__setitem__("OPTORD", d["OPTORD"].where(d.index != row, d.loc[row, "NUMOPTS"] + 2))),
                      ("value differing from the list", lambda d: d.__setitem__("VALNUM", d["VALNUM"].where(d.index != row, d.loc[row, "VALNUM"] + 1))),
                      ("missing selection row", lambda d: d.drop(index=row, inplace=True)),
                      ("round-level column varying", lambda d: d.__setitem__("NUMSLCT", d["NUMSLCT"].where(d.index != row, d.loc[row, "NUMSLCT"] + 1))),
                      ("listed value of 36 dollars", lambda d: d.__setitem__(col_free, d[col_free].where(~in_round, 36.0))),
                      ("listed value of half a dollar", lambda d: d.__setitem__(col_free, d[col_free].where(~in_round, 0.5))),
                      ("missing cell (40 options, complexity 7, Experiment 2)", lambda d: d.drop(index=d.index[cell_rows], inplace=True))):
    d = raw.copy()
    change(d)
    f = os.path.join(bad, "clicks.csv")
    d.to_csv(f, index=False)
    try:
        cc.load_cdm(f)
        tests[label] = "loaded"
    except ValueError as err:
        tests[label] = str(err)
# the draw pre-check: a cell of Experiment 2 kept for one participant only is left without a drawn participant in some
# replicate, so the run stops before fitting; on the intact file every replicate keeps every cell
one = raw["USERID"][cell_rows].iloc[0]
d = raw.drop(index=raw.index[cell_rows & (raw["USERID"] != one)])
f = os.path.join(bad, "clicks.csv")
d.to_csv(f, index=False)
narrow = {n: rc.Pre(X) for n, X in cc.load_cdm(f).items()}
try:
    rc.precheck_draws(narrow, 1000)
    stops = False
except SystemExit:
    stops = True
try:
    rc.precheck_draws({n: rc.Pre(X) for n, X in data["list"].items()}, 1000)
    passes = True
except SystemExit:
    passes = False
check("the draw pre-check stops before fitting when a cell can lose every drawn participant, and passes on the intact file",
      stops and passes)
shutil.rmtree(bad)
for label, msg in tests.items():
    print(f"      {label}: {msg}")
check("the loader stops with an error on a " + ", a ".join(tests) + " (structure the amendment states as fact)",
      all(m != "loaded" for m in tests.values()))


# ----------------------------------------------------------------------------- 5. maxima of every fit
def brute(e, y, w):
    """Best weighted squared error by brute force: tau = 0, a grid over (log t0, log tau) with t0 from 1e-9,
    and L-BFGS-B polishing (t0 >= 1e-9) from the grid's local minima (at most 25)."""
    lt0 = np.linspace(np.log(srch.T0_MIN), np.log(1e3), 241)
    ltau = np.linspace(-14.0, 6.0, 241)
    ue, inv = np.unique(e, return_inverse=True)
    Wu, Yu, Qu = (np.bincount(inv, weights=x) for x in (w, w * y, w * y * y))
    le = np.log(ue)

    def obj(z):
        L = np.logaddexp(z[0], z[1] + le)
        return float(Qu.sum() - 2 * L @ Yu + (L * L) @ Wu)

    Gr = np.empty((len(lt0), len(ltau)))
    for i, a in enumerate(lt0):
        L = np.logaddexp(a, ltau[:, None] + le[None, :])
        Gr[i] = Qu.sum() - 2 * L @ Yu + (L * L) @ Wu
    best = min(float(w @ (y - (w @ y) / w.sum()) ** 2), float(Gr.min()))
    Gp = np.pad(Gr, 1, constant_values=np.inf)
    is_min = np.ones(Gr.shape, bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_min &= Gr <= Gp[1 + di:1 + di + Gr.shape[0], 1 + dj:1 + dj + Gr.shape[1]]
    for i, j in np.argwhere(is_min)[np.argsort(Gr[is_min], kind="stable")][:25]:
        r = minimize(obj, x0=[lt0[i], ltau[j]], method="L-BFGS-B", bounds=[(np.log(srch.T0_MIN), 20.0), (-60.0, 20.0)])
        best = min(best, float(r.fun))
    return best


worst_static, worst_dense, worst_time = 0.0, -np.inf, -np.inf
for scen in ("list", "random"):
    for name, X in data[scen].items():
        pre = rc.Pre(X)
        for b in (-1, 0):
            wp = np.ones(pre.P) if b < 0 else np.bincount(np.random.default_rng(rc.BOOT_SEED + b).integers(0, pre.P, size=pre.P), minlength=pre.P).astype(float)
            w = wp[X["pidx"]]
            for f in range(cc.N_FOLDS):
                wt = np.where(pre.train[f], w, 0.0)
                for m in cc.STATIC:
                    lam = cc.fit_softmax_groups(pre.G.S[m], pre.G.items, wt)
                    g = hss = 0.0
                    for S, (_, idx, _, c) in zip(pre.G.S[m], pre.G.items):
                        q = np.exp(cc.log_softmax(lam, S))
                        mean = (q * S).sum(axis=1)
                        g += float(wt[idx] @ (S[np.arange(len(idx)), c] - mean))
                        hss += float(wt[idx] @ (q * (S - mean[:, None]) ** 2).sum(axis=1))
                    worst_static = max(worst_static, abs(g / hss) / lam if lam > 0 else max(g, 0.0) / hss)
                lam_sr, info = pre.sr.fit(wt)
                worst_dense = max(worst_dense, info["dense_excess"])
                if f < 2:
                    on = (wt > 0) & X["tf_ok"]
                    E = pre.G.per_round(lambda V, c: cc.sr_expected_looks(lam_sr, V))
                    for eff in (X["n"].astype(float), E):
                        _, _, obj = srch.fit_time_law(eff[on], X["logt_final"][on], wt[on])
                        bf = brute(eff[on], X["logt_final"][on], wt[on])
                        worst_time = max(worst_time, (obj - bf) / bf)
check(f"CL, DbS and MLBA: every fit (all folds, original and resampled weights, both files) is a stationary point of a "
      f"concave likelihood (largest relative Newton step left {worst_static:.1e})", worst_static < 1e-10)
check(f"SR: the grid and bounded Brent search (xatol 1e-10) reach the global maximum in every fit (largest dense-grid excess {worst_dense:.1e})",
      worst_dense <= 1e-9)
check(f"time laws (effort n and SR's expected looks; two folds per experiment, both weightings, both files) at or below brute "
      f"force, a grid polished from each of its local minima (largest relative excess {worst_time:.1e})", worst_time <= 1e-10)

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
