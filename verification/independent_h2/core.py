"""Blind independent implementation of Amendment 11 (H2, the return rule on gaze sequences).

Written from ANALYSIS_PLAN.md and VERIFY_BRIEF_H2.md only. numpy, scipy and pandas.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

CUES = ("return", "dwell", "last")
LN2 = np.log(2.0)


# ---------------------------------------------------------------- cues

def merge_consecutive(items, durs):
    """Merge consecutive gazes on one item into one gaze (durations summed)."""
    out_i, out_d = [], []
    for it, d in zip(items, durs):
        if out_i and out_i[-1] == it:
            out_d[-1] += d
        else:
            out_i.append(it)
            out_d.append(d)
    return out_i, out_d


def trial_cues(items, durs, n):
    """Cue vectors (return, dwell, last) for one trial.

    items: 0 based item indices in gaze order, durs: their durations.
    A trial without gaze has every cue zero.
    """
    ret = np.zeros(n)
    dwell = np.zeros(n)
    last = np.zeros(n)
    first_ret = -1
    if len(items) == 0:
        return ret, dwell, last, first_ret
    items, durs = merge_consecutive(list(items), list(durs))
    seen = set()
    for it in items:
        if it in seen:
            first_ret = it
            break
        seen.add(it)
    if first_ret >= 0:
        ret[first_ret] = 1.0
    tot = float(np.sum(durs))
    if tot > 0:
        for it, d in zip(items, durs):
            dwell[it] += d
        dwell /= tot
    last[items[-1]] = 1.0
    return ret, dwell, last, first_ret


class Group:
    """Trials of one set size of one dataset."""

    def __init__(self, n, V, C, choice, part, first_ret):
        self.n = n
        self.V = np.asarray(V, float)            # T x n ratings
        self.C = {k: np.asarray(C[k], float) for k in CUES}  # T x n each
        self.choice = np.asarray(choice, int)    # T
        self.part = np.asarray(part, int)        # T, participant index in bootstrap order
        self.first_ret = np.asarray(first_ret, int)  # T, -1 when no return


def _build(rows, sizes):
    groups = []
    for n in sizes:
        rs = [r for r in rows if r["n"] == n]
        V = np.array([r["v"] for r in rs])
        C = {k: np.array([r[k] for r in rs]) for k in CUES}
        groups.append(Group(n, V, C, [r["c"] for r in rs], [r["p"] for r in rs],
                            [r["fr"] for r in rs]))
    return groups


def load_snack(agg_dir, gaze_dir):
    sizes = [9, 16, 25, 36]
    aggs = []
    for n in sizes:
        d = pd.read_csv(os.path.join(agg_dir, f"setsize-{n}_desc-data.csv"))
        aggs.append(d)
    # participants in order of first appearance, files read in increasing set size
    order = list(pd.unique(pd.concat([d["subject"] for d in aggs], ignore_index=True)))
    pidx = {s: i for i, s in enumerate(order)}
    files = sorted(glob.glob(os.path.join(gaze_dir, "*_desc-gazes.csv")))
    g = pd.concat([pd.read_csv(f, usecols=["subject", "setsize", "trial", "item", "dur", "gaze_num"])
                   for f in files], ignore_index=True)
    g = g.sort_values(["subject", "trial", "gaze_num"], kind="mergesort")
    gaze = {k: (sub["item"].to_numpy(int), sub["dur"].to_numpy(float))
            for k, sub in g.groupby(["subject", "trial"], sort=False)}
    rows = []
    for n, d in zip(sizes, aggs):
        vcols = [f"item_value_{i}" for i in range(n)]
        for rec in d.itertuples(index=False):
            v = np.array([getattr(rec, c) for c in vcols], float)
            ch = rec.choice
            if np.any(~np.isfinite(v)) or not np.isfinite(ch):
                continue
            ch = int(ch)
            if ch < 0 or ch >= n:
                continue
            it, du = gaze.get((rec.subject, rec.trial), (np.array([], int), np.array([])))
            r, dw, la, fr = trial_cues(it, du, n)
            rows.append(dict(n=n, v=v, c=ch, p=pidx[rec.subject], fr=fr,
                             **{"return": r, "dwell": dw, "last": la}))
    return _build(rows, sizes), len(order)


def load_reutskaja(path):
    r = pd.read_stata(path)
    sizes = [4, 9, 16]
    order = list(pd.unique(r["subject"]))
    pidx = {s: i for i, s in enumerate(order)}
    rows = []
    for (subj, tid), t in r.groupby(["subject", "trialid"], sort=False):
        t = t.sort_values("fixOrder", kind="mergesort")
        n = int(t["numalt"].iloc[0])
        v = t[[f"rating{i}" for i in range(1, n + 1)]].iloc[0].to_numpy(float)
        ch = t["chosendestring"].iloc[0]
        if np.any(~np.isfinite(v)) or not np.isfinite(ch):
            continue
        ch = int(ch) - 1
        if ch < 0 or ch >= n:
            continue
        ok = t["Roinum"].notna() & t["FixDuration"].notna()
        it = t.loc[ok, "Roinum"].to_numpy(int) - 1
        du = t.loc[ok, "FixDuration"].to_numpy(float)
        rr, dw, la, fr = trial_cues(it, du, n)
        rows.append(dict(n=n, v=v, c=ch, p=pidx[subj], fr=fr,
                         **{"return": rr, "dwell": dw, "last": la}))
    return _build(rows, sizes), len(order)


# ---------------------------------------------------------------- model

def _feats(g, cue):
    if cue is None:
        return g.V[:, :, None]
    return np.stack([g.V, g.C[cue]], axis=2)  # T x n x k


def nll_grad_hess(theta, data):
    """Weighted negative log likelihood (nats), gradient and Hessian.

    data: list of (X T x n x k, choice T, w T).
    """
    k = len(theta)
    f = 0.0
    gr = np.zeros(k)
    H = np.zeros((k, k))
    for X, ch, w in data:
        u = X @ theta
        lse = logsumexp(u, axis=1)
        idx = np.arange(len(ch))
        f -= np.sum(w * (u[idx, ch] - lse))
        q = np.exp(u - lse[:, None])
        Ex = np.einsum("tn,tnk->tk", q, X)
        gr -= np.sum(w[:, None] * (X[idx, ch, :] - Ex), axis=0)
        Exx = np.einsum("tn,tnk,tnj->tkj", q, X, X)
        cov = Exx - Ex[:, :, None] * Ex[:, None, :]
        H += np.einsum("t,tkj->kj", w, cov)
    return f, gr, H


def fit(groups, weights, cue):
    """ML fit of (lambda) or (lambda, beta) on the given groups with trial weights.

    L-BFGS-B from several starts, lambda >= 0 and beta free, then a projected
    Newton polish on the concave likelihood.
    """
    data = []
    for g, w in zip(groups, weights):
        m = w > 0
        if m.any():
            data.append((_feats(g, cue)[m], g.choice[m], w[m]))
    k = 1 if cue is None else 2
    starts = [[0.5], [0.05], [2.0]] if k == 1 else \
        [[0.5, 0.0], [0.05, 0.0], [1.0, 2.0], [1.0, -2.0], [2.0, 5.0]]
    bounds = [(0.0, None)] + [(None, None)] * (k - 1)
    best = None
    for s in starts:
        res = minimize(lambda th: nll_grad_hess(th, data)[:2], np.array(s, float), jac=True,
                       method="L-BFGS-B", bounds=bounds,
                       options=dict(ftol=1e-15, gtol=1e-11, maxiter=2000))
        if best is None or res.fun < best[1]:
            best = (res.x.copy(), res.fun)
    th = best[0]
    # projected Newton polish
    for _ in range(50):
        f, gr, H = nll_grad_hess(th, data)
        free = np.ones(k, bool)
        if th[0] <= 0 and gr[0] > 0:
            free[0] = False
        if not free.any():
            break
        step = np.zeros(k)
        try:
            step[free] = np.linalg.solve(H[np.ix_(free, free)], gr[free])
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while t > 1e-12:
            cand = th - t * step
            cand[0] = max(cand[0], 0.0)
            if nll_grad_hess(cand, data)[0] <= f + 1e-13 * max(1.0, abs(f)):
                break
            t /= 2
        if t <= 1e-12:
            break
        th = cand
        if np.max(np.abs(t * step)) < 1e-13:
            break
    return th


def log2_q_chosen(g, theta, cue):
    u = _feats(g, cue) @ theta
    lse = logsumexp(u, axis=1)
    return (u[np.arange(len(g.choice)), g.choice] - lse) / LN2


# ---------------------------------------------------------------- one sample

def weights_from_counts(groups, counts):
    return [counts[g.part].astype(float) for g in groups]


def analyse(groups, counts):
    """Every quantity of Amendment 11 for one dataset at the given draw counts."""
    W = weights_from_counts(groups, counts)
    out = {}
    scores = {}
    for cue in (None,) + CUES:
        num = 0.0
        num_u = 0.0
        den = 0.0
        for k, g in enumerate(groups):
            tr = [x for j, x in enumerate(groups) if j != k]
            trw = [x for j, x in enumerate(W) if j != k]
            th = fit(tr, trw, cue)
            l2 = log2_q_chosen(g, th, cue)
            num += np.sum(W[k] * l2)
            num_u += np.sum(W[k] * (l2 + np.log2(g.n)))
            den += np.sum(W[k])
        name = "CL" if cue is None else cue
        scores[name] = (num / den, num_u / den)
        full = fit(groups, W, cue)
        if cue is None:
            out["CL_lambda_all"] = float(full[0])
        else:
            out[f"{cue}_beta"] = float(full[1])
            out[f"{cue}_lambda_all"] = float(full[0])
    out["CL_log2_score"] = float(scores["CL"][0])
    out["CL_bits_above_uniform"] = float(scores["CL"][1])
    for cue in CUES:
        out[f"{cue}_gain"] = float(scores[cue][0] - scores["CL"][0])
    out["return_minus_dwell"] = out["return_gain"] - out["dwell_gain"]
    for g, w in zip(groups, W):
        has = g.first_ret >= 0
        wr = np.sum(w[has])
        chosen = np.sum(w[has] * (g.first_ret[has] == g.choice[has]))
        out[f"first_return_chosen_n{g.n}"] = float(chosen / wr) if wr > 0 else float("nan")
        out[f"no_return_n{g.n}"] = float(np.sum(w[~has]) / np.sum(w))
    return out


# ---------------------------------------------------------------- intervals

def summarize(point, reps):
    reps = np.asarray(reps, float)
    reps_f = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps_f, [2.5, 97.5])
    pct = [float(lo), float(hi)]
    basic = [float(2 * point - hi), float(2 * point - lo)]
    excl = lambda iv: iv[0] > 0 or iv[1] < 0
    clear = bool(excl(pct) and excl(basic))
    p = float(min(1.0, 2 * min(np.mean(reps_f <= 0), np.mean(reps_f >= 0))))
    return dict(point=float(point), percentile=pct, basic=basic, clear=clear,
                clear_positive=bool(pct[0] > 0 and basic[0] > 0),
                p=p, replicates=[float(x) for x in reps])
