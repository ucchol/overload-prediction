"""Independent implementation of Amendment 12 (decision time from trial structure), written blind from the plan.

Only the package's data loaders are used to read the raw files. Features, centering, folds, the search model,
CL, fitting, regression, transfer and scoring are written here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.laguerre import laggauss
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

DEV_CSV = "/mnt/user-data/uploads/Updated_Choice_Overload.csv"
SETS = ("dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016")
# value scale (low, high) of each set, item 1 of Amendment 12
SCALES = {
    "dev_file": (0.0, 100.0),
    "dev_laptop": (0.0, 100.0),
    "snack": (-3.0, 3.0),
    "reutskaja": (-5.0, 5.0),
    "cdm": (1.0, 35.0),
    "krajbich2010": (0.0, 10.0),
    "krajbich2011": (0.0, 10.0),
    "folke2016": (0.0, 10.0),
}
FOLD_SEED = 20261014
BOOT_BASE = 120000
LOG_LAM_LO, LOG_LAM_HI = -7.0, 9.0  # on values rescaled to 0..1
GRID = np.linspace(LOG_LAM_LO, LOG_LAM_HI, 33)


# ----------------------------------------------------------------------------- data


def _pack(values_list, choice, seconds, pid_raw, name):
    """values_list: list of 1-d arrays. pid_raw: sortable participant keys. Returns a dict for one set."""
    nmax = max(len(v) for v in values_list)
    T = len(values_list)
    V = np.full((T, nmax), np.nan)
    for t, v in enumerate(values_list):
        V[t, : len(v)] = v
    n = np.array([len(v) for v in values_list], dtype=int)
    choice = np.asarray(choice, dtype=int)
    seconds = np.asarray(seconds, dtype=float)
    ok = (
        np.isfinite(seconds)
        & (seconds > 0)
        & (choice >= 0)
        & (choice < n)
        & np.array([np.all(np.isfinite(v)) for v in values_list])
    )
    keys = np.asarray(pid_raw)[ok]
    ids = sorted(set(keys.tolist()))
    index = {k: i for i, k in enumerate(ids)}
    lo, hi = SCALES[name]
    V = (V[ok] - lo) / (hi - lo)
    return {
        "name": name,
        "V": V,  # trials by nmax, rescaled to 0..1, NaN past n
        "n": n[ok],
        "choice": choice[ok],
        "seconds": seconds[ok],
        "pid": np.array([index[k] for k in keys.tolist()], dtype=int),
        "P": len(ids),
    }


def load_dev(task):
    from overload.data import _match, load_task

    df = pd.read_csv(DEV_CSV)
    t = load_task(df, task)
    vals, ch, sec, pid = [], [], [], []
    for r, (_, row) in enumerate(df.iterrows()):
        for tr in t.trials:
            idx = {o.lower(): i for i, o in enumerate(tr.options)}
            i = _match(row[tr.column], idx)
            if i is None:
                continue
            s = pd.to_numeric(pd.Series([row[f"{tr.column}_timer"]]), errors="coerce").iloc[0]
            vals.append(np.asarray(tr.U, float))
            ch.append(i)
            sec.append(float(s))
            pid.append(r)
    return _pack(vals, ch, sec, pid, f"dev_{task}")


def load_snack_set():
    from overload.snack import load_snack

    d = load_snack(REPO / "data" / "snack")
    vals, ch, sec, pid = [], [], [], []
    for b in d.blocks:
        for t in range(b.trials):
            vals.append(b.values[t].astype(float))
            ch.append(int(b.choice[t]))
            sec.append(float(b.rt[t]) / 1000.0)
            pid.append(int(b.participant[t]))
    return _pack(vals, ch, sec, pid, "snack")


def load_reutskaja():
    df = pd.read_stata(REPO / "data" / "reutskaja" / "fixations_data.dta")
    g = df.groupby(["subject", "trialid"], sort=True).first().reset_index()
    vals, ch, sec, pid = [], [], [], []
    for _, row in g.iterrows():
        n = int(row["numalt"])
        vals.append(np.array([row[f"rating{i}"] for i in range(1, n + 1)], dtype=float))
        c = row["chosendestring"]
        ch.append(int(c) - 1 if np.isfinite(c) else -1)
        sec.append(float(row["rt"]) / 1000.0)
        pid.append(int(row["subject"]))
    return _pack(vals, ch, sec, pid, "reutskaja")


def load_cdm_set():
    from overload.cdm import load_cdm

    d = load_cdm(REPO / "data" / "cdm")
    vals = [d.values[r, : d.n[r]].astype(float) for r in range(d.rounds)]
    return _pack(vals, d.choice, d.final_time, d.participant, "cdm")


def load_gaze_sets():
    from overload.gaze import load_gaze

    g = load_gaze(REPO / "data" / "gaze")
    out = {}
    for name in ("krajbich2010", "krajbich2011", "folke2016"):
        s = g[name]
        vals = [s.values[t].astype(float) for t in range(len(s.choice))]
        out[name] = _pack(vals, s.choice, s.seconds, s.participant, name)
    return out


def load_all():
    data = {"dev_file": load_dev("file"), "dev_laptop": load_dev("laptop"), "snack": load_snack_set(),
            "reutskaja": load_reutskaja(), "cdm": load_cdm_set()}
    data.update(load_gaze_sets())
    return {k: data[k] for k in SETS}


# ----------------------------------------------------------------------------- models


def softmax_rows(lam, V):
    """Row softmax of lam V with NaN entries (absent options) at probability zero."""
    absent = np.isnan(V)
    Z = np.where(absent, -np.inf, lam * np.where(absent, 0.0, V))
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _nodes(nmax):
    # sum_k (k+1)! e_k(r) = int_0^inf e^-t t prod(1 + r_j t) dt, a polynomial of degree <= nmax, so Gauss Laguerre
    # with m nodes is exact when 2m - 1 >= nmax.
    m = max(8, nmax // 2 + 4)
    x, w = laggauss(m)
    return x, np.log(w)


def sr_terms(q):
    """Per trial log prod_j (1 + q_j t_m) on the Laguerre nodes, trials by nodes."""
    x, lw = _nodes(q.shape[1])
    L = np.log1p(q[:, :, None] * x[None, None, :]).sum(axis=1)
    return x, lw, L


def sr_log_looks(q):
    """log of expected looks, log sum_k k! e_k(q)."""
    x, lw, L = sr_terms(q)
    return logsumexp(lw[None, :] + L, axis=1)


def sr_log_prob(q, idx):
    """log P(choose idx) = log q_i^2 sum_k (k+1)! e_k(q without i), per trial."""
    x, lw, L = sr_terms(q)
    qi = q[np.arange(q.shape[0]), idx]
    with np.errstate(divide="ignore"):
        inner = logsumexp(lw[None, :] + np.log(x)[None, :] + L - np.log1p(qi[:, None] * x[None, :]), axis=1)
        return 2.0 * np.log(qi) + inner


def sr_probs(q):
    """Full choice distribution, trials by items (for checks)."""
    x, lw, L = sr_terms(q)
    out = np.zeros_like(q)
    for i in range(q.shape[1]):
        qi = q[:, i]
        with np.errstate(divide="ignore"):
            inner = logsumexp(lw[None, :] + np.log(x)[None, :] + L - np.log1p(qi[:, None] * x[None, :]), axis=1)
            out[:, i] = np.exp(2.0 * np.log(qi) + inner)
    return out


def cl_entropy_bits(q):
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(q > 0, q * np.log2(q), 0.0).sum(axis=1)
    return h


def chosen_logp(kind, lam, V, choice):
    q = softmax_rows(lam, V)
    if kind == "cl":
        return np.log(np.maximum(q[np.arange(len(choice)), choice], 1e-300))
    return np.maximum(sr_log_prob(q, choice), np.log(1e-300))


def fit_lambda(kind, V, choice, w):
    """Weighted maximum likelihood over log lambda in [LOG_LAM_LO, LOG_LAM_HI]: grid, then bounded Brent search
    between the neighbors of the best grid value."""
    keep = w > 0
    V, choice, w = V[keep], choice[keep], w[keep]

    def nll(u):
        return -float(np.dot(w, chosen_logp(kind, np.exp(u), V, choice)))

    vals = np.array([nll(u) for u in GRID])
    j = int(np.argmin(vals))
    lo, hi = GRID[max(j - 1, 0)], GRID[min(j + 1, len(GRID) - 1)]
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
    u = res.x if res.fun <= vals[j] else GRID[j]
    return float(np.exp(u))


# ----------------------------------------------------------------------------- features, folds, laws


def center(x, pid, P):
    """Subtract each participant's unweighted mean over their trials (columns of a 2-d array or a vector)."""
    x = np.asarray(x, float)
    cnt = np.bincount(pid, minlength=P).astype(float)
    if x.ndim == 1:
        m = np.bincount(pid, weights=x, minlength=P) / np.maximum(cnt, 1)
        return x - m[pid]
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        m = np.bincount(pid, weights=x[:, c], minlength=P) / np.maximum(cnt, 1)
        out[:, c] = x[:, c] - m[pid]
    return out


def raw_features(d):
    V = d["V"]
    S = -np.sort(-np.where(np.isnan(V), -np.inf, V), axis=1)  # descending, -inf past n
    n = d["n"]
    best = S[:, 0]
    second = S[:, 1]
    rest_sum = np.where(np.isfinite(S[:, 1:]), S[:, 1:], 0.0).sum(axis=1)
    mean_rest = rest_sum / (n - 1)
    return np.column_stack([np.log2(n), best, best - second, mean_rest])


def prepare(d):
    d = dict(d)
    d["X"] = center(raw_features(d), d["pid"], d["P"])
    d["y"] = center(np.log(d["seconds"]), d["pid"], d["P"])
    d["folds"] = folds(d["P"])[d["pid"]]
    return d


def folds(P):
    perm = np.random.default_rng(FOLD_SEED).permutation(P)
    f = np.empty(P, dtype=int)
    f[perm] = np.arange(P) % 10
    return f


def wls(X, y, w):
    """Weighted least squares without intercept, minimum norm solution."""
    s = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * s[:, None], y * s, rcond=None)
    return beta


def r2(y, yhat, w):
    return float(1.0 - np.sum(w * (y - yhat) ** 2) / np.sum(w * y**2))


def within_set(d, w, want_lams=False):
    """Held out R2 of S1..S4 over ten folds of participants, trial weights w."""
    X, y, f = d["X"], d["y"], d["folds"]
    preds = {k: np.zeros_like(y) for k in ("S1", "S2", "S3", "S4")}
    lams = []
    for k in range(10):
        test = f == k
        if not test.any():
            continue
        tr = ~test
        wt = np.where(tr, w, 0.0)
        lam_sr = fit_lambda("sr", d["V"], d["choice"], wt)
        lam_cl = fit_lambda("cl", d["V"], d["choice"], wt)
        lams.append((lam_sr, lam_cl))
        looks = center(sr_log_looks(softmax_rows(lam_sr, d["V"])), d["pid"], d["P"])
        ent = center(cl_entropy_bits(softmax_rows(lam_cl, d["V"])), d["pid"], d["P"])
        designs = {
            "S1": X[:, :1],
            "S2": X,
            "S3": np.column_stack([X, looks]),
            "S4": np.column_stack([X, ent]),
        }
        for name, D in designs.items():
            b = wls(D[tr], y[tr], w[tr])
            preds[name][test] = D[test] @ b
    out = {name: r2(y, preds[name], w) for name in preds}
    if want_lams:
        out["lams"] = lams
    return out


def transfer(prep, weights):
    """S2 fitted on the pooled other sets (each set's weights scaled to sum one), applied to the held out set."""
    out = {}
    for name in SETS:
        Xs, ys, ws = [], [], []
        for other in SETS:
            if other == name:
                continue
            w = weights[other]
            Xs.append(prep[other]["X"])
            ys.append(prep[other]["y"])
            ws.append(w / w.sum())
        b = wls(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
        d = prep[name]
        out[name] = {"coef": b.tolist(), "R2": r2(d["y"], d["X"] @ b, weights[name])}
    return out


def analyse(prep, weights, want_lams=False):
    res = {}
    tr = transfer(prep, weights)
    for name in SETS:
        d, w = prep[name], weights[name]
        ws = within_set(d, w, want_lams)
        r = {k: ws[k] for k in ("S1", "S2", "S3", "S4")}
        r["T2"] = tr[name]["R2"]
        r["T1"] = r["S2"] - r["S1"]
        r["T3"] = r["S3"] - r["S2"]
        r["S4_minus_S2"] = r["S4"] - r["S2"]
        r["transfer_coef"] = tr[name]["coef"]
        if want_lams:
            r["fold_lambdas_sr_cl"] = ws["lams"]
        res[name] = r
    return res


def replicate_weights(prep, b):
    rng = np.random.default_rng(BOOT_BASE + b)
    out = {}
    for name in SETS:
        P = prep[name]["P"]
        draw = rng.integers(0, P, size=P)
        mult = np.bincount(draw, minlength=P).astype(float)
        out[name] = mult[prep[name]["pid"]]
    return out
