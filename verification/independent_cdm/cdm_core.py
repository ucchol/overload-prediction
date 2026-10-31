"""Amendment 7 (VERIFY_BRIEF_CDM.md): the search model's stopping rule on the click data of Caplin, Dean and
Martin (2011). Data, models and fits.

Rounds hold n options (list positions 2 .. n + 1, values in dollars); the choice is the last selection.
  CL      softmax(lam v)
  DbS     softmax(lam s), s_i = sum_{j != i} sign(v_i - v_j) / (n - 1)                 (DbS_avg, Amendment 3)
  MLBA    softmax(lam s), s_i = sum_{j != i} exp(-|d|) d / (n - 1), d = (6/35)(v_i - v_j)  (MLBA_avg, kappa = 1)
  SR      race probabilities of q = softmax(lam v): P(i) = q_i^2 sum_k (k+1)! e_k(q_-i),
          expected looks E_SR(q) = sum_k k! e_k(q), attention on option j with probability q_j.
Stopping rules: penalized logistic regressions (standard normal prior on standardized slopes), Newton's method.
Reuses ../independent_search/search_core.py (time law of Amendment 6, summaries of Section 4).
Nothing here prints or stores identifiers, demographics, or participant, round or selection counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_search"))
import search_core as srch  # noqa: E402
from search_core import sc  # noqa: E402

srch.assert_local(srch, sc, sys.modules[__name__])

CDM_DIR = os.path.realpath(os.path.join(VERIFY, "cdm"))
CLICK_FILE = "ProcessedCP_All_Final.csv"
EXPERIMENTS = (("standard", 0), ("process", 1))  # Experiment 1 (tracking 0) first
SIZES = (10, 20, 40)
CPLX = (3, 7)
STATIC = ("CL", "DbS", "MLBA")
MODELS = ("CL", "DbS", "MLBA", "SR")
MLBA_SCALE = 6.0 / 35.0
LOG_FLOOR = float(np.log(1e-300))
SR_GRID = np.concatenate([[0.0], np.logspace(-2, 2, 21)])  # Amendment 5 grid
SR_XATOL = 1e-10
DENSE_GRID = np.concatenate([[0.0], np.logspace(-4, 3, 561)])  # global-maximum diagnostic only
N_FOLDS = 10
FOLD_SEED = 20261015
LN2 = float(np.log(2.0))
VALS = [f"val{i}" for i in range(1, 42)]
NEED = ["tracking", "treatid", "userid", "rndnum", "rndcode", "practice", "submit", "numopts", "rndmaxnumoper",
        "slctnum", "numslct", "timeslct", "valnum", "optord"] + VALS
ROUND_LEVEL = ["treatid", "rndcode", "practice", "submit", "numopts", "rndmaxnumoper", "numslct"] + VALS


# ============================================================================ data
def read_clicks(path: str, allow_real: bool = False) -> pd.DataFrame:
    """The needed columns only (names matched after stripping and lower-casing), as numbers."""
    if os.path.realpath(os.path.dirname(os.path.abspath(path))) == CDM_DIR and not allow_real:
        raise PermissionError("the real click file is analysed only after the coordinator's go-ahead (--real)")
    want = set(NEED)
    df = pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in want, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in NEED if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from the click file: {missing}")
    return df[NEED].apply(pd.to_numeric, errors="coerce")


def included(D: pd.DataFrame) -> np.ndarray:
    """Amendment 7 item 1: the authors' standard data set, Experiments 1 and 2."""
    n = D["numopts"]
    sets = ((n == 10) & D["rndcode"].isin([1, 2, 3, 4, 5, 6])) | (n.isin([20, 40]) & D["rndcode"].notna())
    return (D["tracking"].isin([0, 1]) & (D["practice"] == 0) & (D["submit"] == 1) & D["rndmaxnumoper"].isin([3, 7])
            & ~D["treatid"].isin([21, 24, 25]) & sets).to_numpy()


def build_experiment(E: pd.DataFrame, name: str) -> dict:
    """Rounds and selections of one experiment, with consistency checks that stop the analysis on failure."""
    if E[["userid", "rndnum"]].isna().any().any():
        raise ValueError(f"{name}: an included row lacks its participant or round number")
    E = E.sort_values(["userid", "rndnum", "slctnum"], kind="mergesort", na_position="last").reset_index(drop=True)
    rid = E.groupby(["userid", "rndnum"], sort=True).ngroup().to_numpy()
    first_row = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    for col in ROUND_LEVEL:  # round-level columns must not vary within a round
        x = E[col].to_numpy(float)
        x0 = x[first_row][rid]
        if not np.all((x == x0) | (np.isnan(x) & np.isnan(x0))):
            raise ValueError(f"{name}: column {col} varies within a round")
    numslct = E["numslct"].to_numpy(float)[first_row]
    is_sel = E["slctnum"].notna().to_numpy()
    nsel_rows = np.bincount(rid, weights=is_sel.astype(float), minlength=len(first_row))
    has_sel = np.nan_to_num(numslct, nan=0.0) >= 1
    if np.any(nsel_rows[~has_sel] > 0):
        raise ValueError(f"{name}: a round without selections holds selection rows")
    if np.any(nsel_rows[has_sel] != numslct[has_sel]):
        raise ValueError(f"{name}: a round's selection rows do not match its number of selections")
    keep_round = has_sel
    rows = is_sel & keep_round[rid]
    S = E.loc[rows].reset_index(drop=True)
    old_rid = rid[rows]
    new_index = np.cumsum(keep_round) - 1
    srid = new_index[old_rid]  # round of each selection, rounds renumbered
    R = int(keep_round.sum())
    fr = first_row[keep_round]
    n = E["numopts"].to_numpy(float)[fr].astype(int)
    cplx = E["rndmaxnumoper"].to_numpy(float)[fr].astype(int)
    nsl = numslct[keep_round].astype(int)
    userid = E["userid"].to_numpy()[fr]
    allvals = E[VALS].to_numpy(float)[fr]
    slct = S["slctnum"].to_numpy(float)
    new_round = np.r_[True, srid[1:] != srid[:-1]]
    starts = np.flatnonzero(new_round)
    pos_in_round = np.arange(len(S)) - starts[np.cumsum(new_round) - 1]  # 0, 1, ... within each round
    if not np.array_equal(slct, pos_in_round + 1.0):
        raise ValueError(f"{name}: selection numbers do not run 1 .. number of selections within a round")
    optord = S["optord"].to_numpy(float)
    n_s = n[srid]
    if not np.all((optord == np.floor(optord)) & (optord >= 2) & (optord <= n_s + 1)):
        raise ValueError(f"{name}: a selection's position lies outside 2 .. n + 1")
    j_s = optord.astype(int) - 2  # option index 0 .. n - 1 in list order
    groups, Vlist = {}, [None] * R
    for size in SIZES:
        idx = np.flatnonzero(n == size)
        V = allvals[idx, 1:size + 1]  # positions 2 .. n + 1
        if not np.isfinite(V).all():
            raise ValueError(f"{name}: an option value is missing")
        groups[size] = idx
        for k, r in enumerate(idx):
            Vlist[r] = V[k]
    if sum(len(g) for g in groups.values()) != R:
        raise ValueError(f"{name}: an included round has a set size other than 10, 20 or 40")
    if not np.all(np.isin(cplx, CPLX)):
        raise ValueError(f"{name}: complexity outside 3 and 7")
    for V in Vlist:  # the amendment's dollar scale (MLBA's 6/35) assumes whole dollars from 1 to 35
        if not np.all((V == np.floor(V)) & (V >= 1) & (V <= 35)):
            raise ValueError(f"{name}: an option value is not a whole number of dollars from 1 to 35")
    present = set(zip(n.tolist(), cplx.tolist()))
    if any((s, c) not in present for s in SIZES for c in CPLX):
        raise ValueError(f"{name}: a cell of set size and complexity has no included round")
    v_s = np.array([Vlist[r][j] for r, j in zip(srid, j_s)])
    valnum = S["valnum"].to_numpy(float)
    if not np.all(np.abs(valnum - v_s) <= 1e-9 * np.maximum(1.0, np.abs(v_s))):
        raise ValueError(f"{name}: a selection's value differs from the value listed at its position")
    best = np.array([Vlist[r].max() for r in srid])
    n_better = np.array([(Vlist[r] > v).sum() for r, v in zip(srid, v_s)])
    t_s = S["timeslct"].to_numpy(float)
    last = slct == nsl[srid]
    # round-level quantities
    first_sel = np.flatnonzero(slct == 1)
    last_sel = np.flatnonzero(last)
    if not (np.array_equal(srid[first_sel], np.arange(R)) and np.array_equal(srid[last_sel], np.arange(R))):
        raise ValueError(f"{name}: every round needs one first and one last selection")
    chosen = j_s[last_sel]
    place = j_s[first_sel] / (n - 1.0)
    t_final = t_s[last_sel]
    # participants and folds
    ids = np.unique(userid)
    pidx = np.searchsorted(ids, userid)
    P = len(ids)
    perm = np.random.default_rng(FOLD_SEED).permutation(P)  # fresh generator per experiment
    fold_of = np.empty(P, int)
    fold_of[perm] = np.arange(P) % N_FOLDS
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ok = np.isfinite(t_s) & (t_s > 0)
        logt = np.where(t_ok, np.log(np.where(t_ok, t_s, 1.0)), np.nan)
        tf_ok = np.isfinite(t_final) & (t_final > 0)
    size_idx = np.searchsorted(SIZES, n)
    cplx_idx = np.searchsorted(CPLX, cplx)
    return dict(  # ids and round numbers stay in memory for the checks and are never written out
        name=name, R=R, P=P, ids=ids, pidx=pidx, rndnum=E["rndnum"].to_numpy(float)[fr], fold_of=fold_of, fold=fold_of[pidx], n=n, cplx=cplx,
        size_idx=size_idx, cplx_idx=cplx_idx, cell=size_idx * 2 + cplx_idx, groups=groups, Vlist=Vlist,
        chosen=chosen, place=place, t_final=t_final, tf_ok=tf_ok, logt_final=np.where(tf_ok, logt[last_sel], np.nan),
        # selections
        s_round=srid, s_pidx=pidx[srid], s_fold=fold_of[pidx][srid], s_cell=(size_idx * 2 + cplx_idx)[srid],
        s_last=last.astype(float), s_value=v_s, s_share=n_better / n_s, s_scored=v_s < best, s_logt=logt, s_t_ok=t_ok,
        diag=dict(share_rounds_without_selection=float(1.0 - keep_round.mean()),
                  share_scored_selections_without_valid_time=float(1.0 - t_ok[v_s < best].mean()) if (v_s < best).any() else 0.0,
                  share_rounds_without_valid_final_time=float(1.0 - tf_ok.mean()),
                  smallest_option_value=float(min(v.min() for v in Vlist)), largest_option_value=float(max(v.max() for v in Vlist))),
    )


def unsubmitted_shares(D: pd.DataFrame, tracking: int) -> list:
    """Share of rounds left out as not submitted (submit other than 1, missing included), among the rounds that
    meet every other inclusion rule of item 1, per cell in the order (10,3),(10,7),(20,3),(20,7),(40,3),(40,7)."""
    n = D["numopts"]
    sets = ((n == 10) & D["rndcode"].isin([1, 2, 3, 4, 5, 6])) | (n.isin([20, 40]) & D["rndcode"].notna())
    base = ((D["tracking"] == tracking) & (D["practice"] == 0) & D["rndmaxnumoper"].isin(CPLX)
            & ~D["treatid"].isin([21, 24, 25]) & sets & D["rndnum"].notna())
    R = D.loc[base, ["userid", "rndnum", "numopts", "rndmaxnumoper", "submit"]].groupby(["userid", "rndnum"], sort=True).first()
    out = []
    for s in SIZES:
        for c in CPLX:
            cell = R[(R["numopts"] == s) & (R["rndmaxnumoper"] == c)]
            out.append(float((cell["submit"] != 1).mean()) if len(cell) else float("nan"))
    return out


def load_cdm(path: str, allow_real: bool = False) -> dict:
    D = read_clicks(path, allow_real)
    keep = included(D)
    data = {name: build_experiment(D.loc[keep & (D["tracking"] == t).to_numpy()], name) for name, t in EXPERIMENTS}
    for name, t in EXPERIMENTS:
        data[name]["unsubmitted"] = unsubmitted_shares(D, t)
    return data


# ============================================================================ scores of the static models
def dbs_scores(V: np.ndarray) -> np.ndarray:
    n = V.shape[1]
    return np.sign(V[:, :, None] - V[:, None, :]).sum(axis=2) / (n - 1)


def mlba_scores(V: np.ndarray, scale: float = MLBA_SCALE) -> np.ndarray:
    n = V.shape[1]
    d = scale * (V[:, :, None] - V[:, None, :])  # the option itself has d = 0 and adds nothing
    return (np.exp(-np.abs(d)) * d).sum(axis=2) / (n - 1)


def static_scores(V: np.ndarray) -> dict:
    return {"CL": V, "DbS": dbs_scores(V), "MLBA": mlba_scores(V)}


# ============================================================================ softmax and sample until return
def log_softmax(lam: float, S: np.ndarray) -> np.ndarray:
    z = lam * S
    zmax = z.max(axis=1, keepdims=True)
    return z - zmax - np.log(np.exp(z - zmax).sum(axis=1, keepdims=True))


def esym(p: np.ndarray) -> np.ndarray:
    """f_k = k! e_k(p) of every row, k = 0 .. m, returned as (T, m + 1). Only the nonzero part is updated."""
    T, m = p.shape
    f = np.zeros((m + 1, T))
    f[0] = 1.0
    k = np.arange(1, m + 1, dtype=float)[:, None]
    pt = np.ascontiguousarray(p.T)
    for j in range(m):
        upd = k[:j + 1] * pt[j] * f[:j + 1]  # computed from the old values before the update
        f[1:j + 2] += upd
    return f.T


def _drop(x: np.ndarray, col: np.ndarray) -> np.ndarray:
    T, n = x.shape
    mask = np.ones((T, n), bool)
    mask[np.arange(T), col] = False
    return x[mask].reshape(T, n - 1)


def sr_log_chosen(lam: float, V: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """log P_SR(chosen option), floored at log(1e-300)."""
    logq = log_softmax(lam, V)
    q = np.exp(logq)
    g = esym(_drop(q, chosen))
    lp = 2.0 * logq[np.arange(len(chosen)), chosen] + np.log(g @ np.arange(1.0, V.shape[1] + 1.0))
    return np.maximum(lp, LOG_FLOOR)


def sr_log_probs(lam: float, V: np.ndarray) -> np.ndarray:
    """log race probabilities of every option (T, n)."""
    T, n = V.shape
    return np.column_stack([sr_log_chosen(lam, V, np.full(T, i)) for i in range(n)])


def sr_expected_looks(lam: float, V: np.ndarray) -> np.ndarray:
    return esym(np.exp(log_softmax(lam, V))).sum(axis=1)


def attention_place(lam: float, V: np.ndarray) -> np.ndarray:
    """Mean place of SR's attention, sum_j softmax(lam v)_j j / (n - 1)."""
    n = V.shape[1]
    return np.exp(log_softmax(lam, V)) @ np.arange(n, dtype=float) / (n - 1.0)


# ============================================================================ grouped data and fits
class Groups:
    """Rounds of one experiment grouped by set size: values, chosen options and static scores."""

    def __init__(self, X: dict):
        self.R = X["R"]
        self.items = []  # (size, round indices, V, chosen)
        for size in SIZES:
            idx = X["groups"][size]
            if len(idx):
                V = np.array([X["Vlist"][r] for r in idx])
                self.items.append((size, idx, V, X["chosen"][idx]))
        self.S = {m: [static_scores(V)[m] for _, _, V, _ in self.items] for m in STATIC}

    def per_round(self, fn) -> np.ndarray:
        out = np.zeros(self.R)
        for _, idx, V, c in self.items:
            out[idx] = fn(V, c)
        return out


def fit_softmax_groups(S_list, items, w: np.ndarray, init: float | None = None, rtol: float = 1e-13) -> float:
    """Weighted ML of lambda >= 0 for softmax(lambda s) over rounds of several set sizes: safeguarded Newton
    root finding on the score of the concave likelihood; lambda = 0 when the score at zero is not positive."""
    data = []
    for S, (_, idx, _, c) in zip(S_list, items):
        wg = w[idx]
        on = wg > 0
        if on.any():
            Sg = S[on]
            data.append((Sg, Sg[np.arange(on.sum()), c[on]], wg[on]))

    def score(lam):
        g = h = 0.0
        for Sg, sc_, wg in data:
            q = np.exp(log_softmax(lam, Sg))
            m = (q * Sg).sum(axis=1)
            g += float(wg @ (sc_ - m))
            h -= float(wg @ (q * (Sg - m[:, None]) ** 2).sum(axis=1))
        return g, h

    g0, _ = score(0.0)
    spread = sum(float(wg @ (Sg.max(axis=1) - Sg.min(axis=1))) for Sg, _, wg in data)
    if spread <= 0 or g0 <= 1e-13 * spread:
        return 0.0
    lo, hi = 0.0, (1.5 * init if init else 1.0)
    while score(hi)[0] > 0:
        lo, hi = hi, 2 * hi
        if hi > 1e8:
            raise RuntimeError("lambda diverges")
    x = init if (init and lo < init < hi) else 0.5 * (lo + hi)
    for _ in range(300):
        g, h = score(x)
        if g > 0:
            lo = x
        elif g < 0:
            hi = x
        step = -g / h if h < 0 else np.inf
        if np.isfinite(step) and abs(step) <= rtol * abs(x):
            return x + step
        xn = x + step
        if not (lo < xn < hi):
            xn = 0.5 * (lo + hi)
        if hi - lo <= 4 * np.finfo(float).eps * max(abs(xn), 1e-300):
            return xn
        x = xn
    raise RuntimeError("softmax fit did not converge")


class SRFitGroups:
    """SR's lambda by the procedure of Amendment 5: the grid of zero and 21 values spaced evenly in log between
    0.01 and 100, the best grid value refined by bounded Brent search (xatol 1e-10) between its neighbours.
    Per-round log likelihoods on the grid and on a dense diagnostic grid are tabulated once (weight free)."""

    def __init__(self, G: Groups):
        self.items = G.items
        self.tab = np.zeros((len(SR_GRID), G.R))
        self.tab_dense = np.zeros((len(DENSE_GRID), G.R))
        for _, idx, V, c in self.items:
            for i, x in enumerate(SR_GRID):
                self.tab[i, idx] = sr_log_chosen(x, V, c)
            for i, x in enumerate(DENSE_GRID):
                self.tab_dense[i, idx] = sr_log_chosen(x, V, c)

    def loglik(self, lam: float, w: np.ndarray) -> float:
        tot = 0.0
        for _, idx, V, c in self.items:
            wg = w[idx]
            on = wg > 0
            if on.any():
                tot += float(wg[on] @ sr_log_chosen(lam, V[on], c[on]))
        return tot

    def fit(self, w: np.ndarray):
        LLg = self.tab @ w
        g = int(np.argmax(LLg))
        lo, hi = SR_GRID[max(g - 1, 0)], SR_GRID[min(g + 1, len(SR_GRID) - 1)]
        res = minimize_scalar(lambda x: -self.loglik(x, w), bounds=(lo, hi), method="bounded",
                              options={"xatol": SR_XATOL, "maxiter": 2000})
        lam, ll = float(res.x), -float(res.fun)
        return lam, {"ll": ll, "dense_excess": float((self.tab_dense @ w).max() - ll), "grid_index": g}


# ============================================================================ stopping rules
def wstandardize(x: np.ndarray, w: np.ndarray):
    """Weighted mean and population standard deviation."""
    W = w.sum()
    m = float(w @ x / W)
    return m, float(np.sqrt(w @ (x - m) ** 2 / W))


def zscore(x: np.ndarray, m: float, sd: float) -> np.ndarray:
    return (x - m) / sd if sd > 0 else np.zeros_like(x)


def logit_objective(beta: np.ndarray, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    eta = X @ beta
    return float(w @ (np.logaddexp(0.0, eta) - y * eta) + 0.5 * beta[1:] @ beta[1:])


def fit_logit(Z: np.ndarray, y: np.ndarray, w: np.ndarray, tol: float = 1e-11, max_iter: int = 200) -> np.ndarray:
    """Minimise -sum w [y log p + (1 - y) log(1 - p)] + 0.5 sum_k b_k^2 (slopes only, the intercept free) by
    Newton's method with step halving. Z (N, K) holds standardized predictors. When every training outcome is 0
    (1) the objective has no minimum; its infimum is the limit intercept -inf (+inf) with zero slopes."""
    N, K = Z.shape
    W = w.sum()
    ybar = float(w @ y / W)
    if ybar <= 0.0 or ybar >= 1.0:
        return np.r_[-np.inf if ybar <= 0.0 else np.inf, np.zeros(K)]
    X = np.column_stack([np.ones(N), Z])
    pen = np.eye(K + 1)
    pen[0, 0] = 0.0
    beta = np.zeros(K + 1)
    beta[0] = np.log(ybar / (1.0 - ybar))
    f = logit_objective(beta, X, y, w)
    for _ in range(max_iter):
        eta = X @ beta
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))  # logistic, stable
        g = X.T @ (w * (p - y)) + pen @ beta
        H = (X.T * (w * p * (1.0 - p))) @ X + pen
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            nb = beta - t * step
            nf = logit_objective(nb, X, y, w)
            if nf <= f + 1e-13 * (1.0 + abs(f)) or t < 1e-12:
                break
            t *= 0.5
        beta, f = nb, nf
        if np.max(np.abs(t * step)) <= tol * (1.0 + np.max(np.abs(beta))):
            return beta
    raise RuntimeError("logistic fit did not converge")


def logit_log2_prob(beta: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """log2 of the probability given to the observed outcome, floored at 1e-300."""
    eta = beta[0] + (Z @ beta[1:] if Z.shape[1] else 0.0)
    with np.errstate(invalid="ignore"):
        lp = np.where(y > 0.5, -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta))
    return np.maximum(lp, LOG_FLOOR) / LN2
