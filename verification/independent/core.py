"""Independent reimplementation of the pre-registered choice-model analysis (R0 and H1).

Written from ANALYSIS_PLAN.md (Sections 2 to 5, Section 7 for R0 and H1, Amendments 1 and 2)
and VERIFY_BRIEF.md only. The only code reused is the data definition in coding_definition.py.

Privacy: nothing in this module prints or stores identifiers, participant counts or choice counts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
if VERIFY not in sys.path:
    sys.path.insert(0, VERIFY)
import coding_definition as cd  # noqa: E402

CSV_PATH = os.path.join(VERIFY, "Updated_Choice_Overload.csv")
TASKS = ("file", "laptop")
TASK_INDEX = {"file": 0, "laptop": 1}
ATTRS = {"file": cd.FILE_ATTRIBUTES, "laptop": cd.LAPTOP_ATTRIBUTES}
N_RUBRIC_ATTRS = {"file": 5, "laptop": 6}  # laptop price left out of the NNLS, share 0
BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
KAPPA0 = 1.0
ESAMP_DRAWS = 2000


def beta_name(base: str, beta: float) -> str:
    return f"{base}|beta={float(beta) + 0.0:+.1f}"


SOFTMAX_MODELS = ["QR", "DbS_sum", "MLBA_sum"] + [
    beta_name(base, b) for base in ("CL", "DbS_avg", "MLBA_avg") for b in BETAS
]
EBA_MODELS = ["EBA", "EBA_best"]
ALL_MODELS = SOFTMAX_MODELS + EBA_MODELS
R0_ALIAS = {
    "QR": "QR",
    "CL": beta_name("CL", 0.0),
    "DbS_avg": beta_name("DbS_avg", 0.0),
    "MLBA_avg": beta_name("MLBA_avg", 0.0),
    "DbS_sum": "DbS_sum",
    "MLBA_sum": "MLBA_sum",
}


# ============================================================================ data
def load_trials(task: str, stimuli_file: str | None = None) -> list[dict]:
    """``stimuli_file`` names a definition file in verify/stimuli (default {task}_trials.json)."""
    fname = stimuli_file or f"{task}_trials.json"
    with open(os.path.join(VERIFY, "stimuli", fname)) as fh:
        trials = json.load(fh)  # insertion order = trial index 0..5
    with open(os.path.join(VERIFY, "stimuli", "rubric.json")) as fh:
        rubric = json.load(fh)[task]
    out = []
    for name, t in trials.items():
        options = t["options"]
        names = list(options.keys())
        X = cd.code_file_trial(options) if task == "file" else cd.code_laptop_trial(options)
        U = np.array([float(rubric[t["rubric_key"]][o]) for o in names])
        out.append(
            dict(
                name=name,
                column=t["column"],
                info=t["information"],
                size=int(t["intended_options"]),
                names=names,
                X=np.asarray(X, float),
                U=U,
                n=len(names),
            )
        )
    return out


def load_choice_matrix(trials: list[dict], csv_path: str = CSV_PATH) -> np.ndarray:
    """Matched option index per (panel member, trial), -1 when no matched choice.

    Only the six response columns are read, so no identifier ever enters memory.
    """
    cols = [t["column"] for t in trials]
    df = pd.read_csv(csv_path, usecols=cols)
    rows = len(df)
    C = np.full((rows, len(trials)), -1, dtype=int)
    for c, t in enumerate(trials):
        lookup = {nm.lower(): i for i, nm in enumerate(t["names"])}
        assert len(lookup) == len(t["names"]), "option names not unique after lowercasing"
        for r, v in enumerate(df[t["column"]].tolist()):
            if isinstance(v, str):
                i = lookup.get(v.strip().lower())
                if i is not None:
                    C[r, c] = i
    panel = (C >= 0).any(axis=1)  # at least one matched choice in this task
    return C[panel]  # data file order is preserved


# ============================================================================ rubric shares
def rubric_shares(trials: list[dict], k_used: int) -> np.ndarray:
    A, b = [], []
    for t in trials:
        X = t["X"][:, :k_used]
        A.append(X - X.mean(axis=0))
        b.append(t["U"] - t["U"].mean())
    w, _ = nnls(np.vstack(A), np.concatenate(b))
    r = np.zeros(trials[0]["X"].shape[1])
    r[:k_used] = w / w.sum()
    return r


# ============================================================================ model scores
def comparison_scores(X: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Summed DbS and MLBA scores. The j = i term has d = 0 and contributes nothing."""
    d = X[:, None, :] - X[None, :, :]  # d[i, j, k] = x_ik - x_jk
    dbs = (np.sign(d) * r).sum(axis=(1, 2))
    mlba = (np.exp(-KAPPA0 * np.abs(d)) * d * r).sum(axis=(1, 2))
    return dbs, mlba


def trial_scores(t: dict, r: np.ndarray) -> dict[str, np.ndarray]:
    n = t["n"]
    cl = t["X"] @ r
    dbs_sum, mlba_sum = comparison_scores(t["X"], r)
    base = {"CL": cl, "DbS_avg": dbs_sum / (n - 1), "MLBA_avg": mlba_sum / (n - 1)}
    out = {"QR": t["U"].copy(), "DbS_sum": dbs_sum, "MLBA_sum": mlba_sum}
    for name, v in base.items():
        for b in BETAS:
            out[beta_name(name, b)] = ((n - 1) / 7.0) ** b * v
    return out


# ============================================================================ elimination by aspects
def eba_aspects(X: np.ndarray, r: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray]:
    ks = np.where(r > 0)[0]
    Xk = X[:, ks]
    if rule == "median":
        H = Xk >= np.median(Xk, axis=0)
    elif rule == "best":
        H = Xk == Xk.max(axis=0)
    else:
        raise ValueError(rule)
    return H, r[ks]


def eba_from_aspects(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Choice probabilities of Tversky's EBA for a fixed aspect matrix H (options x aspects)."""
    n = H.shape[0]
    memo: dict[tuple, np.ndarray] = {}

    def prob(S: tuple) -> np.ndarray:
        if S in memo:
            return memo[S]
        held = H[list(S)].sum(axis=0)
        splits = (held > 0) & (held < len(S))  # some but not all options in S have it
        out = np.zeros(n)
        if not splits.any():
            out[list(S)] = 1.0 / len(S)
        else:
            total = w[splits].sum()
            for k in np.where(splits)[0]:
                survivors = tuple(i for i in S if H[i, k])
                out += (w[k] / total) * prob(survivors)
        memo[S] = out
        return out

    return prob(tuple(range(n)))


def eba_probs(X: np.ndarray, r: np.ndarray, rule: str) -> np.ndarray:
    H, w = eba_aspects(X, r, rule)
    return eba_from_aspects(H, w)


# ============================================================================ task container
def build_task(task: str, rubric_trials: list[int] | None = None, stimuli_file: str | None = None) -> dict:
    """All six trials are always modelled; ``rubric_trials`` only restricts the NNLS for r."""
    trials = load_trials(task, stimuli_file)
    C = len(trials)
    I = max(t["n"] for t in trials)
    valid = np.zeros((C, I), bool)
    for c, t in enumerate(trials):
        valid[c, : t["n"]] = True
    stack = trials if rubric_trials is None else [trials[i] for i in rubric_trials]
    r = rubric_shares(stack, N_RUBRIC_ATTRS[task])
    V = np.zeros((len(SOFTMAX_MODELS), C, I))
    for c, t in enumerate(trials):
        sc = trial_scores(t, r)
        for m, name in enumerate(SOFTMAX_MODELS):
            V[m, c, : t["n"]] = sc[name]
    A = np.zeros((len(EBA_MODELS), C, I))
    for c, t in enumerate(trials):
        A[0, c, : t["n"]] = eba_probs(t["X"], r, "median")
        A[1, c, : t["n"]] = eba_probs(t["X"], r, "best")
    choice = load_choice_matrix(trials)
    P = choice.shape[0]
    Y = np.zeros((P, C, I))
    for c in range(C):
        rows = np.where(choice[:, c] >= 0)[0]
        Y[rows, c, choice[rows, c]] = 1.0
    return dict(
        task=task,
        trials=trials,
        names=[t["name"] for t in trials],
        nvec=np.array([t["n"] for t in trials], float),
        sizes=np.array([t["size"] for t in trials]),
        info=np.array([t["info"] for t in trials]),
        valid=valid,
        r=r,
        V=V,
        A=A,
        Y=Y,
        P=P,
    )


def counts_from_weights(task: dict, w: np.ndarray) -> np.ndarray:
    """Choice counts per (trial, option) when panel member p enters w[p] times."""
    return np.tensordot(w, task["Y"], axes=(0, 0))


# ============================================================================ fitting
def fold_masks(C: int, include_full: bool) -> np.ndarray:
    """Row f < C trains on every trial but f; the optional last row trains on all trials."""
    T = ~np.eye(C, dtype=bool)
    if include_full:
        T = np.vstack([T, np.ones((1, C), bool)])
    return T


def _softmax_weights(lam: np.ndarray, V: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """lam (M, F), V (M, C, I) -> softmax weights (M, F, C, I), zero on padded options."""
    Z = lam[:, :, None, None] * V[:, None, :, :]
    Z = np.where(valid[None, None], Z, -np.inf)
    Z = Z - Z.max(axis=3, keepdims=True)
    W = np.exp(Z)
    return W / W.sum(axis=3, keepdims=True)


def softmax_score(lam, V, valid, cnt, T):
    """Score (first derivative of the pooled log likelihood) and its derivative, per (model, fold)."""
    N = cnt.sum(axis=1)
    Tf = T.astype(float)
    a = (V * cnt[None]).sum(axis=2)  # (M, C)
    W = _softmax_weights(lam, V, valid)
    Vb = V[:, None]
    E = (W * Vb).sum(axis=3)
    Var = (W * (Vb - E[..., None]) ** 2).sum(axis=3)
    S = a @ Tf.T - np.einsum("mfc,c,fc->mf", E, N, Tf)
    D = -np.einsum("mfc,c,fc->mf", Var, N, Tf)
    return S, D


def softmax_loglik(lam, V, valid, cnt, T):
    W = _softmax_weights(lam, V, valid)
    with np.errstate(divide="ignore"):
        logW = np.where(cnt[None, None] > 0, np.log(np.where(W > 0, W, 1.0)), 0.0)
    ll = (logW * cnt[None, None]).sum(axis=3)  # (M, F, C)
    return np.einsum("mfc,fc->mf", ll, T.astype(float))


def fit_softmax(V, valid, cnt, T, rtol=1e-15, maxit=400, init=None):
    """Maximum likelihood sensitivity lambda >= 0 for every (model, training set).

    The score is decreasing in lambda (the log likelihood is concave). lambda = 0 when the score at
    zero is not positive. Otherwise the root is bracketed by doubling and found by Newton steps
    safeguarded by bisection, until the step is below rtol relative to lambda. ``init`` (M, F) is an
    optional warm start; it only changes where the bracket search starts, not the root.
    """
    M, F = V.shape[0], T.shape[0]
    N = cnt.sum(axis=1)
    big = np.where(valid, V, -np.inf).max(axis=2)
    small = np.where(valid, V, np.inf).min(axis=2)
    spread = big - small  # (M, C)
    S0, _ = softmax_score(np.zeros((M, F)), V, valid, cnt, T)
    mag = np.einsum("mc,c,fc->mf", spread, N, T.astype(float))
    zero = (S0 <= 1e-13 * mag) | (mag <= 0)
    smax = spread.max(axis=1)
    hi = (1.0 / np.where(smax > 0, smax, 1.0))[:, None] * np.ones((M, F))
    if init is not None:
        hi = np.where(init > 0, 1.5 * init, hi)
    hi = np.where(zero, 0.0, hi)
    lo = np.zeros((M, F))
    for _ in range(200):
        S_hi, _ = softmax_score(hi, V, valid, cnt, T)
        need = (~zero) & (S_hi > 0)
        if not need.any():
            break
        lo = np.where(need, hi, lo)
        hi = np.where(need, 2.0 * hi, hi)
    else:
        raise RuntimeError("no finite maximum: lambda diverges")
    x = 0.5 * (lo + hi)
    if init is not None:
        x = np.where((init > lo) & (init < hi), init, x)
    x = np.where(zero, 0.0, x)
    for _ in range(maxit):
        S, D = softmax_score(x, V, valid, cnt, T)
        lo = np.where((S > 0) & ~zero, x, lo)
        hi = np.where((S < 0) & ~zero, x, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = x - S / D
        bad = ~np.isfinite(xn) | (xn <= lo) | (xn >= hi)
        xn = np.where(bad, 0.5 * (lo + hi), xn)
        xn = np.where(zero, 0.0, xn)
        done = np.abs(xn - x) <= rtol * np.abs(xn)
        x = xn
        if done.all():
            break
    else:
        raise RuntimeError("softmax fit did not converge")
    return x


def lapse_score(eps, A, valid, cnt, nvec, T):
    """Score of the EBA lapse and its derivative, per (EBA variant, fold). eps (E, F)."""
    inv_n = (1.0 / nvec)[:, None]  # (C, 1)
    e = eps[:, :, None, None]
    P = (1.0 - e) * A[:, None] + e * inv_n[None, None]  # (E, F, C, I)
    g = (inv_n[None, None] - A[:, None])
    pos = (cnt > 0)[None, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pos, g / P, 0.0)
        S_ci = np.where(pos, cnt[None, None] * ratio, 0.0)
        D_ci = np.where(pos, -cnt[None, None] * ratio**2, 0.0)
    Tf = T.astype(float)
    S = np.einsum("efci,fc->ef", S_ci, Tf)
    D = np.einsum("efci,fc->ef", D_ci, Tf)
    return S, D


def lapse_loglik(eps, A, valid, cnt, nvec, T):
    inv_n = (1.0 / nvec)[:, None]
    e = eps[:, :, None, None]
    P = (1.0 - e) * A[:, None] + e * inv_n[None, None]
    pos = (cnt > 0)[None, None]
    with np.errstate(divide="ignore"):
        ll = np.where(pos, cnt[None, None] * np.log(np.where(pos, P, 1.0)), 0.0)
    return np.einsum("efci,fc->ef", ll, T.astype(float))


def fit_lapse(A, valid, cnt, nvec, T, rtol=1e-15, maxit=400):
    """Maximum likelihood lapse epsilon in [0, 1] (concave log likelihood)."""
    E_, F = A.shape[0], T.shape[0]
    S0, _ = lapse_score(np.zeros((E_, F)), A, valid, cnt, nvec, T)
    S1, _ = lapse_score(np.ones((E_, F)), A, valid, cnt, nvec, T)
    at0 = S0 <= 0
    at1 = (~at0) & (S1 >= 0)
    inner = ~(at0 | at1)
    lo = np.zeros((E_, F))
    hi = np.ones((E_, F))
    x = np.full((E_, F), 0.5)
    for _ in range(maxit):
        S, D = lapse_score(x, A, valid, cnt, nvec, T)
        lo = np.where(S > 0, x, lo)
        hi = np.where(S < 0, x, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = x - S / D
        bad = ~np.isfinite(xn) | (xn <= lo) | (xn >= hi)
        xn = np.where(bad, 0.5 * (lo + hi), xn)
        done = (np.abs(xn - x) <= rtol * np.abs(xn)) | ~inner
        x = xn
        if done.all():
            break
    else:
        raise RuntimeError("lapse fit did not converge")
    return np.where(at0, 0.0, np.where(at1, 1.0, x))


# ============================================================================ predictions and measures
def softmax_predict(lam_c: np.ndarray, V: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """lam_c (M, C) applied to trial c -> probabilities (M, C, I)."""
    Z = lam_c[:, :, None] * V
    Z = np.where(valid[None], Z, -np.inf)
    Z = Z - Z.max(axis=2, keepdims=True)
    W = np.exp(Z)
    return W / W.sum(axis=2, keepdims=True)


def lapse_predict(eps_c: np.ndarray, A: np.ndarray, valid: np.ndarray, nvec: np.ndarray) -> np.ndarray:
    Q = (1.0 - eps_c[:, :, None]) * A + eps_c[:, :, None] / nvec[None, :, None]
    return np.where(valid[None], Q, 0.0)


def log_scores(Q: np.ndarray, cnt: np.ndarray, nvec: np.ndarray) -> np.ndarray:
    """L_c in bits per choice above a uniform guess, for every model row of Q (M, C, I)."""
    N = cnt.sum(axis=1)
    pos = (cnt > 0)[None]
    with np.errstate(divide="ignore"):
        term = np.where(pos, cnt[None] * np.log2(np.where(pos, nvec[None, :, None] * Q, 1.0)), 0.0)
    return term.sum(axis=2) / N[None]


def overlaps(Q: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    p = cnt / cnt.sum(axis=1, keepdims=True)
    return np.minimum(p[None], Q).sum(axis=2)


def lstar(cnt: np.ndarray, nvec: np.ndarray) -> np.ndarray:
    N = cnt.sum(axis=1)
    p = cnt / N[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0).sum(axis=1)
    m = (cnt > 0).sum(axis=1)
    return np.log2(nvec) - H - (m - 1) / (2.0 * N * np.log(2.0))


def effective_number(p: np.ndarray) -> float:
    p = np.asarray(p, float)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_number_sampled(q: np.ndarray, N: int, seed: tuple) -> float:
    rng = np.random.default_rng(seed)
    q = np.asarray(q, float)
    s = rng.multinomial(int(N), q / q.sum(), size=ESAMP_DRAWS) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(s > 0, s * np.log(np.where(s > 0, s, 1.0)), 0.0).sum(axis=1)
    return float(np.exp(H).mean())


# ============================================================================ one sample
def evaluate(task: dict, cnt: np.ndarray, include_full: bool, init=None) -> dict:
    """Fit every model in every fold of one sample (original or replicate) and score it."""
    V, A, valid, nvec = task["V"], task["A"], task["valid"], task["nvec"]
    C = cnt.shape[0]
    T = fold_masks(C, include_full)
    lam = fit_softmax(V, valid, cnt, T, init=init)
    eps = fit_lapse(A, valid, cnt, nvec, T)
    idx = np.arange(C)
    Q_ho = np.concatenate(
        [softmax_predict(lam[:, idx], V, valid), lapse_predict(eps[:, idx], A, valid, nvec)], axis=0
    )
    out = dict(
        lam=lam,
        eps=eps,
        Q_ho=Q_ho,
        L_ho=log_scores(Q_ho, cnt, nvec),
        O_ho=overlaps(Q_ho, cnt),
    )
    if include_full:
        full = np.full(C, C)
        Q_fit = np.concatenate(
            [softmax_predict(lam[:, full], V, valid), lapse_predict(eps[:, full], A, valid, nvec)], axis=0
        )
        out["Q_fit"] = Q_fit
        out["L_fit"] = log_scores(Q_fit, cnt, nvec)
    return out


def b1_components(task: dict, cnt: np.ndarray, q_cl: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """E(p_c) and E_samp(q_c) of every trial of one sample; b = -1 for the original sample."""
    ti = TASK_INDEX[task["task"]]
    C = cnt.shape[0]
    e_obs = np.zeros(C)
    e_mod = np.zeros(C)
    for c in range(C):
        n = int(task["nvec"][c])
        Nc = int(round(cnt[c].sum()))
        p = cnt[c, :n] / cnt[c, :n].sum()
        e_obs[c] = effective_number(p)
        e_mod[c] = effective_number_sampled(q_cl[c, :n], Nc, (1, ti, c, b + 1))
    return e_obs, e_mod


def b1_world_observed(task: dict, cnt: np.ndarray) -> np.ndarray:
    """E_samp(p_c) of the full sample shares, role 2, b = -1 (Amendment 2)."""
    ti = TASK_INDEX[task["task"]]
    out = np.zeros(cnt.shape[0])
    for c in range(cnt.shape[0]):
        n = int(task["nvec"][c])
        Nc = int(round(cnt[c].sum()))
        out[c] = effective_number_sampled(cnt[c, :n] / cnt[c, :n].sum(), Nc, (2, ti, c, 0))
    return out


# ============================================================================ summaries
def completeness(L: np.ndarray, Ls: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """K_S = 100 sum L_c / sum L*_c over the trials in mask (last axis of L is trials)."""
    if mask is None:
        mask = np.ones(L.shape[-1], bool)
    return 100.0 * L[..., mask].sum(axis=-1) / Ls[mask].sum()


def clear_rule(pi, bi) -> bool:
    return bool((pi[0] > 0 and bi[0] > 0) or (pi[1] < 0 and bi[1] < 0))


def boot_p(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    return float(min(1.0, 2.0 * min(np.mean(v <= 0), np.mean(v >= 0))))


def summary_level(t: float, v: np.ndarray) -> dict:
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"point": float(t), "percentile": [float(lo), float(hi)], "basic": [float(2 * t - hi), float(2 * t - lo)]}


def summary_diff(t: float, v: np.ndarray) -> dict:
    d = summary_level(t, v)
    d["p"] = boot_p(v)
    d["clear"] = clear_rule(d["percentile"], d["basic"])
    return d


def summary_centered(t: float, err: np.ndarray) -> dict:
    e_lo, e_hi = np.percentile(err, [2.5, 97.5])
    pi = [float(t + e_lo), float(t + e_hi)]
    bi = [float(t - e_hi), float(t - e_lo)]
    return {"point": float(t), "percentile": pi, "basic": bi, "p": boot_p(t + err), "clear": clear_rule(pi, bi)}


def holm(pvals) -> list[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return [float(x) for x in adj]


def ls_slope(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Least squares slope of every row of Y (..., C) on x (C,)."""
    xc = x - x.mean()
    return (Y - Y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)
