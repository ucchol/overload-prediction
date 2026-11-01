"""Independent reimplementation of test A3b of H3 (ANALYSIS_PLAN.md, Amendment 9), file selection only.

Written blind from the plan (Sections 3 to 5, Amendment 1, Amendment 9), VERIFY_BRIEF.md and the
caller's task statement. Reuses the earlier blind implementation in verification/independent/core.py
(coding, rubric shares, rule scores, softmax fit, log scores, L*). Nothing from the package under test
is imported except its data definition (coding_definition.py), as the earlier verifier was given it.

Usage:
  python -I h3.py --csv PATH --coding-dir DIR [--B 50] [--out DIR]
DIR holds coding_definition.py. Nothing here prints or writes identifiers or counts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
VERIF = os.path.dirname(HERE)
REPO = os.path.dirname(VERIF)
STIM = os.path.join(REPO, "src", "overload", "stimuli")  # data files only: json inputs
RULES = ("CL", "DbS_avg", "MLBA_avg")
ANCHOR = "which file is most worth taking"
SUBSETS = ("all", "low", "high")


# ============================================================================ cue
def name_words(name: str) -> list[str]:
    """Words of a file name: extension (text after the last dot) removed, split at non-letters,
    at lower-to-upper boundaries and before the last capital of a capital run followed by lowercase."""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    words = []
    for chunk in re.split(r"[^A-Za-z]+", stem):
        if chunk:
            words += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+", chunk)
    return [w.lower() for w in words]


def mean_vector(words, vec):
    known = [np.asarray(vec[w], float) for w in words if w in vec]
    return np.mean(known, axis=0) if known else None


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def trial_cues(names, vec) -> np.ndarray:
    anchor = mean_vector(ANCHOR.split(), vec)
    z = np.full(len(names), np.nan)
    for i, nm in enumerate(names):
        m = mean_vector(name_words(nm), vec)
        if m is not None:
            z[i] = cosine(m, anchor)
    missing = np.isnan(z)
    z[missing] = z[~missing].mean()  # mean cue of the other (known) names of the trial
    return z


# ============================================================================ task
def build_file_task(csv_path: str, core) -> dict:
    with open(os.path.join(STIM, "file_trials.json")) as fh:
        trials_def = json.load(fh)
    with open(os.path.join(STIM, "rubric.json")) as fh:
        rubric = json.load(fh)["file"]
    with open(os.path.join(STIM, "filename_glove.json")) as fh:
        vec = json.load(fh)["vectors"]
    trials = []
    for name, t in trials_def.items():
        opts = t["options"]
        names = list(opts.keys())
        trials.append(dict(
            name=name, column=t["column"], info=t["information"], size=int(t["intended_options"]),
            names=names, n=len(names),
            X=np.asarray(core.cd.code_file_trial(opts), float),
            U=np.array([float(rubric[t["rubric_key"]][o]) for o in names]),
        ))
    r = core.rubric_shares(trials, core.N_RUBRIC_ATTRS["file"])
    C, I = len(trials), max(t["n"] for t in trials)
    valid = np.zeros((C, I), bool)
    S = np.zeros((len(RULES), C, I))
    Z = np.zeros((C, I))
    cues = {}
    for c, t in enumerate(trials):
        valid[c, : t["n"]] = True
        sc = core.trial_scores(t, r)
        for m, rule in enumerate(RULES):
            S[m, c, : t["n"]] = sc[core.beta_name(rule, 0.0)]
        z = trial_cues(t["names"], vec)
        Z[c, : t["n"]] = z
        cues[t["name"]] = {nm: float(v) for nm, v in zip(t["names"], z)}
    choice = core.load_choice_matrix(trials, csv_path)
    P = choice.shape[0]
    Y = np.zeros((P, C, I))
    for c in range(C):
        rows = np.where(choice[:, c] >= 0)[0]
        Y[rows, c, choice[rows, c]] = 1.0
    return dict(trials=trials, names=[t["name"] for t in trials], info=np.array([t["info"] for t in trials]),
                nvec=np.array([t["n"] for t in trials], float), valid=valid, r=r, S=S, Z=Z, Y=Y, P=P, cues=cues)


# ============================================================================ two parameter fit
def _probs(lam, gam, s, z, valid):
    u = np.where(valid, lam * s + gam * z, -np.inf)
    u = u - u.max(axis=1, keepdims=True)
    w = np.exp(u)
    return w / w.sum(axis=1, keepdims=True)


def _nll_grad_hess(theta, s, z, valid, cnt):
    """Negative log likelihood, gradient and Hessian of softmax(lam s + gam z), pooled over trials."""
    lam, gam = theta
    q = _probs(lam, gam, s, z, valid)
    N = cnt.sum(axis=1)
    with np.errstate(divide="ignore"):
        lq = np.where(cnt > 0, np.log(np.where(q > 0, q, 1.0)), 0.0)
    nll = -(cnt * lq).sum()
    F = np.stack([np.where(valid, s, 0.0), np.where(valid, z, 0.0)])  # (2, C, I)
    Ef = (q[None] * F).sum(axis=2)  # (2, C)
    obs = (cnt[None] * F).sum(axis=2)
    g = -(obs - Ef * N[None]).sum(axis=1)
    Fc = F - Ef[..., None]
    H = np.einsum("aci,bci,ci,c->ab", Fc, Fc, q, N)
    return nll, g, H


def _newton(theta, fixed_lam0, s, z, valid, cnt, tol=1e-13, maxit=200):
    """Damped Newton on the concave log likelihood; with fixed_lam0 only gamma moves (lam = 0)."""
    theta = np.array(theta, float)
    if fixed_lam0:
        theta[0] = 0.0
    f, g, H = _nll_grad_hess(theta, s, z, valid, cnt)
    for _ in range(maxit):
        if fixed_lam0:
            step = np.array([0.0, -g[1] / H[1, 1]]) if H[1, 1] > 0 else np.zeros(2)
        else:
            step = -np.linalg.solve(H, g)
        t = 1.0
        while True:
            cand = theta + t * step
            fc, gc, Hc = _nll_grad_hess(cand, s, z, valid, cnt)
            if fc <= f + 1e-12 * abs(f) or t < 1e-12:
                break
            t *= 0.5
        done = np.all(np.abs(cand - theta) <= tol * np.maximum(1.0, np.abs(cand)))
        theta, f, g, H = cand, fc, gc, Hc
        if done:
            break
    return theta, f, g


def fit_cue(s, z, valid, cnt):
    """ML of (lam >= 0, gam free). Interior Newton; if lam < 0 there, the optimum is on lam = 0."""
    theta, f, g = _newton([0.0, 0.0], False, s, z, valid, cnt)
    if theta[0] < 0 or not np.isfinite(f):
        theta, f, g = _newton([0.0, 0.0], True, s, z, valid, cnt)
        assert g[0] >= -1e-8 * max(1.0, abs(f)), "KKT failed at lam = 0"
    return theta, f


def fit_cue_check(s, z, valid, cnt, theta):
    """Independent check by L-BFGS-B from four starts (used in the original sample only)."""
    fun = lambda th: _nll_grad_hess(th, s, z, valid, cnt)[:2]
    best = None
    for st in ([0, 0], [1, 1], [3, -3], [0.5, 10]):
        res = minimize(fun, np.array(st, float), jac=True, method="L-BFGS-B",
                       bounds=[(0, None), (None, None)], options=dict(ftol=1e-15, gtol=1e-10, maxiter=5000))
        if best is None or res.fun < best.fun:
            best = res
    f_newton = _nll_grad_hess(theta, s, z, valid, cnt)[0]
    return float(best.fun - f_newton), best.x


# ============================================================================ one sample
def evaluate(task, cnt, core, Lstar, check=False):
    C = cnt.shape[0]
    valid, nvec, S, Z = task["valid"], task["nvec"], task["S"], task["Z"]
    T = core.fold_masks(C, include_full=True)  # rows 0..5 hold out trial f, row 6 uses all
    lam = core.fit_softmax(S, valid, cnt, T)  # (3, 7)
    idx = np.arange(C)
    Q = core.softmax_predict(lam[:, idx], S, valid)  # (3, C, I) held out, no cue
    Qc = np.zeros_like(Q)
    gam_full = np.zeros(len(RULES))
    checks = []
    for m in range(len(RULES)):
        for f in range(C + 1):
            tr = T[f]
            theta, _ = fit_cue(S[m][tr], Z[tr], valid[tr], cnt[tr])
            if check:
                checks.append(fit_cue_check(S[m][tr], Z[tr], valid[tr], cnt[tr], theta)[0])
            if f < C:
                Qc[m, f] = _probs(theta[0], theta[1], S[m][f:f + 1], Z[f:f + 1], valid[f:f + 1])[0]
            else:
                gam_full[m] = theta[1]
    L = core.log_scores(Q, cnt, nvec)
    Lc = core.log_scores(Qc, cnt, nvec)
    masks = {"all": np.ones(C, bool), "low": task["info"] == "low", "high": task["info"] == "high"}
    out = {}
    for sub, mk in masks.items():
        K = {rule: core.completeness(L[m], Lstar, mk) for m, rule in enumerate(RULES)}
        Kc = {rule: core.completeness(Lc[m], Lstar, mk) for m, rule in enumerate(RULES)}
        spread = max(K.values()) - min(K.values())
        out[f"{sub}|spread"] = spread
        for rule in RULES:
            out[f"{sub}|K|{rule}"] = K[rule]
            out[f"{sub}|K|{rule}+cue"] = Kc[rule]
            out[f"{sub}|gain|{rule}"] = Kc[rule] - K[rule]
            out[f"{sub}|A3b|{rule}"] = Kc[rule] - K[rule] - spread
    for m, rule in enumerate(RULES):
        out[f"gamma|{rule}"] = gam_full[m]
    out = {k: float(v) for k, v in out.items()}
    extra = dict(lam=lam, L=L, Lc=Lc, checks=checks)
    return out, extra


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coding-dir", required=True)
    ap.add_argument("--B", type=int, default=50)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    sys.path.insert(0, a.coding_dir)
    sys.path.insert(0, os.path.join(VERIF, "independent"))
    import core  # noqa: E402  (imports coding_definition from coding_dir)

    t0 = time.time()
    task = build_file_task(a.csv, core)
    cnt0 = core.counts_from_weights(task, np.ones(task["P"]))
    Lstar = core.lstar(cnt0, task["nvec"])
    point, ex = evaluate(task, cnt0, core, Lstar, check=True)
    print("rubric shares", np.round(task["r"], 6).tolist())
    print("L*", np.round(Lstar, 6).tolist())
    print("lambda (held-out folds + full)", np.round(ex["lam"], 6).tolist())
    print("max(L-BFGS-B nll - Newton nll) over 21 fits", max(ex["checks"]), flush=True)
    reps = {}
    for b in range(a.B):
        rng = np.random.default_rng(10000 + b)
        draw = rng.integers(0, task["P"], size=task["P"])  # file panel first; laptop draws follow
        w = np.bincount(draw, minlength=task["P"]).astype(float)
        cnt = core.counts_from_weights(task, w)
        reps[str(b)], _ = evaluate(task, cnt, core, Lstar)
        if (b + 1) % 10 == 0:
            print(f"replicate {b + 1}/{a.B}, {time.time() - t0:.1f}s", flush=True)
    with open(os.path.join(a.out, "independent_results.json"), "w") as fh:
        json.dump({"point": point, "replicates": reps}, fh, indent=1)
    with open(os.path.join(a.out, "cue.json"), "w") as fh:
        json.dump(task["cues"], fh, indent=1)
    # summaries for the report (not part of the required output)
    summ = {}
    for k, t in point.items():
        v = np.array([reps[str(b)][k] for b in range(a.B)])
        summ[k] = core.summary_diff(t, v) if a.B > 0 else {"point": t}
    with open(os.path.join(a.out, "summary_B%d.json" % a.B), "w") as fh:
        json.dump(summ, fh, indent=1)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
