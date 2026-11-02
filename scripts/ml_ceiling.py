"""Machine learning ceiling for completeness on the open datasets (exploratory, docs/ml_ceiling_plan.md).

Usage, python3 -I scripts/ml_ceiling.py OUT [B]
"""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from model_recovery import probabilities  # noqa: E402
from overload.experiments.overload_premonition import groups  # noqa: E402

FOLD_SEED, FOLDS, INNER = 20261014, 10, 5
PENALTIES = (0.001, 0.01, 0.1, 1.0)
HIDDEN = 16
RANGE = {"snack": 6.0, "reutskaja": 10.0, "cdm": 34.0}
FEATURES = ("value", "minus_best", "minus_mean", "rank", "is_best", "gap_other", "position", "near_share", "log2n")


def subset(p, rows):
    return dataclasses.replace(p, values=p.values[rows], n=p.n[rows], choice=p.choice[rows], participant=p.participant[rows],
                               best_seen=None if p.best_seen is None else p.best_seen[rows])


def features(p):
    v = p.values
    ok = np.isfinite(v)
    rng_ = RANGE["cdm" if p.name.startswith("cdm") else p.name]
    vv = np.where(ok, v, -np.inf)
    top = vv.max(axis=1, keepdims=True)
    mean = np.nanmean(v, axis=1, keepdims=True)
    order = np.argsort(np.argsort(-vv, axis=1, kind="stable"), axis=1, kind="stable").astype(float)
    n = p.n[:, None].astype(float)
    srt = np.sort(vv, axis=1)[:, ::-1]
    second = srt[:, 1:2]
    other_best = np.where(vv == top, second, top)
    pos = np.arange(v.shape[1])[None, :] / np.maximum(n - 1, 1)
    near = np.stack([(np.abs(vv - vv[:, [j]]) <= 1).sum(axis=1) for j in range(v.shape[1])], axis=1) / n
    F = np.stack([np.nan_to_num(v) / rng_, (vv - top) / rng_, np.nan_to_num(v - mean) / rng_, order / np.maximum(n - 1, 1),
                  (vv == top).astype(float), (vv - other_best) / rng_, pos + 0 * v, near, np.log2(n) + 0 * v], axis=2)
    F = np.where(ok[:, :, None], F, 0.0)
    return np.nan_to_num(F, posinf=0.0, neginf=0.0), ok


# ------------------------------------------------------------------ neural network softmax
def _unpack(th, d):
    W1 = th[: d * HIDDEN].reshape(d, HIDDEN)
    b1 = th[d * HIDDEN: d * HIDDEN + HIDDEN]
    w2 = th[d * HIDDEN + HIDDEN:]
    return W1, b1, w2


def _scores(th, F):
    W1, b1, w2 = _unpack(th, F.shape[2])
    H = np.tanh(F @ W1 + b1)
    return H @ w2, H


def _loss(th, F, ok, ch, pen):
    d = F.shape[2]
    W1, b1, w2 = _unpack(th, d)
    s, H = _scores(th, F)
    s = np.where(ok, s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    e = np.where(ok, np.exp(s - m), 0.0)
    Z = e.sum(axis=1, keepdims=True)
    q = e / Z
    T = F.shape[0]
    idx = np.arange(T)
    nll = -(s[idx, ch] - m[:, 0] - np.log(Z[:, 0])).sum() / T
    f = nll + pen * (np.sum(W1 ** 2) + np.sum(w2 ** 2))
    ds = q.copy(); ds[idx, ch] -= 1.0; ds /= T
    gw2 = np.einsum("tn,tnh->h", ds, H) + 2 * pen * w2
    dZ = ds[:, :, None] * w2[None, None, :] * (1 - H ** 2)
    gW1 = np.einsum("tnd,tnh->dh", F, dZ) + 2 * pen * W1
    gb1 = dZ.sum(axis=(0, 1))
    return f, np.concatenate([gW1.ravel(), gb1, gw2])


def fit_nn(F, ok, ch, pen, seed, starts):
    d = F.shape[2]
    best = None
    for s in range(starts):
        rng = np.random.default_rng(seed * 10 + s)
        th0 = rng.normal(0, 0.3, d * HIDDEN + 2 * HIDDEN)
        r = minimize(_loss, th0, args=(F, ok, ch, pen), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def logp_nn(th, F, ok, ch):
    s, _ = _scores(th, F)
    s = np.where(ok, s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    lz = m[:, 0] + np.log(np.where(ok, np.exp(s - m), 0).sum(axis=1))
    return (s[np.arange(len(ch)), ch] - lz) / np.log(2)


def nn_outer(F, ok, ch, part, train, test, seed):
    """Choose the penalty by inner folds of the training participants, refit, return held out log2 p."""
    tp = np.unique(part[train])
    perm = np.random.default_rng(seed).permutation(tp)
    inner = {p: k % INNER for k, p in enumerate(perm)}
    ifold = np.array([inner[p] for p in part[train]])
    tr_idx = np.where(train)[0]
    score = {}
    for pen in PENALTIES:
        tot = 0.0
        for k in range(INNER):
            a, b = tr_idx[ifold != k], tr_idx[ifold == k]
            th = fit_nn(F[a], ok[a], ch[a], pen, seed + k, 1)
            tot += logp_nn(th, F[b], ok[b], ch[b]).sum()
        score[pen] = tot
    pen = max(score, key=score.get)
    th = fit_nn(F[train], ok[train], ch[train], pen, seed, 3)
    return logp_nn(th, F[test], ok[test], ch[test]), pen


# ------------------------------------------------------------------ boosted trees
def gbm_outer(F, ok, ch, train, test):
    T, N, d = F.shape
    X = F.reshape(T * N, d); keep = ok.reshape(-1)
    y = np.zeros((T, N)); y[np.arange(T), ch] = 1; y = y.reshape(-1)
    rows_tr = np.repeat(train, N) & keep
    clf = HistGradientBoostingClassifier(random_state=0).fit(X[rows_tr], y[rows_tr])
    p = np.zeros(T * N)
    rows_te = np.repeat(test, N) & keep
    p[rows_te] = clf.predict_proba(X[rows_te])[:, 1]
    P = p.reshape(T, N)[test]
    P = P / P.sum(axis=1, keepdims=True)
    return np.log2(P[np.arange(test.sum()), ch[test]])


# ------------------------------------------------------------------ one parameter models
def fit_model(p, model):
    def nll(loglam):
        P = probabilities(p, model, float(np.exp(loglam)))
        return -np.log(np.maximum(P[np.arange(len(p.choice)), p.choice], 1e-300)).sum()
    r = minimize_scalar(nll, bounds=(-8, 4), method="bounded", options={"xatol": 1e-8})
    return float(np.exp(r.x))


def logp_model(p, model, lam):
    P = probabilities(p, model, lam)
    return np.log2(np.maximum(P[np.arange(len(p.choice)), p.choice], 1e-300))


def panel_scores(p, seed):
    F, ok = features(p)
    ch = p.choice
    P = int(p.participant.max()) + 1
    perm = np.random.default_rng(FOLD_SEED).permutation(P)
    fold_p = np.zeros(P, int); fold_p[perm] = np.arange(P) % FOLDS
    fold = fold_p[p.participant]
    out = {m: np.full(len(ch), np.nan) for m in ("CL", "SR", "NN", "GBM")}
    pens = []
    for k in range(FOLDS):
        test, train = fold == k, fold != k
        if not test.any():
            continue
        tr, te = subset(p, np.where(train)[0]), subset(p, np.where(test)[0])
        for m in ("CL", "SR"):
            out[m][test] = logp_model(te, m, fit_model(tr, m))
        out["NN"][test], pen = nn_outer(F, ok, ch, p.participant, train, test, seed + k)
        pens.append(pen)
        out["GBM"][test] = gbm_outer(F, ok, ch, train, test)
    out["uniform"] = -np.log2(p.n.astype(float))
    return out, pens


def summarize_boot(vals, reps):
    a = np.array(reps)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"point": float(vals), "percentile": [float(lo), float(hi)]}


if __name__ == "__main__":
    np.seterr(all="ignore")
    out = Path(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    G = groups({"cdm": "data/cdm", "reutskaja": "data/reutskaja", "snack": "data/snack"})
    panels = [(p.name, p) for p in G["snack"][0] + G["reutskaja"][0] + G["cdm_standard"][0] + G["cdm_process"][0]]
    scores, pens = {}, {}
    for i, (name, p) in enumerate(panels):
        scores[name], pens[name] = panel_scores(p, 160000 + 100 * i)
        print(name, {m: float(np.mean(v - scores[name]["uniform"])) for m, v in scores[name].items() if m != "uniform"}, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "trial_scores.npz", **{f"{n}|{m}": v for n, d in scores.items() for m, v in d.items()},
                        **{f"{n}|participant": p.participant for n, p in panels})
    groups_out = {"snack": ["snack"], "reutskaja": ["reutskaja"], "cdm": [n for n, _ in panels if n.startswith("cdm")]}

    def measure(sel):
        """sel maps panel name to trial weights. Mean bits above uniform per panel, averaged over a group's panels."""
        res = {}
        for g, names in groups_out.items():
            per = {m: [] for m in ("CL", "SR", "NN", "GBM")}
            for n in names:
                w = sel[n]
                for m in per:
                    per[m].append(float(w @ (scores[n][m] - scores[n]["uniform"]) / w.sum()))
            res[g] = {m: float(np.mean(v)) for m, v in per.items()}
        return res

    full = {n: np.ones(len(p.choice)) for n, p in panels}
    point = measure(full)
    ceiling = {g: ("NN" if point[g]["NN"] >= point[g]["GBM"] else "GBM") for g in point}
    # Participants are resampled within each source group (cdm_standard, cdm_process, reutskaja, snack, in the
    # order groups() returns them), as in the forecasting bootstrap, because the two Caplin, Dean and Martin
    # treatments hold different people.
    reps = []
    for b in range(B):
        rng = np.random.default_rng(170000 + b)
        sel = {}
        for _, (gpanels, P) in G.items():
            mult = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
            for p in gpanels:
                sel[p.name] = mult[p.participant]
        reps.append(measure(sel))
    s = {}
    for g in point:
        c = ceiling[g]
        for m in ("CL", "SR", "NN", "GBM"):
            s[f"{g}|bits|{m}"] = summarize_boot(point[g][m], [r[g][m] for r in reps])
        for m in ("CL", "SR"):
            s[f"{g}|share_of_ceiling|{m}"] = summarize_boot(100 * point[g][m] / point[g][c], [100 * r[g][m] / r[g][c] for r in reps])
        s[f"{g}|ceiling_model"] = c
    s["penalties_chosen"] = pens
    json.dump(s, open(out / "summary.json", "w"), indent=1, sort_keys=True)
    names = {"snack": "Snacks at a glance", "reutskaja": "Snacks in three seconds", "cdm": "Sums worked out"}
    f = lambda e, d=3: f"{e['point']:.{d}f} [{e['percentile'][0]:.{d}f}, {e['percentile'][1]:.{d}f}]"
    L = ["# A machine learning ceiling for completeness (exploratory)", "",
         "Held out bits per choice above a uniform guess, ten folds of participants. The ceiling is the better of the neural network and the boosted trees.", "",
         "| Dataset | CL | Sample until return | Neural network | Boosted trees | CL, percent of ceiling | Sample until return, percent of ceiling |",
         "|---|---|---|---|---|---|---|"]
    for g in ("snack", "reutskaja", "cdm"):
        L.append(f"| {names[g]} | {f(s[g+'|bits|CL'])} | {f(s[g+'|bits|SR'])} | {f(s[g+'|bits|NN'])} | {f(s[g+'|bits|GBM'])} | "
                 f"{f(s[g+'|share_of_ceiling|CL'], 0)} | {f(s[g+'|share_of_ceiling|SR'], 0)} |")
    L += ["", f"Brackets give 95% percentile intervals from a participant bootstrap of held out trial scores (B = {B}, seeds 170000 + b, no refitting).",
          "A share of the ceiling is an upper bound on completeness (Fudenberg et al. 2022), because the ceiling's own error is at least the irreducible error."]
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
