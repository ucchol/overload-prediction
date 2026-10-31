"""Amendment 8 (VERIFY_BRIEF_OVERLOAD.md): forecasting choice overload. Data, panels, models and measures.

Panels: cdm_standard_3, cdm_standard_7, cdm_process_3, cdm_process_7 (Amendment 7 rounds, dollars) and reutskaja
(ratings -5..5 shifted by 6 to whole levels 1..11). A miss is a choice worth less than the set's largest value.
Models, one sensitivity each, every set size of a panel forecast from fits to the panel's other set sizes:
  CL       softmax(lam v)
  CL_comp  softmax(lam f(n) v), f(n) = ((n - 1) / 7) ** 0.18
  SR       race probabilities of softmax(lam v) (sample until return)
Reuses ../independent_cdm/cdm_core.py (loader of Amendment 7, softmax root finding, SR grid and Brent, race code).
Nothing here prints or stores identifiers or participant, trial or selection counts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(VERIFY, "independent_cdm"))
import cdm_core as cc  # noqa: E402
from cdm_core import srch  # noqa: E402

srch.assert_local(cc, srch, sys.modules[__name__])

REUT_DIR = os.path.realpath(os.path.join(VERIFY, "reutskaja"))
REUT_FILE = "fixations_data.dta"
PANELS = ("cdm_standard_3", "cdm_standard_7", "cdm_process_3", "cdm_process_7", "reutskaja")
GROUPS = ("standard", "process", "reutskaja")  # bootstrap groups, drawn in this order
COMP = 0.18
MODELS = ("CL", "CL_comp", "SR")
FORECASTS = ("CL", "CL_comp", "SR", "near_best", "minus_gap")
ROUND_SIG = 10  # model forecasts are rounded with format .10g before ranking (item 5)
BOOT_SEED = 80000


def f_comp(n):
    return ((np.asarray(n, float) - 1.0) / 7.0) ** COMP


# ============================================================================ data
def read_reutskaja(path: str, allow_real: bool = False) -> dict:
    """Trials of Reutskaja et al.: ratings in screen order, chosen index, fixated positions."""
    if os.path.realpath(os.path.dirname(os.path.abspath(path))) == REUT_DIR and not allow_real:
        raise PermissionError("the real Reutskaja file is analysed only after the coordinator's go-ahead (--real)")
    D = pd.read_stata(path, convert_categoricals=False)
    need = ["subject", "trialid", "numalt", "chosendestring", "Roinum", "fixOrder"] + [f"rating{i}" for i in range(1, 17)]
    missing = [c for c in need if c not in D.columns]
    if missing:
        raise ValueError(f"columns missing from the Stata file: {missing}")
    D = D[need].apply(pd.to_numeric, errors="coerce")
    if D[["subject", "trialid"]].isna().any().any():
        raise ValueError("a row lacks its subject or trial id")
    trial_cols = ["numalt", "chosendestring"] + [f"rating{i}" for i in range(1, 17)]
    g = D.groupby(["subject", "trialid"], sort=True)
    first = g[trial_cols].first()
    if not (g[trial_cols].nunique(dropna=False) <= 1).all().all():
        raise ValueError("a trial field varies within a trial")
    n = first["numalt"].to_numpy(float)
    if not np.all(np.isin(n, [4, 9, 16])):
        raise ValueError("set size other than 4, 9 or 16")
    n = n.astype(int)
    R = first[[f"rating{i}" for i in range(1, 17)]].to_numpy(float)
    ch = first["chosendestring"].to_numpy(float)
    if not np.all((ch == np.floor(ch)) & (ch >= 1) & (ch <= n)):
        raise ValueError("chosen position outside 1 .. set size")
    V = []
    for r, k in zip(R, n):
        v = r[:k]
        if not np.all(np.isfinite(v) & (v == np.floor(v)) & (v >= -5) & (v <= 5)):
            raise ValueError("a rating is not a whole number from -5 to 5")
        V.append(v + 6.0)  # whole levels 1 .. 11
    keys = first.index.to_frame(index=False)
    fix = D.dropna(subset=["Roinum"]).groupby(["subject", "trialid"], sort=True)["Roinum"].apply(lambda s: set(int(x) for x in s))
    fixated = [fix.get((s, t), set()) for s, t in zip(keys["subject"], keys["trialid"])]
    return dict(subject=keys["subject"].to_numpy(), n=n, V=V, chosen=ch.astype(int) - 1, fixated=fixated)


def make_panel(name, n, V, chosen, pidx, group, fixated=None):
    sizes = sorted(set(int(x) for x in n))
    if len(sizes) != 3:
        raise ValueError(f"{name}: a panel needs three set sizes")
    top = np.array([v.max() for v in V])
    v_ch = np.array([v[c] for v, c in zip(V, chosen)])
    miss = (v_ch < top).astype(float)
    near = np.array([float(((v >= t - 1) & (v < t)).sum()) for v, t in zip(V, top)])
    gap = []
    for v, t in zip(V, top):
        below = v[v < t]
        gap.append(-(t - below.max()) if len(below) else -np.inf)
    items = []
    for s in sizes:
        idx = np.flatnonzero(n == s)
        Vg = np.array([V[i] for i in idx])
        items.append((s, idx, Vg, chosen[idx]))
    P = dict(name=name, sizes=sizes, n=np.asarray(n, int), T=len(n), V=V, chosen=chosen, pidx=pidx, group=group,
             miss=miss, near=near, minus_gap=np.array(gap), items=items, size_idx=np.searchsorted(sizes, n), top=top)
    if fixated is not None:  # best option never fixated (positions are 1 based)
        P["best_unseen"] = np.array([not any((j + 1) in fx for j in np.flatnonzero(v == t)) for v, t, fx in zip(V, top, fixated)])
    return P


def build(cdm_path: str, reut_path: str, allow_real: bool = False) -> tuple[list, dict]:
    """The five panels and the participant counts of the three bootstrap groups (kept in memory only)."""
    data = cc.load_cdm(cdm_path, allow_real=allow_real)
    panels, P = [], {}
    for name in ("standard", "process"):
        X = data[name]
        P[name] = X["P"]
        for c in cc.CPLX:
            r = np.flatnonzero(X["cplx"] == c)
            panels.append(make_panel(f"cdm_{name}_{c}", X["n"][r], [X["Vlist"][i] for i in r], X["chosen"][r], X["pidx"][r], name))
    Rt = read_reutskaja(reut_path, allow_real)
    ids = np.unique(Rt["subject"])
    P["reutskaja"] = len(ids)
    panels.append(make_panel("reutskaja", Rt["n"], Rt["V"], Rt["chosen"], np.searchsorted(ids, Rt["subject"]), "reutskaja", Rt["fixated"]))
    return panels, P


# ============================================================================ models
class PanelFits:
    """Weight-free precomputation of a panel: scores of CL and CL_comp, SR's grid tables."""

    def __init__(self, P: dict):
        self.P = P
        self.items = P["items"]
        self.S = {"CL": [V for _, _, V, _ in self.items], "CL_comp": [f_comp(s) * V for s, _, V, _ in self.items]}
        G = type("G", (), {})()
        G.items, G.R = self.items, P["T"]
        self.sr = cc.SRFitGroups(G)
        # one option at the top value of each trial, for SR's probability of choosing the best
        self.top_idx = [np.argmax(V, axis=1) for _, _, V, _ in self.items]
        self.ntop = [(V == V.max(axis=1, keepdims=True)).sum(axis=1) for _, _, V, _ in self.items]

    def fit(self, model, w, init=None):
        if model == "SR":
            lam, info = self.sr.fit(w)
            return lam, info
        return cc.fit_softmax_groups(self.S[model], self.items, w, init=init), {}

    def miss_prob(self, model, lam, k):
        """Probability of a miss for every trial of the k-th set size."""
        s, idx, V, c = self.items[k]
        if model == "SR":
            p_best = self.ntop[k] * np.exp(cc.sr_log_chosen(lam, V, self.top_idx[k]))
        else:
            S = self.S[model][k]
            q = np.exp(cc.log_softmax(lam, S))
            p_best = (q * (V == V.max(axis=1, keepdims=True))).sum(axis=1)
        return np.clip(1.0 - p_best, 0.0, 1.0)


# ============================================================================ measures
def wauc(score, miss, w):
    """Weighted probability that a miss trial outscores a hit trial, ties one half, over trials with w > 0."""
    on = w > 0
    score, miss, w = score[on], miss[on] > 0.5, w[on]
    Wm, Wh = w[miss].sum(), w[~miss].sum()
    if Wm == 0 or Wh == 0:
        return np.nan
    u, inv = np.unique(score, return_inverse=True)
    wm = np.bincount(inv, weights=np.where(miss, w, 0.0), minlength=len(u))
    wh = np.bincount(inv, weights=np.where(miss, 0.0, w), minlength=len(u))
    below = np.cumsum(wh) - wh
    return float((wm * (below + 0.5 * wh)).sum() / (Wm * Wh))


def round_sig(x, sig=ROUND_SIG):
    return np.array([float(format(v, f".{sig}g")) for v in np.asarray(x, float)])


def evaluate_panel(pf: PanelFits, w_t: np.ndarray, init=None) -> dict:
    """Leave one set size out: lambdas, held out miss probabilities, rates, growth and areas under the curve."""
    P = pf.P
    sizes = P["sizes"]
    lam = {m: np.zeros(3) for m in MODELS}
    prob = {m: np.zeros(P["T"]) for m in MODELS}
    dense = -np.inf
    for k in range(3):
        wt = np.where(P["size_idx"] != k, w_t, 0.0)
        idx = pf.items[k][1]
        for m in MODELS:
            lam[m][k], info = pf.fit(m, wt, None if init is None else init[m][k])
            if m == "SR":
                dense = max(dense, info["dense_excess"])
            prob[m][idx] = pf.miss_prob(m, lam[m][k], k)
    x = np.log2(np.array(sizes, float))
    rates = {}
    for key, vals in [("observed", P["miss"])] + [(m, prob[m]) for m in MODELS]:
        rates[key] = np.array([srch.wmean(vals[P["size_idx"] == k], w_t[P["size_idx"] == k]) for k in range(3)])
    growth = {key: float(np.polyfit(x, r, 1)[0]) for key, r in rates.items()}
    fc = {m: round_sig(prob[m]) for m in MODELS}
    fc["near_best"] = P["near"]
    fc["minus_gap"] = P["minus_gap"]
    auc = {}
    for f in FORECASTS:
        num = den = 0.0
        for k in range(3):
            sel = P["size_idx"] == k
            a = wauc(fc[f][sel], P["miss"][sel], w_t[sel])
            if np.isfinite(a):
                W = w_t[sel].sum()
                num += W * a
                den += W
        auc[f] = num / den if den > 0 else np.nan
    return dict(lam=lam, rates=rates, growth=growth, auc=auc, dense_excess=dense)
