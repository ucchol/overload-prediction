"""Exploratory, participant level views of H2 and H3 (after Amendments 8 and 11).

H2. With the held out fits of Amendment 11 (each set size from fits to the others), each participant's mean log2
probability of their choices under CL plus the return cue and CL plus the dwell cue. Reported, the share of
participants for whom dwell predicts better, and for whom return beats CL alone.

H3. Each participant's own growth of missed best options per doubling, observed and as forecast by CL and sample
until return fitted to the other set sizes (Amendment 8 forecasts), for participants seen at every set size.
Reported, the share of participants whose own growth is positive, and the rank correlation across participants
between observed and forecast growth.

Shares are percentages. Participant bootstrap, numpy.random.default_rng(140000 + b).
Usage, python3 -I scripts/individual_level.py OUT [B]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.insert(0, "src")
from overload.bootstrap import summarize  # noqa: E402
from overload.experiments import h2_return as H2  # noqa: E402
from overload.experiments.overload_premonition import groups  # noqa: E402
from overload.overload_forecast import miss_forecasts  # noqa: E402


def h2_person_scores(blocks, P, w):
    """Per participant mean log2 p(chosen), held out by set size, for CL, CL+return and CL+dwell."""
    tot = {m: np.zeros(P) for m in ("CL", "return", "dwell")}
    cnt = np.zeros(P)
    for k, b in enumerate(blocks):
        others = [x for j, x in enumerate(blocks) if j != k]
        wo = [x for j, x in enumerate(w) if j != k]
        for m in tot:
            cue = None if m == "CL" else m
            lam, beta = H2.fit(others, wo, cue)
            Z = b.cue[cue] if cue else np.zeros_like(b.V)
            U = lam * b.V + beta * Z
            lp = (U[np.arange(b.choice.size), b.choice] - logsumexp(U, axis=1)) / np.log(2)
            np.add.at(tot[m], b.person, lp)
        np.add.at(cnt, b.person, 1.0)
    return {m: tot[m] / np.maximum(cnt, 1) for m in tot}, cnt > 0


def slope(x, y):
    xm = x.mean()
    return float(np.sum((x - xm) * (y - y.mean())) / np.sum((x - xm) ** 2))


def h3_person(panel, P, w):
    fc, _ = miss_forecasts(panel, w, ("CL", "SR"))
    y = panel.miss().astype(float)
    sizes = np.unique(panel.n)
    obs, cl, sr = [], [], []
    for p in range(P):
        r = panel.participant == p
        if not all(np.any(r & (panel.n == s)) for s in sizes):
            continue
        x = np.log2(sizes.astype(float))
        rate = lambda a: np.array([a[r & (panel.n == s)].mean() for s in sizes])
        obs.append(slope(x, rate(y))); cl.append(slope(x, rate(fc["CL"]))); sr.append(slope(x, rate(fc["SR"])))
    return np.array(obs), np.array(cl), np.array(sr)


def measures(h2data, G, b):
    rng = None if b < 0 else np.random.default_rng(140000 + b)
    out = {}
    for name in H2.DATASETS:  # snack, then reutskaja
        blocks, P = h2data[name]
        draw = np.arange(P) if rng is None else rng.integers(0, P, size=P)
        mult = np.bincount(draw, minlength=P).astype(float)
        w = [mult[bl.person] for bl in blocks]
        sc, seen = h2_person_scores(blocks, P, w)
        people = np.repeat(np.arange(P), mult.astype(int))
        people = people[seen[people]]
        out[f"H2|{name}|share_dwell_beats_return"] = 100 * float(np.mean(sc["dwell"][people] > sc["return"][people]))
        out[f"H2|{name}|share_return_beats_CL"] = 100 * float(np.mean(sc["return"][people] > sc["CL"][people]))
    for name in ("snack", "reutskaja"):
        (panel,), P = G[name]
        draw = np.arange(P) if rng is None else rng.integers(0, P, size=P)
        mult = np.bincount(draw, minlength=P).astype(float)
        o, c, s = h3_person(panel, P, mult[panel.participant])
        # participants drawn several times enter once per draw
        idx = [p for p in range(P) if all(np.any((panel.participant == p) & (panel.n == z)) for z in np.unique(panel.n))]
        rep = np.repeat(np.arange(len(idx)), mult[idx].astype(int))
        out[f"H3|{name}|share_own_growth_positive"] = 100 * float(np.mean(o[rep] > 0))
        out[f"H3|{name}|median_own_growth"] = float(np.median(o[rep]))
        out[f"H3|{name}|rank_corr_observed_SR"] = float(spearmanr(o[rep], s[rep])[0])
        out[f"H3|{name}|rank_corr_observed_CL"] = float(spearmanr(o[rep], c[rep])[0])
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    out = Path(sys.argv[1]); B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    h2data = {"snack": H2.load_snack_gaze(Path("data/snack"), Path("data/snack_gaze")),
              "reutskaja": H2.load_reutskaja_gaze(Path("data/reutskaja"))}
    G = groups({"reutskaja": "data/reutskaja", "snack": "data/snack"})
    point = measures(h2data, G, -1)
    reps = []
    for b in range(B):
        reps.append(measures(h2data, G, b))
    s = {k: summarize(v, [r[k] for r in reps]) for k, v in point.items()}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(out / "summary.json", "w"), indent=1, sort_keys=True)
    f = lambda e, d=2: f"{e['point']:.{d}f} [{e['percentile'][0]:.{d}f}, {e['percentile'][1]:.{d}f}]"
    L = ["# Participant level views of H2 and H3 (exploratory)", "", "| Measure | Snacks at a glance | Snacks in three seconds |",
         "|---|---|---|",
         f"| H2, participants for whom dwell predicts better than return (%) | {f(s['H2|snack|share_dwell_beats_return'], 0)} | {f(s['H2|reutskaja|share_dwell_beats_return'], 0)} |",
         f"| H2, participants for whom return beats CL alone (%) | {f(s['H2|snack|share_return_beats_CL'], 0)} | {f(s['H2|reutskaja|share_return_beats_CL'], 0)} |",
         f"| H3, participants whose own misses grow with set size (%) | {f(s['H3|snack|share_own_growth_positive'], 0)} | {f(s['H3|reutskaja|share_own_growth_positive'], 0)} |",
         f"| H3, median own growth per doubling | {f(s['H3|snack|median_own_growth'], 3)} | {f(s['H3|reutskaja|median_own_growth'], 3)} |",
         f"| H3, rank correlation of own growth with sample until return's forecast | {f(s['H3|snack|rank_corr_observed_SR'])} | {f(s['H3|reutskaja|rank_corr_observed_SR'])} |",
         f"| H3, rank correlation of own growth with CL's forecast | {f(s['H3|snack|rank_corr_observed_CL'])} | {f(s['H3|reutskaja|rank_corr_observed_CL'])} |",
         "", f"Brackets give 95% percentile intervals from the participant bootstrap (B = {B}, seeds 140000 + b)."]
    (out / "table.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
