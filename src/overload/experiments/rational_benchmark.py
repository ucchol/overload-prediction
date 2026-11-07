"""Overload against a rational search benchmark, ANALYSIS_PLAN.md Amendment 13.

Part V, welfare. Within person slope of the chosen value (and of the best value, the mean option value and the chosen
value minus the mean option value) on log2 n.
Part R, the growth of misses forecast by a satisficer with a fixed threshold (SAT, random search order, exact) and by
one whose threshold falls with each option inspected (FT, Monte Carlo over fixed orders), each set size forecast from
fits to the panel's other set sizes. Screen order versions for Caplin, Dean and Martin.
Part S, what ends a search in the Caplin, Dean and Martin selections, a linear probability regression with participant
fixed effects of stopping on value, log time and log2 n.
Part D, monotonicity of the default share over nested menus in the Aguiar data.

Usage
    python -I -m overload.experiments.rational_benchmark --data data --out results/rational_benchmark [--B 1000]
    python -I -m overload.experiments.rational_benchmark --data data --simulate {sat,ft,fixed_stop} --out ...
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bootstrap import summarize, two_sided_p, holm
from ..cdm import load_cdm
from ..engine import provenance, write_json
from ..overload_forecast import cdm_panels, load_reutskaja, miss_forecasts, slope_on_log2
from ..snack import load_snack
from . import h1_within as aguiar

DELTAS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
N_ORDERS = 500
ORDER_SEED = 131_000_000
BOOT_SEED = 131_000
AGUIAR_SEED = 132_000
EM_TOL = 1e-10
EM_MAX = 5000
BENCH_SEED = 133_000


# ----------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Panel:
    name: str
    group: str  # bootstrap group, participants are drawn within it
    values: np.ndarray  # trials by max n, NaN past n
    n: np.ndarray
    choice: np.ndarray
    participant: np.ndarray  # index into the group's sorted participants
    levels: np.ndarray  # whole levels of the value scale, then one level above it (never stop early)
    order_offset: int  # global index of the panel's first trial, for the Monte Carlo orders
    cl_panel: object | None = None  # the Amendment 8 Panel, for the CL and SR log scores

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(int(x) for x in np.unique(self.n))

    def miss(self) -> np.ndarray:
        top = np.nanmax(self.values, axis=1)
        return self.values[np.arange(self.n.size), self.choice] < top


def load_panels(root: Path) -> tuple[list[Panel], dict[str, int]]:
    """Snack, Reutskaja and the four Caplin, Dean and Martin panels, and the participants of each group."""
    panels, groups, offset = [], {}, 0
    sd = load_snack(root / "snack")
    m = max(b.n for b in sd.blocks)
    V = np.vstack([np.pad(b.values, ((0, 0), (0, m - b.n)), constant_values=np.nan) for b in sd.blocks])
    n = np.concatenate([np.full(b.trials, b.n) for b in sd.blocks])
    ch = np.concatenate([b.choice for b in sd.blocks])
    pid = np.concatenate([b.participant for b in sd.blocks])
    panels.append(Panel("snack", "snack", V, n, ch, pid, np.arange(-3.0, 5.0), offset))
    groups["snack"] = len(sd.participants)
    offset += n.size
    rp, rid = load_reutskaja(root / "reutskaja")
    panels.append(Panel("reutskaja", "reutskaja", rp.values, rp.n, rp.choice, rp.participant,
                        np.arange(-5.0, 7.0), offset, rp))
    groups["reutskaja"] = len(rid)
    offset += rp.n.size
    data = load_cdm(root / "cdm")
    for key, (pl, people) in cdm_panels(data).items():
        groups[key] = people
        for q in pl:
            panels.append(Panel(q.name, key, q.values, q.n, q.choice, q.participant, np.arange(1.0, 37.0), offset, q))
            offset += q.n.size
    return panels, groups


# ----------------------------------------------------------------------------- part V
def within_slope(y: np.ndarray, x: np.ndarray, pid: np.ndarray, wp: np.ndarray) -> float:
    """Least squares slope of y on x after removing each participant's mean, participants weighted by wp."""
    P = wp.size
    cnt = np.bincount(pid, minlength=P).astype(float)
    safe = np.maximum(cnt, 1.0)
    xm = np.bincount(pid, x, P) / safe
    ym = np.bincount(pid, y, P) / safe
    xd, yd = x - xm[pid], y - ym[pid]
    w = wp[pid]
    return float(np.sum(w * xd * yd) / np.sum(w * xd * xd))


def welfare_series(p: Panel) -> dict[str, np.ndarray]:
    r = np.arange(p.n.size)
    chosen = p.values[r, p.choice]
    mean = np.nanmean(p.values, axis=1)
    return {"chosen": chosen, "best": np.nanmax(p.values, axis=1), "mean_option": mean, "chosen_minus_mean": chosen - mean}


# ----------------------------------------------------------------------------- part R, choice probabilities given a threshold
def sat_tables(p: Panel) -> tuple[np.ndarray, np.ndarray]:
    """Exact random order SAT. Probability of the chosen option and of a best option, trials by levels."""
    V = p.values
    top = np.nanmax(V, axis=1)
    r = np.arange(p.n.size)
    vc = V[r, p.choice]
    n_top = np.sum(V == top[:, None], axis=1).astype(float)
    cnt = np.sum(np.where(np.isfinite(V)[:, :, None], V[:, :, None] >= p.levels[None, None, :], False), axis=1).astype(float)
    some = cnt > 0
    a_ch = np.where(some, (vc[:, None] >= p.levels[None, :]) / np.maximum(cnt, 1.0), ((vc == top) / n_top)[:, None])
    a_best = np.where(some, n_top[:, None] / np.maximum(cnt, 1.0), 1.0)
    return a_ch, a_best


def _orders(p: Panel, i: int, screen: bool) -> np.ndarray:
    k = int(p.n[i])
    if screen:
        return np.arange(k)[None, :]
    rng = np.random.default_rng(ORDER_SEED + p.order_offset + i)
    return np.argsort(rng.random((N_ORDERS, k)), axis=1)


def ft_tables(p: Panel, delta: float, screen: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """FT (SAT at delta 0) over fixed orders. Threshold t - delta (k - 1) at the k-th inspected option."""
    T = p.n.size
    L = p.levels
    a_ch, a_best = np.zeros((T, L.size)), np.zeros((T, L.size))
    for i in range(T):
        k = int(p.n[i])
        v = p.values[i, :k]
        top = v.max()
        O = _orders(p, i, screen)
        U = v[O]  # orders by k
        M = np.maximum.accumulate(U, axis=1)
        # position (within the order) of the first occurrence of the running maximum
        prev = np.concatenate([np.full((O.shape[0], 1), -np.inf), M[:, :-1]], axis=1)
        A = np.maximum.accumulate(np.where(U > prev, np.arange(k)[None, :], 0), axis=1)
        H = np.maximum.accumulate(M + delta * scale_unit(p) * np.arange(k)[None, :], axis=1)
        reach = H[:, :, None] >= L[None, None, :]  # orders, k, levels
        any_reach = reach.any(axis=1)
        kstar = np.where(any_reach, reach.argmax(axis=1), k - 1)  # orders by levels
        pos = np.take_along_axis(A, kstar, axis=1)
        chosen = np.take_along_axis(O, pos, axis=1)  # option index, orders by levels
        a_ch[i] = (chosen == p.choice[i]).mean(axis=0)
        a_best[i] = (v[chosen] == top).mean(axis=0)
    return a_ch, a_best


def scale_unit(p: Panel) -> float:
    return 1.0  # every scale here moves in whole units (rating points or dollars)


# ----------------------------------------------------------------------------- EM
def _compress(a: np.ndarray, inv: np.ndarray, w: np.ndarray):
    """Merge trials with identical rows, adding their weights (the likelihood is unchanged)."""
    B = np.column_stack([a, inv])
    uniq, idx = np.unique(B, axis=0, return_inverse=True)
    return uniq, np.bincount(idx.ravel(), w, uniq.shape[0])


def fit_em(a_ch: np.ndarray, n: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Mixture over thresholds with a uniform lapse, maximum likelihood by EM accelerated by SQUAREM.

    With c_t = pi_t (1 - e) and c_lapse = e the likelihood of a trial is linear in c, so the log likelihood is concave
    on the simplex and EM climbs to its maximum. Start from uniform pi and e = 0.1, stop when the log likelihood rises
    by less than EM_TOL or after EM_MAX accelerated steps. e is kept within [1e-4, 0.9]."""
    keep = w > 0
    Bm, ww = _compress(a_ch[keep], 1.0 / n[keep], w[keep])
    W = ww.sum()
    K = a_ch.shape[1]
    c = np.r_[np.full(K, 0.9 / K), 0.1]

    def ll_of(c):
        return float(ww @ np.log(np.maximum(Bm @ c, 1e-300)))

    def em(c):
        L = np.maximum(Bm @ c, 1e-300)
        c = c * ((ww / L) @ Bm) / W
        e = min(max(c[-1], 1e-4), 0.9)
        c[:-1] *= (1.0 - e) / c[:-1].sum()
        c[-1] = e
        return c

    ll_old = ll_of(c)
    for _ in range(EM_MAX):
        c1 = em(c)
        c2 = em(c1)
        r, v = c1 - c, c2 - 2 * c1 + c
        nv = np.linalg.norm(v)
        if nv > 0:
            alpha = -np.linalg.norm(r) / nv
            cn = np.maximum(c - 2 * alpha * r + alpha * alpha * v, 0.0)
            cn = em(cn / cn.sum()) if cn.sum() > 0 else c2
            if not (ll_of(cn) >= ll_of(c2)):
                cn = c2
        else:
            cn = c2
        ll = ll_of(cn)
        c = cn
        if ll - ll_old < EM_TOL:
            break
        ll_old = ll
    e = float(c[-1])
    return c[:-1] / (1.0 - e), e, ll_of(c)


def loglik(a_ch, n, w, pi, e) -> float:
    L = (1.0 - e) * (a_ch @ pi) + e / n
    keep = w > 0
    return float(w[keep] @ np.log(L[keep]))


def miss_prob(a_best, n_top_share, pi, e) -> np.ndarray:
    return 1.0 - ((1.0 - e) * (a_best @ pi) + e * n_top_share)


# ----------------------------------------------------------------------------- part R per panel
@dataclass
class Tables:
    sat: tuple[np.ndarray, np.ndarray]
    ft: dict[float, tuple[np.ndarray, np.ndarray]]
    sat_screen: tuple[np.ndarray, np.ndarray] | None
    ft_screen: dict[float, tuple[np.ndarray, np.ndarray]] | None
    top_share: np.ndarray


def tables(p: Panel) -> Tables:
    top = np.nanmax(p.values, axis=1)
    top_share = np.sum(p.values == top[:, None], axis=1) / p.n
    cdm = p.name.startswith("cdm")
    ft = {d: ft_tables(p, d) for d in DELTAS}
    fts = {d: ft_tables(p, d, screen=True) for d in DELTAS} if cdm else None
    return Tables(sat_tables(p), ft, fts[0.0] if cdm else None, fts, top_share)


def forecast(p: Panel, tb: Tables, w: np.ndarray, version: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Held out probability of a miss and log2 probability of the choice, each size from fits to the others."""
    pm, lg, info = np.zeros(p.n.size), np.zeros(p.n.size), []
    for s in p.sizes:
        te = p.n == s
        tr = ~te
        wt = np.where(tr, w, 0.0)
        if version in ("SAT", "SAT_screen"):
            a_ch, a_best = tb.sat if version == "SAT" else tb.sat_screen
            pi, e, _ = fit_em(a_ch, p.n, wt)
            d = 0.0
        else:
            grid = tb.ft if version == "FT" else tb.ft_screen
            fits = {d: fit_em(grid[d][0], p.n, wt) for d in DELTAS}
            d = max(DELTAS, key=lambda x: fits[x][2])  # ties keep the smaller delta
            pi, e, _ = fits[d]
            a_ch, a_best = grid[d]
        pm[te] = miss_prob(a_best[te], tb.top_share[te], pi, e)
        q = (1.0 - e) * (a_ch[te] @ pi) + e / p.n[te]
        lg[te] = np.log2(np.maximum(q, 1e-300))
        info.append({"size": s, "delta": d, "lapse": e})
    return pm, lg, info


def rates(p: Panel, x: np.ndarray, w: np.ndarray) -> list[float]:
    return [float(w[p.n == s] @ x[p.n == s] / w[p.n == s].sum()) for s in p.sizes]


def versions(p: Panel) -> tuple[str, ...]:
    return ("SAT", "FT", "SAT_screen", "FT_screen") if p.name.startswith("cdm") else ("SAT", "FT")


def panel_r(p: Panel, tb: Tables, w: np.ndarray, extra: bool = False) -> dict:
    y = p.miss().astype(float)
    out = {"sizes": list(p.sizes), "rate": {"observed": rates(p, y, w)}, "growth": {}, "fits": {}, "loglik": {}}
    for v in versions(p):
        pm, lg, info = forecast(p, tb, w, v)
        out["rate"][v] = rates(p, pm, w)
        out["fits"][v] = info
        out["loglik"][v] = float(w @ lg / w.sum())
    for k, r in out["rate"].items():
        out["growth"][k] = slope_on_log2(p.sizes, r)
    if extra and p.cl_panel is not None:  # CL and SR log scores, original sample only
        ll: dict = {}
        miss_forecasts(p.cl_panel, w, models=("CL", "SR"), loglik=ll)
        for m in ("CL", "SR"):
            out["loglik"][m] = float(w @ ll[m] / w.sum())
    return out


# ----------------------------------------------------------------------------- part S
def stopping(data, experiment: int, wp: np.ndarray, stand: dict | None = None) -> tuple[dict, dict]:
    """Linear probability of a selection being the last, participant fixed effects, value, log time and log2 n."""
    r = data.sel_round
    keep = (data.experiment[r] == experiment) & (data.sel_better > 0)
    y = data.sel_stop[keep].astype(float)
    X = np.column_stack([data.sel_value[keep], np.log(data.sel_time[keep]), np.log2(data.n[r[keep]].astype(float))])
    if stand is None:
        stand = {"mean": X.mean(axis=0).tolist(), "sd": X.std(axis=0).tolist()}
    X = (X - np.array(stand["mean"])) / np.array(stand["sd"])
    people = np.unique(data.participant[data.experiment == experiment])
    local = np.full(len(data.participants), -1)
    local[people] = np.arange(people.size)
    pid = local[data.participant[r[keep]]]
    P = people.size
    cnt = np.maximum(np.bincount(pid, minlength=P), 1).astype(float)
    Xd = X - np.column_stack([np.bincount(pid, X[:, j], P) / cnt for j in range(3)])[pid]
    yd = y - (np.bincount(pid, y, P) / cnt)[pid]
    w = wp[pid]
    beta = np.linalg.solve((Xd * w[:, None]).T @ Xd, (Xd * w[:, None]).T @ yd)
    return {"value": float(beta[0]), "log_time": float(beta[1]), "log2_n": float(beta[2])}, stand


def stopping_benchmark(data, panels, stand: dict, sims: int = 100) -> dict:
    """Coefficients of the Part S regression on selections simulated from SAT, averaged over sims datasets.

    SAT is fitted to every set size of each Caplin, Dean and Martin panel's choices. In a simulated round t is drawn
    from pi, options are inspected in a uniformly random order, every new best found is selected (the first option
    always is, since the default is worth nothing), and search stops once the best found reaches t or the list ends.
    Inspection times are gamma with shape 2, their mean per panel set so that the mean time of the final selection
    equals people's. Lapses are not simulated. Dataset s uses numpy.random.default_rng(BENCH_SEED + s)."""
    fits = {p.name: fit_em(sat_tables(p)[0], p.n, np.ones(p.n.size))[0] for p in panels if p.name.startswith("cdm")}
    names = {0: "cdm_standard", 1: "cdm_process"}
    key = [f"{names[int(e)]}_{int(c)}" for e, c in zip(data.experiment, data.complexity, strict=True)]
    levels = np.arange(1.0, 37.0)
    # mean number inspected under the benchmark, per panel, for the time scale
    rng = np.random.default_rng(BENCH_SEED - 1)
    insp = {k: [] for k in fits}
    for r in range(data.rounds):
        for _ in range(20):
            _, _, k = _bench_round(data.values[r, : data.n[r]], levels[rng.choice(levels.size, p=fits[key[r]])], rng)
            insp[key[r]].append(k)
    scale = {}
    for k in fits:
        rows = np.array(key) == k
        scale[k] = float(np.mean(data.final_time[rows]) / np.mean(insp[k]))
    coefs = {g: [] for g in names.values()}
    for s in range(sims):
        rng = np.random.default_rng(BENCH_SEED + s)
        sr, sv, st, sb, ss = [], [], [], [], []
        for r in range(data.rounds):
            v = data.values[r, : data.n[r]]
            t = levels[rng.choice(levels.size, p=fits[key[r]])]
            sels, order, _ = _bench_round(v, t, rng)
            times = np.cumsum(rng.gamma(2.0, scale[key[r]] / 2.0, size=v.size))
            for q, (j, pos) in enumerate(sels):
                sr.append(r); sv.append(v[j]); st.append(max(times[pos], 1.0))
                sb.append(int((v > v[j]).sum())); ss.append(q == len(sels) - 1)
        sim = replace(data, sel_round=np.array(sr), sel_value=np.array(sv, float), sel_time=np.array(st),
                      sel_better=np.array(sb), sel_stop=np.array(ss))
        for e, g in names.items():
            P = len(np.unique(data.participant[data.experiment == e]))
            coefs[g].append(stopping(sim, e, np.ones(P), stand[g])[0])
    return {g: {k: {"mean": float(np.mean([c[k] for c in cs])), "mc_sd": float(np.std([c[k] for c in cs]) / np.sqrt(sims))}
                for k in cs[0]} for g, cs in coefs.items()} | {"time_scale": scale}


def _bench_round(v, t, rng):
    order = rng.permutation(v.size)
    best, sels = -np.inf, []
    for k, j in enumerate(order):
        if v[j] > best:
            best = v[j]
            sels.append((j, k))
        if best >= t:
            return sels, order, k + 1
    return sels, order, v.size


# ----------------------------------------------------------------------------- part D
def nested_pairs() -> list[tuple[int, int]]:
    menus = list(range(2, len(aguiar.MENUS) + 1))
    sets = {m: set(aguiar.MENUS[m - 1]) for m in menus}
    return [(a, b) for a in menus for b in menus if sets[a] < sets[b]]


def default_measures(counts: dict[str, dict[int, np.ndarray]]) -> dict:
    out = {}
    pairs = nested_pairs()
    for c in aguiar.COSTS:
        share = {m: float(k[0] / k.sum()) for m, k in counts[c].items()}
        diff = [share[b] - share[a] for a, b in pairs]
        sizes = sorted({aguiar.size(m) for m in counts[c]})
        x = np.log2([aguiar.size(m) for m in counts[c]])
        y = np.array([share[m] for m in counts[c]])
        wts = np.array([k.sum() for k in counts[c].values()])
        out[c] = {"diff": diff, "growth": aguiar.slope(x, y, wts),
                  "share_by_size": [float(np.average([share[m] for m in counts[c] if aguiar.size(m) == s],
                                                     weights=[counts[c][m].sum() for m in counts[c] if aguiar.size(m) == s]))
                                    for s in sizes]}
    return out


# ----------------------------------------------------------------------------- run
def draw(groups: dict[str, int], b: int) -> dict[str, np.ndarray]:
    """Participant draw counts per group, in the order snack, reutskaja, cdm_standard, cdm_process."""
    rng = np.random.default_rng(BOOT_SEED + b)
    out = {}
    for g in ("snack", "reutskaja", "cdm_standard", "cdm_process"):
        P = groups[g]
        out[g] = np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
    return out


def replicate(panels, tabs, data, groups, wp: dict[str, np.ndarray], stand, extra=False, bench=None) -> dict:
    rec = {"V": {}, "R": {}, "S": {}}
    for p in panels:
        w = wp[p.group][p.participant]
        x = np.log2(p.n.astype(float))
        rec["V"][p.name] = {k: within_slope(y, x, p.participant, wp[p.group]) for k, y in welfare_series(p).items()}
        rec["R"][p.name] = panel_r(p, tabs[p.name], w, extra)
    for e, g in ((0, "cdm_standard"), (1, "cdm_process")):
        rec["S"][g], _ = stopping(data, e, wp[g], stand[g])
        for k in ("value", "log_time", "log2_n"):
            rec["S"][g][f"{k}_minus_benchmark"] = rec["S"][g][k] - bench[g][k]["mean"]
    # dataset means
    cdm = [rec["R"][q]["growth"] for q in rec["R"] if q.startswith("cdm")]
    rec["R"]["cdm_mean"] = {"growth": {k: float(np.mean([g[k] for g in cdm])) for k in cdm[0]}}
    for d in ("cdm_mean", "reutskaja", "snack"):
        g = rec["R"][d]["growth"]
        rec["R"][d]["R1"] = g["observed"] - g["SAT"]
        rec["R"][d]["error_FT_minus_SAT"] = abs(g["observed"] - g["FT"]) - abs(g["observed"] - g["SAT"])
        if "SAT_screen" in g:
            rec["R"][d]["observed_minus_SAT_screen"] = g["observed"] - g["SAT_screen"]
            rec["R"][d]["observed_minus_FT_screen"] = g["observed"] - g["FT_screen"]
    return rec


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f"{key}/{i}"] = float(x)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--simulate", choices=("sat", "ft", "fixed_stop", "ft_stop"))
    args = ap.parse_args(argv)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    panels, groups = load_panels(root)
    data = load_cdm(root / "cdm")
    counts = aguiar.load(root / "aguiar")
    if args.simulate:
        from . import rational_benchmark_sim as sim
        panels, data = sim.simulate(panels, data, args.simulate)
    panels_bench, data_bench = panels, data
    tabs = {p.name: tables(p) for p in panels}
    ones = {g: np.ones(P) for g, P in groups.items()}
    stand = {g: stopping(data, e, ones[g])[1] for e, g in ((0, "cdm_standard"), (1, "cdm_process"))}
    bench = stopping_benchmark(data_bench if args.simulate else data, panels_bench if args.simulate else panels, stand)
    point = replicate(panels, tabs, data, groups, ones, stand, extra=not args.simulate, bench=bench)
    point["D"] = default_measures(counts)
    flat_point = flatten({k: point[k] for k in ("V", "R", "S")})
    boots = {k: [] for k in flat_point}
    d_boot = {c: [] for c in aguiar.COSTS}
    d_growth = {c: [] for c in aguiar.COSTS}
    for b in range(args.B):
        rec = flatten(replicate(panels, tabs, data, groups, draw(groups, b), stand, bench=bench))
        for k in boots:
            boots[k].append(rec.get(k, np.nan))
        dm = default_measures(aguiar.resample(counts, b - aguiar.SEED_OFFSET + AGUIAR_SEED))
        for c in aguiar.COSTS:
            d_boot[c].append(dm[c]["diff"])
            d_growth[c].append(dm[c]["growth"])
        if b % 50 == 0:
            print(f"replicate {b} at {time.time() - t0:.0f} s", flush=True)
    summary = {k: summarize(flat_point[k], boots[k]) for k in flat_point} if args.B else {}
    D = {}
    pairs = nested_pairs()
    for c in aguiar.COSTS:
        if not args.B:
            break
        arr = np.array(d_boot[c])
        diffs = point["D"][c]["diff"]
        rows = [summarize(diffs[j], arr[:, j]) for j in range(len(pairs))]
        adj = holm({str(j): rows[j]["p"] for j in range(len(pairs))})
        viol = [j for j in range(len(pairs)) if diffs[j] > 0 and rows[j]["clear"] and adj[str(j)] < 0.05]
        D[c] = {"pairs": len(pairs), "violations_clear_holm": len(viol),
                "positive_differences": int(np.sum(np.array(diffs) > 0)),
                "largest_difference": float(np.max(diffs)), "growth": summarize(point["D"][c]["growth"], d_growth[c]),
                "share_by_size": point["D"][c]["share_by_size"],
                "violating_pairs": [[aguiar.options(pairs[j][0]), aguiar.options(pairs[j][1]), diffs[j]] for j in viol]}
    res = {"S_benchmark": bench, "point": point, "summary": summary, "D1": D, "B": args.B, "simulate": args.simulate,
           "seconds": time.time() - t0, "provenance": provenance()}
    write_json(out / "summary.json", res)
    print(json.dumps({k: v for k, v in summary.items() if k.split("/")[-1] in ("R1",) or k.startswith("S/")}, indent=1, default=str))


if __name__ == "__main__":
    main()
