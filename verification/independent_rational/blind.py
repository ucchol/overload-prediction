"""Blind reimplementation of Amendment 13, written from the amendment text alone by a separate agent on 7 Nov 2026,
without reading overload.experiments.rational_benchmark. Original sample only. Run with python3 -I from anywhere."""
"""Blind reimplementation of Amendment 13 on the original sample (weights 1, no bootstrap)."""
import sys, json, itertools
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2] / 'src'))
from pathlib import Path
import numpy as np
from overload.snack import load_snack
from overload.overload_forecast import load_reutskaja, cdm_panels, slope_on_log2
from overload.cdm import load_cdm
from overload.experiments.h1_within import load as load_aguiar, MENUS, options, size, COSTS

D = str(_P(__file__).resolve().parents[2] / 'data') + '/'
OUT = str(_P(__file__).resolve().parent / 'blind.json')
DS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
NORD = 500
TOL = 1e-9

# ----------------------------------------------------------------------------- data
snack = load_snack(D + 'snack')
blocks = sorted(snack.blocks, key=lambda b: b.n)
assert [b.n for b in blocks] == [9, 16, 25, 36]
sv = np.vstack([np.pad(b.values, ((0, 0), (0, 36 - b.n)), constant_values=np.nan) for b in blocks])
panels = [dict(name='snack', values=sv, n=np.concatenate([np.full(b.trials, b.n) for b in blocks]),
               choice=np.concatenate([b.choice for b in blocks]),
               person=np.concatenate([b.participant for b in blocks]),
               levels=np.arange(-3.0, 5.0))]
rp, _ = load_reutskaja(D + 'reutskaja')
panels.append(dict(name='reutskaja', values=rp.values, n=rp.n, choice=rp.choice, person=rp.participant,
                   levels=np.arange(-5.0, 7.0)))
cdm = load_cdm(D + 'cdm')
cps = cdm_panels(cdm)
for key in ('cdm_standard', 'cdm_process'):
    for p in cps[key][0]:
        panels.append(dict(name=p.name, values=p.values, n=p.n, choice=p.choice, person=p.participant,
                           levels=np.arange(1.0, 37.0)))
SUMS = [p['name'] for p in panels[2:]]


def within_slope(x, y, g):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xd = x.copy(); yd = y.copy()
    for u in np.unique(g):
        m = g == u
        xd[m] -= x[m].mean(); yd[m] -= y[m].mean()
    return float((xd * yd).sum() / (xd * xd).sum())


# ----------------------------------------------------------------------------- Part V
partV = {}
for p in panels:
    v = p['values']; T = v.shape[0]
    x = np.log2(p['n'].astype(float))
    chosen = v[np.arange(T), p['choice']]
    best = np.nanmax(v, axis=1); mean = np.nanmean(v, axis=1)
    partV[p['name']] = {k: within_slope(x, y, p['person']) for k, y in
                        (('chosen', chosen), ('best', best), ('mean', mean), ('chosen_minus_mean', chosen - mean))}


# ----------------------------------------------------------------------------- component probabilities
def sat_random(v, c, L):
    """Exact SAT probabilities under random order: chosen and best, trials by levels."""
    T = v.shape[0]
    top = np.nanmax(v, axis=1)
    vc = v[np.arange(T), c]
    ge = (v[:, :, None] >= L[None, None, :]).sum(axis=1)  # T x K, NaN compares False
    nbest = (v == top[:, None]).sum(axis=1)
    reach = ge > 0
    pc = np.where(reach, (vc[:, None] >= L[None, :]) / np.maximum(ge, 1),
                  (vc == top)[:, None] / nbest[:, None])
    pb = np.where(reach, nbest[:, None] / np.maximum(ge, 1), 1.0)
    return pc, pb


def ft_orders(vrow, c, L, orders, d, end='uniform'):
    """FT probability of the chosen option and of the best set, per level, averaged over the given orders.
    Orders: O x n (indices). Threshold at the k-th inspected option (k from 0) is t - d k. Stop at the first k
    where the best value seen so far reaches it, choose the best seen (first found among ties). If never, the
    best of the set, ties uniform."""
    O, n = orders.shape
    seq = vrow[orders]  # O x n
    M = np.maximum.accumulate(seq, axis=1)
    # record holder: first-found index of the running max
    newbest = np.ones_like(seq, bool)
    newbest[:, 1:] = seq[:, 1:] > M[:, :-1]
    pos = np.where(newbest, np.arange(n)[None, :], 0)
    pos = np.maximum.accumulate(pos, axis=1)
    R = np.take_along_axis(orders, pos, axis=1)  # O x n original index of best seen
    S = M + d * np.arange(n)[None, :] + TOL  # nondecreasing in k
    K = L.size
    big = 1e6
    flatS = (S + big * np.arange(O)[:, None]).ravel()
    q = (L[None, :] + big * np.arange(O)[:, None]).ravel()
    kstar = np.searchsorted(flatS, q, side='left').reshape(O, K) - n * np.arange(O)[:, None]  # count S_k < t
    top = vrow.max()
    isbest = vrow == top
    nb = isbest.sum()
    stop = kstar < n
    pick = np.take_along_axis(R, np.minimum(kstar, n - 1), axis=1)  # O x K
    if end == 'first':  # variant, not the text: at the list's end keep the first found best (recall)
        endpick = (R[:, n - 1] == c)[:, None]
    else:
        endpick = (vrow[c] == top) / nb
    pc = np.where(stop, pick == c, endpick).mean(axis=0)
    pb = np.where(stop, isbest[pick], 1.0).mean(axis=0)
    return pc, pb


def screen_rows(v, c, L, d, end='uniform'):
    T, W = v.shape
    pcs = np.zeros((T, L.size)); pbs = np.zeros((T, L.size))
    for i in range(T):
        n = int(np.isfinite(v[i]).sum())
        a, b = ft_orders(v[i, :n], c[i], L, np.arange(n)[None, :], d, end)
        pcs[i] = a; pbs[i] = b
    return pcs, pbs


# Monte Carlo orders, trial index over panels in order
comp = {}  # (panel, model, d) -> (pc, pb)
tidx = 0
for p in panels:
    v, c, L = p['values'], p['choice'], p['levels']
    T = v.shape[0]
    comp[(p['name'], 'SAT', 0.0)] = sat_random(v, c, L)
    ft = {d: (np.zeros((T, L.size)), np.zeros((T, L.size))) for d in DS}
    ftf = {d: np.zeros((T, L.size)) for d in DS}
    for i in range(T):
        n = int(p['n'][i])
        rng = np.random.default_rng(131000000 + tidx)
        tidx += 1
        orders = np.argsort(rng.random((NORD, n)), axis=1)
        for d in DS:
            a, b = ft_orders(v[i, :n], c[i], L, orders, d)
            ft[d][0][i] = a; ft[d][1][i] = b
            ftf[d][i] = ft_orders(v[i, :n], c[i], L, orders, d, 'first')[0]
    for d in DS:
        comp[(p['name'], 'FT', d)] = ft[d]
        comp[(p['name'], 'FT_firstbest', d)] = (ftf[d], ft[d][1])
    if p['name'] in SUMS:
        for d in DS:
            comp[(p['name'], 'FTscreen', d)] = screen_rows(v, c, L, d)
        comp[(p['name'], 'SATscreen', 0.0)] = comp[(p['name'], 'FTscreen', 0.0)]
        for d in DS:
            comp[(p['name'], 'FTscreen_firstbest', d)] = screen_rows(v, c, L, d, 'first')
        comp[(p['name'], 'SATscreen_firstbest', 0.0)] = comp[(p['name'], 'FTscreen_firstbest', 0.0)]
    print('components done', p['name'], flush=True)


# ----------------------------------------------------------------------------- EM with SQUAREM
def em_fit(A, w, max_steps=5000, tol=1e-10):
    """A: trials x (K+1) likelihood matrix, last column the lapse. Maximize sum w log(A c) over the simplex
    with c_last in [1e-4, 0.9]."""
    # merge identical rows
    U, inv = np.unique(A, axis=0, return_inverse=True)
    W = np.bincount(inv.ravel(), weights=w, minlength=U.shape[0])
    A, w = U, W
    K = A.shape[1] - 1

    def ll(c):
        return float((w * np.log(np.maximum(A @ c, 1e-300))).sum())

    def F(c):
        m = A @ c
        r = (w / np.maximum(m, 1e-300)) @ A * c  # expected counts
        tot = r.sum()
        e = min(max(r[-1] / tot, 1e-4), 0.9)
        rest = r[:-1]
        out = np.empty_like(c)
        out[:-1] = (1 - e) * rest / rest.sum()
        out[-1] = e
        return out

    def feasible(c):
        return np.all(c[:-1] >= 0) and 1e-4 <= c[-1] <= 0.9

    c = np.r_[np.full(K, 0.9 / K), 0.1]
    L0 = ll(c)
    for _ in range(max_steps):
        c1 = F(c); c2 = F(c1)
        r = c1 - c; vv = c2 - 2 * c1 + c
        nr, nv = np.linalg.norm(r), np.linalg.norm(vv)
        if nv == 0 or nr == 0:
            cn = c2
        else:
            alpha = min(-nr / nv, -1.0)
            while True:
                cn = c - 2 * alpha * r + alpha ** 2 * vv
                if feasible(cn) or alpha >= -1.0:
                    break
                alpha = min((alpha - 1) / 2, -1.0)
            if not feasible(cn):
                cn = c2
            cn = F(cn)
            if ll(cn) < ll(c2):
                cn = c2
        L1 = ll(cn)
        c = cn
        if L1 - L0 < tol:
            break
        L0 = L1
    return c, ll(c)


def model_probs(pc, pb, n, nbest, c):
    qc = pc @ c[:-1] + c[-1] / n
    qb = pb @ c[:-1] + c[-1] * nbest / n
    return qc, qb


# ----------------------------------------------------------------------------- Part R
partR = {}
for p in panels:
    name = p['name']
    v, ch, n = p['values'], p['choice'], p['n']
    T = v.shape[0]
    top = np.nanmax(v, axis=1)
    nbest = (v == top[:, None]).sum(axis=1).astype(float)
    miss = v[np.arange(T), ch] < top
    sizes = tuple(int(s) for s in np.unique(n))
    res = {'sizes': list(sizes), 'observed_miss': [float(miss[n == s].mean()) for s in sizes]}
    res['observed_growth'] = slope_on_log2(sizes, res['observed_miss'])
    models = ['SAT', 'FT', 'FT_firstbest'] + (['SATscreen', 'FTscreen', 'SATscreen_firstbest', 'FTscreen_firstbest'] if name in SUMS else [])
    for mdl in models:
        dgrid = (0.0,) if mdl.startswith('SAT') else DS
        qmiss = np.zeros(T); qch = np.zeros(T)
        folds = []
        for s in sizes:
            tr = n != s; te = ~tr
            best = None
            for d in dgrid:
                pc, pb = comp[(name, mdl, d)]
                A = np.column_stack([pc[tr], 1.0 / n[tr]])
                cvec, LL = em_fit(A, np.ones(tr.sum()))
                if best is None or LL > best[2]:
                    best = (d, cvec, LL)
            d, cvec, LL = best
            pc, pb = comp[(name, mdl, d)]
            qc, qb = model_probs(pc[te], pb[te], n[te], nbest[te], cvec)
            qmiss[te] = 1 - qb; qch[te] = qc
            folds.append({'held_out_size': s, 'd': d, 'lapse': float(cvec[-1]), 'train_loglik': LL,
                          'pi': [float(x) for x in cvec[:-1] / (1 - cvec[-1])]})
        rates = [float(qmiss[n == s].mean()) for s in sizes]
        res[mdl] = {'miss_by_size': rates, 'growth': slope_on_log2(sizes, rates),
                    'mean_log2_p_chosen': float(np.log2(np.maximum(qch, 1e-300)).mean()), 'folds': folds}
    partR[name] = res
    print('R done', name, res['observed_growth'], res['SAT']['growth'], res['FT']['growth'], flush=True)

R1 = {'snack': partR['snack']['observed_growth'] - partR['snack']['SAT']['growth'],
      'reutskaja': partR['reutskaja']['observed_growth'] - partR['reutskaja']['SAT']['growth'],
      'cdm_mean': float(np.mean([partR[k]['observed_growth'] - partR[k]['SAT']['growth'] for k in SUMS]))}
R_secondary = {
    'FT_minus_SAT_growth_error': {k: abs(partR[k]['FT']['growth'] - partR[k]['observed_growth'])
                                  - abs(partR[k]['SAT']['growth'] - partR[k]['observed_growth']) for k in partR},
    'R1_screen_cdm_mean': float(np.mean([partR[k]['observed_growth'] - partR[k]['SATscreen']['growth'] for k in SUMS])),
    'R1_per_panel': {k: partR[k]['observed_growth'] - partR[k]['SAT']['growth'] for k in partR},
}


# ----------------------------------------------------------------------------- Part S
def fe_ols(X, y, g):
    Xd = X.copy(); yd = y.astype(float).copy()
    for u in np.unique(g):
        m = g == u
        Xd[m] -= Xd[m].mean(axis=0); yd[m] -= yd[m].mean()
    return np.linalg.lstsq(Xd, yd, rcond=None)[0]


rn = cdm.n.astype(float)
sel_n = rn[cdm.sel_round]
sel_exp = cdm.experiment[cdm.sel_round]
sel_person = cdm.participant[cdm.sel_round]
inc = cdm.sel_better > 0
partS = {}
std = {}
for e in (0, 1):
    m = inc & (sel_exp == e)
    X = np.column_stack([cdm.sel_value[m], np.log(cdm.sel_time[m]), np.log2(sel_n[m])])
    mu, sd = X.mean(axis=0), X.std(axis=0)
    std[e] = (mu, sd)
    b = fe_ols((X - mu) / sd, cdm.sel_stop[m], sel_person[m])
    partS[f'experiment_{e + 1}'] = {'raw': dict(zip(('value', 'log_time', 'log2_n'), map(float, b)))}

# SAT fitted on every set size of each panel
L36 = np.arange(1.0, 37.0)
sat_full = {}
for p in panels[2:]:
    pc, pb = comp[(p['name'], 'SAT', 0.0)]
    cvec, _ = em_fit(np.column_stack([pc, 1.0 / p['n']]), np.ones(p['n'].size))
    pi = cvec[:-1] / cvec[:-1].sum()
    sat_full[p['name']] = pi

pan_rows = {}
for e, key in ((0, 'cdm_standard'), (1, 'cdm_process')):
    for cx in (3, 7):
        pan_rows[f'{key}_{cx}'] = np.where((cdm.experiment == e) & (cdm.complexity == cx))[0]


def simulate_panel(rng, rows, pi, scale=None):
    """Simulate SAT searches over the panel's rounds: returns per round lists of (value, k inspected, time)."""
    vals = cdm.values[rows]; nn = cdm.n[rows]; R = rows.size
    t = L36[rng.choice(L36.size, size=R, p=pi)]
    keys = rng.random((R, vals.shape[1]))
    keys[~np.isfinite(vals)] = np.inf
    order = np.argsort(keys, axis=1)
    seq = np.take_along_axis(vals, order, axis=1)
    if scale is not None:
        gam = rng.gamma(2.0, scale, size=vals.shape)
        tim = np.cumsum(gam, axis=1)
    out = []
    for r in range(R):
        s = seq[r, :nn[r]]
        best = -np.inf; sels = []
        for k in range(nn[r]):
            if s[k] > best:
                best = s[k]
                sels.append((s[k], k + 1, tim[r, k] if scale is not None else np.nan))
                if best >= t[r]:
                    break
        out.append(sels)
    return out


# scale per panel from 20 benchmark searches per round
rng0 = np.random.default_rng(132999)
scales = {}
for name, rows in pan_rows.items():
    ks = []
    for _ in range(20):
        sims = simulate_panel(rng0, rows, sat_full[name])
        ks.extend(s[-1][1] for s in sims)
    scales[name] = float(cdm.final_time[rows].mean() / (2.0 * np.mean(ks)))

bench = {0: [], 1: []}
for sidx in range(100):
    rng = np.random.default_rng(133000 + sidx)
    data = {0: [], 1: []}
    for name, rows in pan_rows.items():
        e = 0 if 'standard' in name else 1
        sims = simulate_panel(rng, rows, sat_full[name], scales[name])
        for r, sels in zip(rows, sims):
            top = np.nanmax(cdm.values[r])
            for j, (val, k, tm) in enumerate(sels):
                if val < top:
                    data[e].append((val, np.log(tm), np.log2(cdm.n[r]), j == len(sels) - 1, cdm.participant[r]))
    for e in (0, 1):
        a = np.array(data[e], float)
        mu, sd = std[e]
        bench[e].append(fe_ols((a[:, :3] - mu) / sd, a[:, 3], a[:, 4].astype(int)))
    if sidx % 10 == 0:
        print('bench', sidx, flush=True)

for e in (0, 1):
    B = np.array(bench[e])
    k = f'experiment_{e + 1}'
    names = ('value', 'log_time', 'log2_n')
    partS[k]['benchmark_mean'] = dict(zip(names, map(float, B.mean(axis=0))))
    partS[k]['benchmark_mcse'] = dict(zip(names, map(float, B.std(axis=0, ddof=1) / np.sqrt(B.shape[0]))))
    partS[k]['raw_minus_benchmark'] = {nm: partS[k]['raw'][nm] - partS[k]['benchmark_mean'][nm] for nm in names}
partS['gamma_scale_by_panel'] = scales

# ----------------------------------------------------------------------------- Part D
ag = load_aguiar(Path(D + 'aguiar'))
partD = {}
for cost in COSTS:
    cnt = ag[cost]
    share = {m: cnt[m][0] / cnt[m].sum() for m in cnt}
    tot = {m: cnt[m].sum() for m in cnt}
    sets = {m: frozenset(MENUS[m - 1]) for m in cnt}
    diffs = [share[b] - share[a] for a in cnt for b in cnt if sets[a] < sets[b]]
    by = {}
    for s in sorted({size(m) for m in cnt}):
        ms = [m for m in cnt if size(m) == s]
        by[s] = float(sum(cnt[m][0] for m in ms) / sum(tot[m] for m in ms))
    x = np.array([np.log2(size(m)) for m in cnt]); y = np.array([share[m] for m in cnt]); w = np.array([tot[m] for m in cnt])
    xm = (w * x).sum() / w.sum(); ym = (w * y).sum() / w.sum()
    g = float((w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum())
    partD[cost] = {'nested_pairs': len(diffs), 'positive_differences': int(sum(d > 0 for d in diffs)),
                   'largest_difference': float(max(diffs)), 'default_share_by_size': {str(k): v for k, v in by.items()},
                   'growth_per_doubling': g}

json.dump({'partV': partV, 'partR': partR, 'R1': R1, 'R_secondary': R_secondary, 'partS': partS, 'partD': partD},
          open(OUT, 'w'), indent=1)
print('written')
