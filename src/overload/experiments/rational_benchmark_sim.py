"""Simulated behavior over the real displays for the checks of Amendment 13 before freezing.

sat. Choices from SAT fitted to every set size of each panel (random order).
ft. Choices from a falling threshold, the SAT distribution with d = 0.5 scale units per inspected option.
fixed_stop (ft_stop, falling by one dollar per option). Caplin, Dean and Martin selections from a fixed threshold search in random order, each new best found
selected, inspection times drawn per option, so stopping depends on value only through the threshold.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from . import rational_benchmark as rb

SIM_SEED = 13


def _fit_all(p):
    a_ch, _ = rb.sat_tables(p)
    pi, e, _ = rb.fit_em(a_ch, p.n, np.ones(p.n.size))
    return pi, e


def _run(v, order, t, delta):
    """Index chosen, list of (option, position in order) selections."""
    best, sels = -np.inf, []
    for k, j in enumerate(order):
        if v[j] > best:
            best, arg = v[j], j
            sels.append((j, k))
        if best >= t - delta * k:
            return arg, sels
    return arg, sels


def simulate(panels, data, kind):
    rng = np.random.default_rng(SIM_SEED)
    out = []
    for p in panels:
        pi, e = _fit_all(p)
        if kind in ("fixed_stop", "ft_stop"):
            out.append(p)
            continue
        delta = 0.5 if kind == "ft" else 0.0
        ch = np.empty(p.n.size, int)
        for i in range(p.n.size):
            k = int(p.n[i])
            v = p.values[i, :k]
            if rng.random() < e:
                ch[i] = rng.integers(k)
                continue
            t = p.levels[rng.choice(p.levels.size, p=pi)]
            ch[i], _ = _run(v, rng.permutation(k), t, delta)
        out.append(replace(p, choice=ch))
    if kind in ("fixed_stop", "ft_stop"):
        data = _selections(data, panels, rng, 1.0 if kind == "ft_stop" else 0.0)
    return out, data


def _selections(data, panels, rng, delta=0.0):
    fits = {}
    for p in panels:
        if p.name.startswith("cdm"):
            fits[p.name] = _fit_all(p)
    names = {0: "cdm_standard", 1: "cdm_process"}
    levels = np.arange(1.0, 37.0)
    sr, sv, st, sp, sb, ss, sf, choice, ft = [], [], [], [], [], [], [], [], []
    for r in range(data.rounds):
        k = int(data.n[r])
        v = data.values[r, :k]
        pi, e = fits[f"{names[int(data.experiment[r])]}_{int(data.complexity[r])}"]
        t = levels[rng.choice(levels.size, p=pi)]
        order = rng.permutation(k)
        arg, sels = _run(v, order, t, delta)
        times = np.cumsum(rng.gamma(2.0, 2.5 if data.complexity[r] == 3 else 5.0, size=k))
        for q, (j, pos) in enumerate(sels):
            sr.append(r); sv.append(v[j]); st.append(max(times[pos], 1.0)); sp.append(j)
            sb.append(int((v > v[j]).sum())); ss.append(q == len(sels) - 1); sf.append(q == 0)
        choice.append(arg); ft.append(max(times[sels[-1][1]], 1.0))
    return replace(data, choice=np.array(choice), final_time=np.array(ft), sel_round=np.array(sr),
                   sel_value=np.array(sv, float), sel_time=np.array(st), sel_position=np.array(sp),
                   sel_better=np.array(sb), sel_stop=np.array(ss), sel_first=np.array(sf))
