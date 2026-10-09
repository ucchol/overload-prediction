"""Bootstrap weighting check: replicate weights equal physically duplicating each drawn participant's trials.

Prints derived quantities only.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_snack as rs  # noqa: E402
import snack_core as sc  # noqa: E402

data = sc.load_data()
probs = rs.build_problems(data)
b = 3
draw = np.random.default_rng(20000 + b).integers(0, data["P"], size=data["P"])
mult = np.bincount(draw, minlength=data["P"])
w = mult.astype(float)[data["pidx"]]
weighted = rs.evaluate_sample(data, probs, w, b)

# Physically duplicated data set, unit weights
subj, chs, Vs, Ls = [], [], [], []
start = 0
for k in range(4):
    V = data["V_by_size"][k]
    L = data["looked_by_size"][k]
    rows = np.arange(start, start + len(V))
    start += len(V)
    rep = np.repeat(np.arange(len(V)), mult[data["pidx"][rows]])
    subj.append(data["pidx"][rows][rep])
    chs.append(data["choice"][rows][rep])
    Vs.append(V[rep])
    Ls.append(L[rep])
dup = sc.build_data(subj, chs, Vs, Ls)
dprobs = rs.build_problems(dup)
plain = rs.evaluate_sample(dup, dprobs, np.ones(len(dup["choice"])), b)

diff = 0.0
for key in weighted["lam"]:
    diff = max(diff, float(np.max(np.abs(weighted["lam"][key] - plain["lam"][key]) / plain["lam"][key])))
    diff = max(diff, float(np.max(np.abs(weighted["L_by"][key] - plain["L_by"][key]))))
for key in ("C3_all", "C3_seen"):
    diff = max(diff, abs(weighted[key][0] - plain[key][0]), abs(weighted[key][1] - plain[key][1]) / plain[key][1])
for name in weighted["alone"]:
    diff = max(diff, float(np.max(np.abs(weighted["alone"][name] - plain["alone"][name]) / plain["alone"][name])))
same_c1 = np.array_equal(weighted["e_mod"], plain["e_mod"]) and np.allclose(weighted["e_obs"], plain["e_obs"], rtol=1e-12)
same_desc = np.allclose(weighted["share_looked"], plain["share_looked"], rtol=1e-12) and np.allclose(weighted["share_chosen_looked"], plain["share_chosen_looked"], rtol=1e-12)
print(f"weighted replicate equals duplicated data: max difference over every fit {diff:.1e}; C1 effective numbers identical {same_c1}; shares identical {same_desc}")
print("PASS" if diff < 1e-9 and same_c1 and same_desc else "FAIL")
