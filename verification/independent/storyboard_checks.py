"""Checks before the Amendment 4 rerun, and the rubric against the documented laptop scoring rule.

1. The file task rebuilt from the current stimuli reproduces every point value in results.json["file"].
2. The pipeline run on laptop_trials_aamas.json reproduces results.json["laptop"] (point values) and the
   first bootstrap replicates of both tasks in boot_B1000.npz, so the reference file is the one used
   before and the code changes since then are neutral.
3. Correlation, per laptop trial, of the designers' rubric scores with the documented scoring rule
   (weights 4, 4, 3, 3, 2, 2, 1 on the coded memory, graphics, storage, processor, battery, brand and
   price levels), under both definition files.
Prints derived values only (no participant or choice counts).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis  # noqa: E402
import core  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
MI = {m: i for i, m in enumerate(core.ALL_MODELS)}
SM = {m: i for i, m in enumerate(core.SOFTMAX_MODELS)}
RULE = np.array([4, 4, 3, 3, 2, 2, 1], float)


def point_values(task: dict, pt: dict) -> dict:
    """Every point value of the R0 and H1 layout that does not need the bootstrap, keyed like results.json."""
    ev, names, C = pt["ev"], task["names"], len(task["names"])
    out = {"rubric_shares": dict(zip(core.ATTRS[task["task"]], task["r"])), "Lstar": dict(zip(names, pt["Lstar"]))}
    cnt = pt["cnt"]
    out["shares"] = {t["name"]: {o: cnt[c, i] / cnt[c].sum() for i, o in enumerate(t["names"])} for c, t in enumerate(task["trials"])}
    for alias, model in core.R0_ALIAS.items():
        m, sm = MI[model], SM[model]
        out[f"R0/{alias}/lambda"] = ev["lam"][sm, C]
        for key, arr in (("ho_bits", ev["L_ho"][m]), ("fit_bits", ev["L_fit"][m]), ("overlap_ho", ev["O_ho"][m])):
            out[f"R0/{alias}/{key}"] = dict(zip(names, arr))
    for e_i, e in enumerate(core.EBA_MODELS):
        out[f"A1b/{e}/lapse"] = ev["eps"][e_i, C]
    nvec = task["nvec"]
    out["B1/observed_share"] = dict(zip(names, pt["e_obs"] / nvec))
    out["B1/model_share"] = dict(zip(names, pt["e_mod"] / nvec))
    out["B1/world_gap"] = dict(zip(names, (pt["e_world"] - pt["e_mod"]) / nvec))
    return out


def stored_values(res: dict, names: list) -> dict:
    out = {"rubric_shares": res["rubric_shares"], "Lstar": res["Lstar"], "shares": res["shares"]}
    for alias in core.R0_ALIAS:
        e = res["R0"][alias]
        out[f"R0/{alias}/lambda"] = e["lambda"]
        for key in ("ho_bits", "fit_bits", "overlap_ho"):
            out[f"R0/{alias}/{key}"] = e[key]
    for e in core.EBA_MODELS:
        out[f"A1b/{e}/lapse"] = res["H1"]["A1b"][e]["lapse"]
    pt = res["H1"]["B1"]["per_trial"]
    for key in ("observed_share", "model_share", "world_gap"):
        out[f"B1/{key}"] = {n: pt[n][key] for n in names}
    return out


def max_diff(a, b) -> float:
    if isinstance(a, dict):
        assert set(a) == set(b), "key mismatch"
        return max(max_diff(a[k], b[k]) for k in a)
    return abs(float(a) - float(b))


def compare(label, task, pt, res):
    mine, stored = point_values(task, pt), stored_values(res, task["names"])
    worst = max(max_diff(mine[k], stored[k]) for k in stored)
    print(f"{'PASS' if worst < 1e-12 else 'FAIL'} {label}: max abs difference over every stored point value {worst:.1e}")
    return worst < 1e-12


ok = True
# ---------------------------------------------------------------------------- 1 and 2, point values
tasks_cur = {"file": core.build_task("file")}
tasks_old = {"file": tasks_cur["file"], "laptop": core.build_task("laptop", stimuli_file="laptop_trials_aamas.json")}
pts_cur = {"file": analysis.point_sample(tasks_cur["file"])}
pts_old = {"laptop": analysis.point_sample(tasks_old["laptop"]), "file": pts_cur["file"]}
ok &= compare("file task from current stimuli reproduces results.json[file]", tasks_cur["file"], pts_cur["file"], R["file"])
ok &= compare("laptop task from laptop_trials_aamas.json reproduces results.json[laptop]", tasks_old["laptop"], pts_old["laptop"], R["laptop"])

# ---------------------------------------------------------------------------- 2, first bootstrap replicates
boot = np.load(os.path.join(HERE, "boot_B1000.npz"))
store = analysis.run_bootstrap(tasks_old, pts_old, 5, log_every=5)
for n in core.TASKS:
    same = all(np.array_equal(store[n][k], boot[f"{n}__{k}"][:5]) for k in ("L", "O", "e_obs", "e_mod"))
    ok &= same
    print(f"{'PASS' if same else 'FAIL'} {n}: replicates 0 to 4 identical to boot_B1000.npz (held out log scores, overlaps, B1 effective numbers)")

# ---------------------------------------------------------------------------- laptop trials 1 to 4 identical as coded
lap_new = core.load_trials("laptop")
lap_old = core.load_trials("laptop", "laptop_trials_aamas.json")
same14 = all(np.array_equal(a["X"], b["X"]) and np.array_equal(a["U"], b["U"]) and a["names"] == b["names"] for a, b in zip(lap_new[:4], lap_old[:4]))
ok &= same14
print(f"{'PASS' if same14 else 'FAIL'} laptop trials 1 to 4 have identical names, coded levels and rubric scores in both files")

# ---------------------------------------------------------------------------- 3, rubric against the documented rule
print("\nCorrelation of rubric scores with the documented rule 4,4,3,3,2,2,1 (Pearson, per laptop trial)")
for label, trials in (("storyboard (laptop_trials.json)", lap_new), ("earlier (laptop_trials_aamas.json)", lap_old)):
    vals = []
    for t in trials:
        rule = t["X"] @ RULE
        price_const = np.ptp(t["X"][:, 6]) == 0
        vals.append(f"{t['name']} {np.corrcoef(rule, t['U'])[0, 1]:.3f}{'' if price_const else ' (price varies)'}")
    print(f"  {label}: " + ", ".join(vals))

# ---------------------------------------------------------------------------- preview of the corrected laptop inputs
new_task = core.build_task("laptop")
print("\nlaptop rubric shares, earlier:", {a: round(float(v), 4) for a, v in zip(core.ATTRS["laptop"], tasks_old["laptop"]["r"])})
print("laptop rubric shares, storyboard:", {a: round(float(v), 4) for a, v in zip(core.ATTRS["laptop"], new_task["r"])})
for label, task in (("earlier", tasks_old["laptop"]), ("storyboard", new_task)):
    cors, incons = [], []
    for t in task["trials"]:
        cors.append(f"{np.corrcoef(t['X'] @ task['r'], t['U'])[0, 1]:.3f}")
        groups = {}
        for i, o in enumerate(t["names"]):
            groups.setdefault(tuple(t["X"][i, :6]), []).append(int(t["U"][i]))
        incons.append(sum(1 for g in groups.values() if len(g) > 1 and max(g) - min(g) >= 15))
    print(f"{label}: corr(U, CL score) per trial {cors}; groups of identically coded options with U range >= 15 per trial {incons}")

print("\nALL PRE-RUN CHECKS PASSED" if ok else "\nSOME PRE-RUN CHECKS FAILED")
