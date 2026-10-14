"""Point-estimate sensitivity of two resolved ambiguities (not part of the specified results).

1. Regret targets. The brief takes every row's regret, so the unmatched file responses (coded regret 100)
   enter the T1 and T6 targets. Alternative: mean regret over matched choices only.
2. Options forecast. The brief's n is the displayed count (30 for LT6). Alternative: intended 32.
Reads only this folder's own results.json. Prints derived values only (no counts).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovl_answers as ov  # noqa: E402
import process_core as pc  # noqa: E402
import run_process as rp  # noqa: E402
from process_core import core  # noqa: E402

pc.assert_local(ov, pc, rp, core)
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
S, regret, seconds = ov.load_answers()
index = ov.overload_index(S)

out = {}
# 1. regret targets over matched choices only
tasks = {n: core.build_task(n) for n in core.TASKS}
matched = {n: core.load_choice_matrix(tasks[n]["trials"]) >= 0 for n in core.TASKS}
rg_all = np.concatenate([regret[:, rp.COLS[n]].mean(axis=0) for n in core.TASKS])
rg_matched = np.concatenate([np.array([regret[matched[n][:, c], rp.COLS[n].start + c].mean() for c in range(6)]) for n in core.TASKS])
names = tasks["file"]["names"] + tasks["laptop"]["names"]
# (per trial target shifts are not stored: with a single coded response they would reveal the sample size)
both = R["premonition"]["both"]
out["both_regret_matched_only"] = {}
for k in rp.REGRET_MODELS:
    f = np.array([both["regret_forecasts"][k][nm]["point"] for nm in names])
    s_all, s_m = pc.regret_stats(f, rg_all), pc.regret_stats(f, rg_matched)
    out["both_regret_matched_only"][k] = {"error": s_m["error"], "error_as_specified": s_all["error"],
                                          "error_minus_constant": s_m["error_minus_constant"],
                                          "error_minus_constant_as_specified": s_all["error_minus_constant"],
                                          "spearman": s_m["spearman"], "spearman_as_specified": s_all["spearman"]}
# 2. options forecast with the intended 32 for LT6
ov_targets = np.concatenate([index[:, rp.COLS[n]].mean(axis=0) for n in core.TASKS])
n_disp = np.concatenate([tasks[n]["nvec"] for n in core.TASKS])
n_int = np.concatenate([tasks[n]["sizes"] for n in core.TASKS]).astype(float)
out["options_spearman"] = {
    "laptop_displayed": pc.spearman0(n_disp[6:], ov_targets[6:]), "laptop_intended": pc.spearman0(n_int[6:], ov_targets[6:]),
    "both_displayed": pc.spearman0(n_disp, ov_targets), "both_intended": pc.spearman0(n_int, ov_targets),
}
with open(os.path.join(HERE, "sensitivity_process.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
