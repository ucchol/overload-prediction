"""Brief test 4: the full pipeline on synthetic data where choices, search and response times come from SR.

Usage: python -I check_e2e.py RESULTS_JSON TRUTH_JSON BOOT_NPZ
"""

from __future__ import annotations

import json
import sys

import numpy as np

res = json.load(open(sys.argv[1]))
truth = json.load(open(sys.argv[2]))
boot = np.load(sys.argv[3])
ok_all = True


def check(label, cond):
    global ok_all
    ok_all &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label, flush=True)


lam_true = truth["lambda"]
lam_sr = np.array(res["lambda"]["SR"])
sd = boot["lam"][:, 3, :].std(axis=0)  # model order CL, DbS_avg, MLBA_avg, SR
z = np.abs(lam_sr - lam_true) / sd
check(f"SR recovers lambda {lam_true}: held-out lambdas {np.round(lam_sr, 4).tolist()}, at most {z.max():.1f} bootstrap s.d. away", z.max() < 4)
s1 = res["S1"]
d = s1["predicted_elasticity"]["point"] - s1["observed_elasticity"]["point"]
check(f"S1: predicted elasticity {s1['predicted_elasticity']['point']:.4f} against observed {s1['observed_elasticity']['point']:.4f} "
      f"(difference {d:+.4f}, percentile {np.round(s1['predicted_minus_observed']['percentile'], 4).tolist()}), within margin {s1['within_margin']}, "
      f"supported {s1['supported']}", abs(d) < 0.05 and s1["within_margin"] and s1["supported"])
ratio = np.array(s1["observed_over_predicted_by_size"])
check(f"S1: observed over predicted distinct items per set size {np.round(ratio, 4).tolist()} (close to one when search is SR)",
      np.all(np.abs(ratio - 1) < 0.05) and np.allclose(ratio, np.array(s1["observed_by_size"]) / np.array(s1["predicted_by_size"])))
s2 = res["S2"]
e = {m: s2["error"][m]["point"] for m in ("CL", "DbS_avg", "MLBA_avg", "SR")}
check(f"S2: time errors CL {e['CL']:.4f}, DbS_avg {e['DbS_avg']:.4f}, MLBA_avg {e['MLBA_avg']:.4f}, SR {e['SR']:.4f}; "
      f"SR - CL {s2['SR_minus']['CL']['point']:+.4f} {np.round(s2['SR_minus']['CL']['percentile'], 4).tolist()}, "
      f"SR - DbS_avg {s2['SR_minus']['DbS_avg']['point']:+.4f}; supported {s2['supported']}",
      e["SR"] < e["CL"] and e["SR"] < e["DbS_avg"] and s2["supported"])
s3 = res["S3"]
c = s3["correlation"]["looks_SR"]
check(f"S3: SR's mean within-participant correlation {c['point']:+.4f} {np.round(c['percentile'], 4).tolist()}, supported {s3['supported']}",
      c["point"] > 0 and s3["supported"])
s5 = res["S5"]["SR_minus_CL_percent"]
print(f"info S5: SR minus CL {s5['point']:+.2f} percent, percentile {np.round(s5['percentile'], 2).tolist()}, basic {np.round(s5['basic'], 2).tolist()}, "
      f"non-inferior {res['S5']['noninferior']}, better {res['S5']['better']}, worse {res['S5']['worse']}")
print(f"info S4: SR minus CL {res['S4']['SR_minus_CL']['point']:+.4f}, SR minus constant {res['S4']['minus_constant']['SR']['point']:+.4f}")
print(f"info implied elasticity {res['implied_elasticity']['point']:+.4f}, implied CL sensitivities {np.round(res['implied_cl_sensitivity'], 4).tolist()}")
print(f"info times chance {({m: round(v['point'], 3) for m, v in res['choices']['times_chance'].items()})}")
print("\nEND-TO-END CHECKS PASSED" if ok_all else "\nSOME END-TO-END CHECKS FAILED")
