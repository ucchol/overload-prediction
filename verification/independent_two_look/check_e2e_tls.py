"""Brief check 3: on synthetic data from two look satisficing, the pipeline recovers lambda and TLS times best.

Usage: python -I check_e2e_tls.py RESULTS_JSON TRUTH_JSON BOOT_NPZ
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


lam = np.array(res["lambda"])
z = np.abs(lam - truth["lambda"]) / boot["lam"].std(axis=0)
check(f"lambda {truth['lambda']} recovered: held-out {np.round(lam, 4).tolist()}, at most {z.max():.1f} bootstrap s.d. away", z.max() < 4)
e = {m: res["E1"]["error"][m]["point"] for m in ("CL", "SR", "TLS")}
d = res["E1"]["TLS_minus"]["CL"]
check(f"E1: time errors CL {e['CL']:.4f}, SR {e['SR']:.4f}, TLS {e['TLS']:.4f}; TLS - CL {d['point']:+.4f} {np.round(d['percentile'], 4).tolist()}, clear {d['clear']}",
      e["TLS"] < e["CL"] and e["TLS"] < e["SR"] and d["clear"])
ratio = np.array(res["E3"]["observed_over_predicted"])
check(f"E3: observed over TLS predicted distinct items {np.round(ratio, 4).tolist()}; elasticities observed "
      f"{res['E3']['observed_elasticity']['point']:.4f}, TLS {res['E3']['predicted_elasticity']['point']:.4f}",
      np.all(np.abs(ratio - 1) < 0.05) and abs(res["E3"]["predicted_minus_observed_elasticity"]["point"]) < 0.05)
c = res["E2"]["correlation"]
check(f"E2: TLS looks {c['looks_TLS']['point']:+.4f} {np.round(c['looks_TLS']['percentile'], 4).tolist()}, SR looks {c['looks_SR']['point']:+.4f}, "
      f"top rating {c['top_rating']['point']:+.4f}", c["looks_TLS"]["point"] > 0 and c["looks_TLS"]["clear"])
print(f"info uniform attention: {res['uniform_attention']}")
print("\nEND-TO-END CHECKS PASSED" if ok_all else "\nSOME END-TO-END CHECKS FAILED")
