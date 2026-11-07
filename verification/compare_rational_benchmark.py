"""Compare the package's Amendment 13 point estimates with the blind reimplementation.

Usage, python3 -I verification/compare_rational_benchmark.py results/rational_benchmark/summary.json
Tolerances. Part V and the raw Part S coefficients 1e-6 (same arithmetic). SAT growth 1e-5 (exact probabilities,
both fits at the concave maximum). FT growth 5e-4 (Monte Carlo stops differ where the blind code compares with a
tolerance of 1e-9). Screen order rows against the blind code's first best variant, which is what the package does
(see docs/deviations_from_plan.md). Part D counts exactly. The benchmark of Part S is simulated with a different
random number order, so it is compared within five Monte Carlo standard errors of the difference plus 0.01.
"""
import json
import sys
from pathlib import Path

pkg = json.load(open(sys.argv[1]))
blind = json.load(open(Path(__file__).parent / "independent_rational" / "blind.json"))
P = pkg["point"]
worst, fails = {}, []


def check(name, a, b, tol):
    d = abs(a - b)
    worst[name.split(":")[0]] = max(worst.get(name.split(":")[0], 0.0), d)
    if d > tol:
        fails.append(f"{name} package {a:.8f} blind {b:.8f}")


for p, v in P["V"].items():
    for k, x in v.items():
        check(f"V:{p}/{k}", x, blind["partV"][p]["mean" if k == "mean_option" else k], 1e-6)
for p, r in P["R"].items():
    if "rate" not in r:
        continue
    br = blind["partR"][p]
    check(f"observed:{p}", r["growth"]["observed"], br["observed_growth"], 1e-9)
    check(f"SAT:{p}", r["growth"]["SAT"], br["SAT"]["growth"], 1e-5)
    check(f"FT:{p}", r["growth"]["FT"], br["FT"]["growth"], 5e-4)
    if "SAT_screen" in r["growth"]:
        check(f"SAT_screen:{p}", r["growth"]["SAT_screen"], br["SATscreen_firstbest"]["growth"], 1e-5)
        check(f"FT_screen:{p}", r["growth"]["FT_screen"], br["FTscreen_firstbest"]["growth"], 5e-4)
for d in ("snack", "reutskaja", "cdm_mean"):
    check(f"R1:{d}", P["R"][d]["R1"], blind["R1"][d], 1e-5)
for g, e in (("cdm_standard", "experiment_1"), ("cdm_process", "experiment_2")):
    for k in ("value", "log_time", "log2_n"):
        check(f"S_raw:{g}/{k}", P["S"][g][k], blind["partS"][e]["raw"][k], 1e-6)
        se = (pkg["S_benchmark"][g][k]["mc_sd"] ** 2 + blind["partS"][e]["benchmark_mcse"][k] ** 2) ** 0.5
        check(f"S_benchmark:{g}/{k}", pkg["S_benchmark"][g][k]["mean"], blind["partS"][e]["benchmark_mean"][k], 5 * se + 0.01)
text = Path(sys.argv[1]).parent / "S1_text_benchmark.json"
if text.exists():
    tb = json.load(open(text))
    for g, e in (("cdm_standard", "experiment_1"), ("cdm_process", "experiment_2")):
        for k in ("value", "log_time", "log2_n"):
            se = (tb["benchmark"][g][k]["mc_sd"] ** 2 + blind["partS"][e]["benchmark_mcse"][k] ** 2) ** 0.5
            check(f"S_text_benchmark:{g}/{k}", tb["benchmark"][g][k]["mean"], blind["partS"][e]["benchmark_mean"][k], 5 * se)
            check(f"S1_text:{g}/{k}", tb["S1"][f"{g}/{k}_minus_benchmark"]["point"], blind["partS"][e]["raw_minus_benchmark"][k], 5 * se)
for c, v in pkg["D1"].items():
    bd = blind["partD"][c]
    if v["pairs"] != bd["nested_pairs"] or v["positive_differences"] != bd["positive_differences"]:
        fails.append(f"D counts {c}")
    check(f"D_largest:{c}", v["largest_difference"], bd["largest_difference"], 1e-9)
    check(f"D_growth:{c}", v["growth"]["point"], bd["growth_per_doubling"], 1e-9)
for k, v in sorted(worst.items()):
    print(f"{k:14s} largest difference {v:.2e}")
print("FAIL\n" + "\n".join(fails) if fails else "all within tolerance")
