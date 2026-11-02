"""Smallest effect each confirmatory test could detect, from its participant bootstrap.

With the bootstrap standard error SE taken as the width of the 95 percent percentile interval divided by 3.92, a two
sided test at alpha 0.05 detects an effect of 2.80 SE with 80 percent power (1.96 + 0.84). Descriptive, for reading
the negative results. Usage, python3 -I scripts/detectable_effects.py > results/detectable_effects.md"""
import json
from pathlib import Path

TESTS = [  # (paper part, label, results folder, path in summary, unit)
    ("H1", "Within one study, high minus low cost growth (H1w)", "h1_within", ["H1w"], "misses per doubling"),
    ("H1", "Three seconds, people's growth minus CL's (O1)", "overload_premonition", ["datasets", "reutskaja", "observed_minus_cl"], "misses per doubling"),
    ("H1", "Sums, people's growth minus CL's (O1)", "overload_premonition", ["datasets", "cdm", "observed_minus_cl"], "misses per doubling"),
    ("H2", "Snack, return gain over CL", "h2_return", ["snack|gain|return"], "bits per choice"),
    ("H2", "Snack, return gain minus dwell gain", "h2_return", ["snack|return_minus_dwell"], "bits per choice"),
    ("H2", "Three seconds, return gain over CL", "h2_return", ["reutskaja|gain|return"], "bits per choice"),
    ("H2", "Three seconds, return gain minus dwell gain", "h2_return", ["reutskaja|return_minus_dwell"], "bits per choice"),
    ("H3", "Three seconds, trial forecast minus near best count (O2)", "overload_premonition", ["datasets", "reutskaja", "sr_auc_minus_near_best"], "area under the curve"),
    ("H3", "Sums, trial forecast minus near best count (O2)", "overload_premonition", ["datasets", "cdm", "sr_auc_minus_near_best"], "area under the curve"),
    ("Supplement", "Filename cue, A3b", "h3_cue", ["all|A3b|CL"], "completeness points"),
    ("Supplement", "S2, held out time, search model minus CL", "search_confirmatory", ["S2", "SR_minus", "CL"], "log time error"),
    ("Supplement", "S3, slow decisions, search model minus CL", "search_confirmatory", ["S3", "SR_minus_CL"], "rank correlation"),
    ("Supplement", "G1, held out time, two look minus search model", "two_look_gaze", ["pooled", "time", "TLS_minus_SR"], "log time error"),
    ("Supplement", "G2, slow decisions, two look minus search model", "two_look_gaze", ["pooled", "slow", "TLS_minus_SR"], "rank correlation"),
    ("Supplement", "C1, what ends a search", "stopping_cdm", ["pooled", "stop", "value_minus_share"], "bits"),
]
L = ["# Smallest detectable effect of each confirmatory test", "",
     "80 percent power, two sided alpha 0.05, from the bootstrap standard error (95 percent interval width / 3.92).", "",
     "| Part | Test | Estimate [95% interval] | Detectable effect | Unit |", "|---|---|---|---|---|"]
for part, label, folder, path, unit in TESTS:
    e = json.load(open(Path("results") / folder / "summary.json"))
    for k in path:
        e = e[k]
    lo, hi = e["percentile"]
    mde = 2.80 * (hi - lo) / 3.92
    L.append(f"| {part} | {label} | {e['point']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {mde:.3f} | {unit} |")
print("\n".join(L))
