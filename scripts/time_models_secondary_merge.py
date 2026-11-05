"""Corrected secondary measures (deviation of 5 November 2026, docs/time_models_log.md).

Rank correlations and bits per choice come from the rerun on the corrected vault predictions, for every model. The
unseen set size entries of the rising threshold LBA race (and its calibrated version) come from the rerun that keeps
the original cold fit at each left out size unless it fails the nesting check, and then uses the warm start fit. Every other unseen set size entry is copied from
the original run. The rising MLBA LBA race's unseen set size entries come from cold fits whose convergence was not
checked, so they are marked and not reported.
Usage, python3 -I scripts/time_models_secondary_merge.py ORIGINAL_JSON FIX_JSON OUT_JSON [RANKBITS_JSON]
RANKBITS_JSON, when given, supplies the rank correlations and bits per choice instead of FIX_JSON (a run on the final
corrected vault predictions with no unseen set size models).
"""
import json
import sys

orig, fix, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
rb = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else fix
res = {}
for k, r in orig.items():
    f = fix[k]
    rec = {"rank_corr": rb[k]["rank_corr"], "bits": rb[k]["bits"], "unseen_size": r.get("unseen_size", {})}
    u = rec["unseen_size"]
    if u:
        fu = f["unseen_size"]
        assert fu["sizes"] == u["sizes"] and fu["rmse_no_difference"] == u["rmse_no_difference"]
        for m in ("LBA_n", "LBA_n_cal"):
            u[m] = fu[m]
        u["convergence"] = fu.get("convergence", {})
        u["note"] = ("LBA_n and LBA_n_cal keep the cold fit at each left out size unless it fails the nesting check, "
                     "and then the warm start fit. "
                     "MLBA_LBA_n entries come from cold fits whose convergence was not checked and are not reported.")
    res[k] = rec
json.dump(res, open(out, "w"), indent=1, default=float)
for k, r in res.items():
    u = r["unseen_size"]
    if u:
        c = u["convergence"].get("LBA_n", [])
        print(k, "LBA_n skill %+.3f" % u["LBA_n"]["skill"], "cal %+.3f" % u["LBA_n_cal"]["skill"],
              "chosen", [x["chosen"] for x in c], "converged", all(x["converged"] for x in c))
