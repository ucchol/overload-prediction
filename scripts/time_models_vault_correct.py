"""Corrected vault predictions for the rising threshold LBA and MLBA LBA races (deviation of 5 November 2026,
docs/time_models_log.md).

For each set and each of LBA_n and MLBA_LBA_n, the original vault fit and the warm start refit are compared by their
negative log likelihood on the training participants. The original is replaced only when it fails the nesting check (worse than the nested fixed threshold fit by more than 1) and the refit passes it, with its calibrated version. Every
other model's predictions are copied unchanged. Writes OUT/<set>_pred.npz and OUT/<set>_summary.json in the format of
scripts/time_models_run.py, plus OUT/corrections.json listing what was replaced.
Usage, python3 -I scripts/time_models_vault_correct.py VAULT_DIR REFIT_DIR OUT
"""
import json
import sys
from pathlib import Path

import numpy as np

vdir, rdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SETS = ["dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016"]
log = {}
for s in SETS:
    z = dict(np.load(vdir / f"{s}_pred.npz")); zr = np.load(rdir / f"{s}_pred.npz")
    sm = json.load(open(vdir / f"{s}_summary.json")); sr = json.load(open(rdir / f"{s}_summary.json"))
    assert np.array_equal(z["logt"], zr["logt"]) and np.array_equal(z["pid"], zr["pid"])
    log[s] = {}
    for m in ("LBA_n", "MLBA_LBA_n"):
        orig, ref = sm["extras"][m][0], sr["extras"][m][0]
        # replace only a fit that fails the nesting check, by a refit that passes it
        use_refit = orig["nll"] > ref["nll_fixed"] + 1.0 and ref["nll"] <= ref["nll_fixed"] + 1.0
        log[s][m] = {"nll_original": orig["nll"], "nll_refit": ref["nll"], "nll_fixed": ref["nll_fixed"],
                     "original_converged": bool(orig["nll"] <= ref["nll_fixed"] + 1.0), "kept": "refit" if use_refit else "original"}
        if use_refit:
            for k in (m, m + "_cal"):
                z[k] = zr[k]
                sm["r2"][k] = sr["r2"][k]
            sm["extras"][m] = sr["extras"][m]
    np.savez_compressed(out / f"{s}_pred.npz", **z)
    json.dump(sm, open(out / f"{s}_summary.json", "w"), indent=1, default=float)
json.dump(log, open(out / "corrections.json", "w"), indent=1)
for s, r in log.items():
    print(s, {m: (v["kept"], round(v["nll_original"]), round(v["nll_refit"]), round(v["nll_fixed"])) for m, v in r.items()})
