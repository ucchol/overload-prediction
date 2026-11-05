"""Final corrected vault predictions (deviation of 5 November 2026, docs/time_models_log.md). Starts from the corrected
predictions of scripts/time_models_vault_correct.py and replaces the rising threshold LBA race (and its calibrated
version) in every set by the fit of the fuller search (scripts/time_models_lba_search.py), with its fit record.
Usage, python3 -I scripts/time_models_vault_final.py CORRECTED_DIR SEARCH_DIR OUT
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

cdir, sdir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
search = json.load(open(sdir / "search.json"))
for s, rec in search.items():
    z = dict(np.load(cdir / f"{s}_pred.npz")); zs = np.load(sdir / f"{s}_pred.npz")
    assert np.array_equal(z["logt"], zs["logt"])
    z["LBA_n"], z["LBA_n_cal"] = zs["LBA_n"], zs["LBA_n_cal"]
    np.savez_compressed(out / f"{s}_pred.npz", **z)
    sm = json.load(open(cdir / f"{s}_summary.json"))
    sm["extras"]["LBA_n"] = [{k: rec[k] for k in ("theta", "nll", "bits_te", "slope", "gamma", "chosen")}]
    json.dump(sm, open(out / f"{s}_summary.json", "w"), indent=1, default=float)
shutil.copy(cdir / "corrections.json", out / "corrections.json")
shutil.copy(sdir / "search.json", out / "lba_search.json")
print("final predictions in", out)
