"""Fuller search for the rising threshold LBA race on the vault (deviation of 5 November 2026, docs/time_models_log.md).

For each set, the race is fitted on the participants outside the vault from ten cold starts and from the ten start
fixed threshold optimum with gamma 0.02, 0.3 and 0.7 (three starts each). Among fits whose expected log times are
finite and within 50 of zero on the training and vault trials, the lowest negative log likelihood is kept. Writes
OUT/<set>_pred.npz with LBA_n and LBA_n_cal (calibrated by one slope on the training participants) and
OUT/search.json with every candidate.
Usage, python3 -I scripts/time_models_lba_search.py DEV_CSV OUT SET...
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from overload import time_models as tm  # noqa: E402
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from time_models_run import vault_mask  # noqa: E402

np.seterr(all="ignore")
dev, out, sets = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
S = load_all(dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
log = {}
for name in sets:
    T = tm.from_timeset(S[name], SCALE[name])
    vault = vault_mask(int(T.pid.max()) + 1)
    D, V = T.take(~vault[T.pid]), T.take(vault[T.pid])
    dl = 3.0 if name == "reutskaja" else None

    def E(R, t):
        if dl is None:
            return R.expected_logt(t)
        return R.expected_logt_deadline(t, tm.relative_time(t) * dl / np.exp(t.logt))

    R0 = tm.Race("lba").fit(D, starts=10, deadline_s=dl)
    cands = [("cold", tm.Race("lba_n").fit(D, starts=10, deadline_s=dl))]
    for g in (0.02, 0.3, 0.7):
        th0 = np.r_[R0.theta[:-1], np.log(g), R0.theta[-1]]
        cands.append((f"warm {g}", tm.Race("lba_n", th0).fit(D, starts=3, deadline_s=dl)))
    rec = {"nll_fixed_ten_starts": R0.nll, "candidates": []}
    best = None
    for lab, R in cands:
        etr, ete = E(R, D), E(R, V)
        ok = bool(np.all(np.isfinite(etr)) and np.all(np.isfinite(ete)) and np.max(np.abs(etr)) < 50 and np.max(np.abs(ete)) < 50)
        rec["candidates"].append({"start": lab, "nll": R.nll, "gamma": float(np.exp(R.theta[-2])), "admissible": ok})
        if ok and (best is None or R.nll < best[1].nll):
            best = (lab, R, etr, ete)
    lab, R, etr, ete = best
    b = tm.lsq_center(etr[:, None], D.logt, D.pid)
    rec.update({"chosen": lab, "nll": R.nll, "gamma": float(np.exp(R.theta[-2])), "slope": float(b[0]),
                "theta": R.theta.tolist(), "bits_te": (R.choice_log2(V) + np.log2(V.n)).tolist()})
    np.savez_compressed(out / f"{name}_pred.npz", logt=V.logt, LBA_n=ete, LBA_n_cal=ete * b[0])
    log[name] = rec
    print(name, "chosen", lab, "nll %.1f" % R.nll, "gamma %.3f" % rec["gamma"], "slope %.2f" % b[0],
          [(c["start"], round(c["nll"], 1), round(c["gamma"], 3), c["admissible"]) for c in rec["candidates"]], flush=True)
    json.dump(log, open(out / "search.json", "w"), indent=1)
