"""Separate recomputation of the secondary time measures (scripts/time_models_secondary.py), written without the
package's model code or the secondary script.

(a) Within person rank correlations of every stored vault prediction with observed time, by pandas group ranks.
(b) The unseen set size skill of the set size law and the structure law, refitted here by numpy least squares on
    within person centered data of the participants outside the vault, at every left out size.
(c) CL bits per choice on the vault, with the CL sensitivity refitted here by scipy minimize_scalar.
Prints the largest absolute difference from the stored values for each part.
Usage, python3 -I verification/check_time_secondary.py DEV_CSV VAULT_DIR SECONDARY_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, "src")
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402  (data loading only)

dev, vdir, sec = Path(sys.argv[1]), Path(sys.argv[2]), json.load(open(sys.argv[3]))
S = load_all(dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))


def trials(ts, scale):
    lo, hi = scale
    rows = []
    for b in ts.blocks:
        for v, c, lt, p in zip(b.values, b.choice, b.logt, b.participant):
            rows.append(((np.asarray(v, float) - lo) / (hi - lo), int(c), float(lt), int(p)))
    return rows


def features(x):
    v = np.sort(x)[::-1]
    return [np.log2(len(x)), v[0], v[0] - v[1], v[1:].mean()]


worst = {"rank": 0.0, "skill": 0.0, "bits": 0.0}
for name, rec in sec.items():
    rows = trials(S[name], SCALE[name])
    P0 = max(r[3] for r in rows) + 1
    perm = np.random.default_rng(20261103).permutation(P0)
    vault_people = set(perm[: int(np.ceil(0.3 * P0))].tolist())
    dev_rows = [r for r in rows if r[3] not in vault_people]
    v_rows = [r for r in rows if r[3] in vault_people]
    z = np.load(vdir / f"{name}_pred.npz")
    assert np.allclose(z["logt"], [r[2] for r in v_rows])
    # (a)
    df = pd.DataFrame({"pid": z["pid"], "y": z["logt"]})
    for m, stored in rec["rank_corr"].items():
        df["p"] = z[m]
        def sp(g):
            if g["p"].nunique() < 2 or g["y"].nunique() < 2:
                return 0.0
            return g["p"].rank().corr(g["y"].rank())
        r = df.groupby("pid")[["p", "y"]].apply(sp).mean()
        worst["rank"] = max(worst["rank"], abs(r - stored["mean"]))
    # (c) CL bits
    xs = [r[0] for r in dev_rows]; cs = [r[1] for r in dev_rows]
    def nll(a):
        lam = np.exp(a); t = 0.0
        for x, c in zip(xs, cs):
            u = lam * x; t -= u[c] - (u.max() + np.log(np.exp(u - u.max()).sum()))
        return t
    lam = np.exp(minimize_scalar(nll, bounds=(-6, 6), method="bounded", options={"xatol": 1e-8}).x)
    bits = []
    for x, c, _, _ in v_rows:
        u = lam * x; lp = u[c] - (u.max() + np.log(np.exp(u - u.max()).sum()))
        bits.append(lp / np.log(2) + np.log2(len(x)))
    worst["bits"] = max(worst["bits"], abs(np.mean(bits) - rec["bits"]["CL"]["bits_per_choice"]))
    # (b) set size and structure laws at unseen sizes
    u = rec.get("unseen_size")
    if u:
        vp = np.unique([r[3] for r in v_rows], return_inverse=True)[1]
        nv = np.array([len(r[0]) for r in v_rows]); yv = np.array([r[2] for r in v_rows])
        Fv = np.array([features(r[0]) for r in v_rows])
        for law, cols in (("S1", [0]), ("S2", [0, 1, 2, 3])):
            sq, sq0 = 0.0, 0.0
            for s in u["sizes"]:
                tr = [r for r in dev_rows if len(r[0]) != s]
                F = np.array([features(r[0]) for r in tr])[:, cols]; y = np.array([r[2] for r in tr])
                pid = np.array([r[3] for r in tr])
                Fc = F - pd.DataFrame(F).groupby(pid).transform("mean").to_numpy()
                yc = y - pd.Series(y).groupby(pid).transform("mean").to_numpy()
                b = np.linalg.lstsq(Fc, yc, rcond=None)[0]
                pred = Fv[:, cols] @ b
                for p in np.unique(vp):
                    m = vp == p; at = m & (nv == s); ot = m & (nv != s)
                    if at.any() and ot.any():
                        dobs = yv[at].mean() - yv[ot].mean(); dp = pred[at].mean() - pred[ot].mean()
                        sq += (dp - dobs) ** 2; sq0 += dobs ** 2
            worst["skill"] = max(worst["skill"], abs((1 - sq / sq0) - u[law]["skill"]))
    print(name, "checked", flush=True)
print("largest differences", worst)
