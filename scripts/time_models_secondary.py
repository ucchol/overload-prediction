"""Secondary measures of the decision time models on the vault (docs/time_models_plan.md, Outcome and score).

(a) Within person rank correlation of predicted and observed time, from the stored vault predictions.
(b) Forecasting an unseen set size, in sets with three or more set sizes. Each set size is left out in turn, every
    model is fitted on the non vault participants' trials at the other sizes and predicts all vault trials. For each
    vault person, D is the mean log time at the left out size minus the mean log time at the other sizes. The error is
    D predicted minus D observed. Reported, the root mean squared error over people and left out sizes, its value for a
    forecast of no difference, and the skill 1 - MSE / MSE of no difference.
(c) Choices. Bits per choice gained over a uniform guess on vault trials, for CL and sample until return (choice
    sensitivity fitted to the non vault choices) and for every race (stored with the vault fits).
Intervals are 95 percent participant bootstrap percentiles (B = 1000, numpy.random.default_rng(130500 + b)).
Usage, python3 -I scripts/time_models_secondary.py --dev DEV_CSV --vault VAULT_DIR --out OUT [--skip_b]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from overload.experiments.time_structure import SCALE, load_all  # noqa: E402
from overload import time_models as tm  # noqa: E402
from time_models_run import fit_predict, vault_mask  # noqa: E402

SETS = ["dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016"]
MODELS = ["S1", "S2", "CL_entropy", "DbS_entropy", "MLBA_entropy", "SR_looks", "TLS_looks", "LBA", "MLBA_LBA", "Wald",
          "RWald", "SR_timing", "Ceiling", "S2+TLS", "S2+SR", "S2+CLH", "TLS_tfit", "Ceiling2", "LBA_n", "Wald_n",
          "RWald_n", "MLBA_LBA_n", "S2+TLS_tfit", "S2+Wald_n"]
B = 1000


def boot_people(P):
    return [np.bincount(np.random.default_rng(130500 + b).integers(0, P, P), minlength=P) for b in range(B)]


def rank_corr(y, yhat, pid):
    P = pid.max() + 1
    r = np.zeros(P)
    for p in range(P):
        m = pid == p
        if np.ptp(yhat[m]) > 1e-12 and np.ptp(y[m]) > 0:
            r[p] = spearmanr(yhat[m], y[m]).statistic
    return r  # zero where the prediction is constant within the person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--vault", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip_b", action="store_true")
    ap.add_argument("--sets", nargs="*", default=SETS)
    a = ap.parse_args()
    np.seterr(all="ignore")
    a.out.mkdir(parents=True, exist_ok=True)
    S = load_all(a.dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
    res = {}
    for name in a.sets:
        T = tm.from_timeset(S[name], SCALE[name])
        P0 = int(T.pid.max()) + 1
        vault = vault_mask(P0)
        D, V = T.take(~vault[T.pid]), T.take(vault[T.pid])
        _, vp = np.unique(V.pid, return_inverse=True)
        V = tm.Trials(V.x, V.n, V.choice, V.logt, vp)
        P = int(vp.max()) + 1
        W = np.array(boot_people(P))
        z = np.load(a.vault / f"{name}_pred.npz")
        assert np.allclose(z["logt"], V.logt)
        rec = {"rank_corr": {}, "bits": {}, "unseen_size": {}}
        # (a)
        for m in [k for k in z.files if k not in ("logt", "pid", "n", "choice")]:
            r = rank_corr(V.logt, z[m], vp)
            bt = (W @ r) / P
            rec["rank_corr"][m] = {"mean": float(r.mean()), "ci": np.percentile(bt, [2.5, 97.5]).tolist()}
        # (c)
        cnt = np.bincount(vp, minlength=P)
        def summarize(bits):
            per = np.bincount(vp, weights=bits, minlength=P)
            bt = (W @ per) / (W @ cnt)
            return per, {"bits_per_choice": float(per.sum() / cnt.sum()), "ci": np.percentile(bt, [2.5, 97.5]).tolist()}
        lam = tm.fit_lambda(D.x, D.choice)
        p = tm.softmax_masked(lam * V.x)
        cl_bits = np.log2(p[np.arange(len(V.n)), V.choice]) + np.log2(V.n)
        cl_per, rec["bits"]["CL"] = summarize(cl_bits)
        lam_sr = tm.fit_sr_lambda(D)
        _, rec["bits"]["SR"] = summarize(tm.sr_choice_log2(lam_sr, V) + np.log2(V.n))
        rec["bits"]["SR"]["note"] = "two look satisficing makes the same choices as sample until return"
        vs = json.load(open(a.vault / f"{name}_summary.json"))["extras"]
        for m, ex in vs.items():
            if "bits_te" in ex[0]:
                b = np.array(ex[0]["bits_te"])
                per, s = summarize(b)
                d = (W @ (per - cl_per)) / (W @ cnt)
                s["minus_CL"] = float((per - cl_per).sum() / cnt.sum())
                s["minus_CL_ci"] = np.percentile(d, [2.5, 97.5]).tolist()
                rec["bits"][m] = s
        print(name, "rank and bits done", flush=True)
        # (b)
        sizes = sorted(set(T.n.tolist()))
        if len(sizes) >= 3 and not a.skip_b:
            opts = {"eps": 0.0, "deadline": 3.0 if name == "reutskaja" else None, "bits": False}
            err = {}
            obs_all = []
            for s in sizes:
                tr = D.take(D.n != s)
                inn = V.n == s
                def diff(y):
                    at = np.bincount(vp[inn], weights=y[inn], minlength=P) / np.maximum(np.bincount(vp[inn], minlength=P), 1)
                    ot = np.bincount(vp[~inn], weights=y[~inn], minlength=P) / np.maximum(np.bincount(vp[~inn], minlength=P), 1)
                    ok = (np.bincount(vp[inn], minlength=P) > 0) & (np.bincount(vp[~inn], minlength=P) > 0)
                    return at - ot, ok
                dobs, ok = diff(V.logt)
                obs_all.append((s, dobs, ok))
                for m in MODELS:
                    t0 = time.time()
                    yhat, e = fit_predict(m, tr, V, opts)
                    outs = {m: yhat}
                    if "cal_pred" in e:
                        outs[m + "_cal"] = e["cal_pred"]
                    for mm, yy in outs.items():
                        dp, _ = diff(yy)
                        err.setdefault(mm, []).append((s, dp - dobs, ok, dp))
                    print(name, "size", s, m, f"{time.time() - t0:.0f}s", flush=True)
            # people with trials at the size and elsewhere, every left out size pooled
            sq0 = np.zeros(P); k0 = np.zeros(P)
            for s, dobs, ok in obs_all:
                sq0 += np.where(ok, dobs ** 2, 0); k0 += ok
            mse0 = sq0.sum() / k0.sum()
            out = {"sizes": sizes, "rmse_no_difference": float(np.sqrt(mse0)),
                   "observed_mean_difference": {str(s): float(dobs[ok].mean()) for s, dobs, ok in obs_all}}
            for mm, lst in err.items():
                sq = np.zeros(P)
                for s, e_, ok, dp in lst:
                    sq += np.where(ok, e_ ** 2, 0)
                bt = 1 - (W @ sq) / (W @ sq0)
                out[mm] = {"rmse": float(np.sqrt(sq.sum() / k0.sum())), "skill": float(1 - sq.sum() / sq0.sum()),
                           "skill_ci": np.percentile(bt, [2.5, 97.5]).tolist(),
                           "predicted_mean_difference": {str(s): float(dp[ok].mean()) for s, e_, ok, dp in lst}}
            rec["unseen_size"] = out
        res[name] = rec
        json.dump(res, open(a.out / "secondary.json", "w"), indent=1, default=float)
    print("done", flush=True)


if __name__ == "__main__":
    main()
