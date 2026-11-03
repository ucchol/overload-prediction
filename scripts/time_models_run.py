"""Decision time from choice models (docs/time_models_plan.md), development folds or the vault.

Usage
    python3 -I scripts/time_models_run.py --dev DEV_CSV --stage dev --round 1 --sets snack reutskaja --out OUT
    python3 -I scripts/time_models_run.py --dev DEV_CSV --stage vault --round final --out OUT

Development stage. Ten folds of the non vault participants, every model fitted on nine and predicting the tenth.
Vault stage. Every model fitted on all non vault participants and predicting the vault participants.
Per trial predictions are saved, so every score can be recomputed and bootstrapped without refitting.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from overload.experiments.time_structure import SCALE, SETS, load_all  # noqa: E402
from overload import time_models as tm  # noqa: E402

VAULT_SEED, FOLD_SEED, FOLDS = 20261103, 20261014, 10


def vault_mask(P: int) -> np.ndarray:
    perm = np.random.default_rng(VAULT_SEED).permutation(P)
    v = np.zeros(P, bool)
    v[perm[: int(np.ceil(0.3 * P))]] = True
    return v


def folds_of(people: np.ndarray) -> dict[int, int]:
    perm = np.random.default_rng(FOLD_SEED).permutation(people.size)
    return {int(people[q]): k % FOLDS for k, q in enumerate(perm)}


MODELS_R1 = ["S1", "S2", "CL_entropy", "DbS_entropy", "MLBA_entropy", "SR_looks", "TLS_looks",
             "LBA", "MLBA_LBA", "Wald", "RWald", "SR_timing", "Ceiling"]


def fit_predict(name: str, tr: tm.Trials, te: tm.Trials, opts: dict) -> tuple[np.ndarray, dict]:
    """Fit model `name` on tr, return predicted log time for te (uncentered) and extras."""
    extra = {}
    if name == "S1":
        F = lambda t: np.log2(t.n)[:, None]
        b = tm.lsq_center(F(tr), tr.logt, tr.pid); return F(te) @ b, {"coef": b.tolist()}
    if name == "S2":
        b = tm.lsq_center(tm.structure(tr), tr.logt, tr.pid); return tm.structure(te) @ b, {"coef": b.tolist()}
    if name.endswith("_entropy"):
        fam = {"CL_entropy": "CL", "DbS_entropy": "DbS_avg", "MLBA_entropy": "MLBA_avg"}[name]
        lam = tm.fit_lambda(tm.scores(tr, fam), tr.choice)
        H = lambda t: tm.entropy_bits(tm.softmax_masked(lam * tm.scores(t, fam)))[:, None]
        b = tm.lsq_center(H(tr), tr.logt, tr.pid); return H(te) @ b, {"lambda": lam, "coef": b.tolist()}
    if name in ("SR_looks", "TLS_looks"):
        lam = tm.fit_sr_lambda(tr)
        g = tm.sr_expected_looks if name == "SR_looks" else tm.tls_expected_looks
        L = lambda t: np.log(g(lam, t))[:, None]
        b = tm.lsq_center(L(tr), tr.logt, tr.pid); return L(te) @ b, {"lambda": lam, "coef": b.tolist()}
    if name in ("LBA", "MLBA_LBA", "Wald", "RWald"):
        kind = {"LBA": "lba", "MLBA_LBA": "mlba", "Wald": "wald", "RWald": "rwald"}[name]
        R = tm.Race(kind).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0))
        extra = {"theta": R.theta.tolist(), "nll": R.nll}
        if opts.get("bits", True):
            extra["bits_te"] = (R.choice_log2(te) + np.log2(te.n)).tolist()
        return R.expected_logt(te), extra
    if name == "SR_timing":
        m = tm.SRTiming().fit(tr)
        return m.expected_logt(te), {"lambda": m.lam, "theta": m.theta.tolist()}
    if name == "Ceiling":
        from sklearn.ensemble import HistGradientBoostingRegressor
        Ftr, Fte = tm.rich_features(tr), tm.rich_features(te)
        g = HistGradientBoostingRegressor(random_state=0, max_iter=300, learning_rate=0.05, min_samples_leaf=40)
        g.fit(np.nan_to_num(Ftr, nan=-1.0), tm.center(tr.logt, tr.pid))
        return g.predict(np.nan_to_num(Fte, nan=-1.0)), {}
    raise ValueError(name)


def run_set(name: str, T: tm.Trials, models: list[str], stage: str, opts: dict, log) -> dict:
    P = int(T.pid.max()) + 1
    vault = vault_mask(P)
    out = {"set": name, "trials_dev": int((~vault[T.pid]).sum()), "trials_vault": int(vault[T.pid].sum())}
    preds, extras = {}, {}
    if stage == "dev":
        D = T.take(~vault[T.pid])
        fold = folds_of(np.unique(D.pid))
        f = np.array([fold[int(p)] for p in D.pid])
        for m in models:
            t0 = time.time(); yhat = np.full(len(D.n), np.nan); ex = []
            for k in range(FOLDS):
                te = f == k
                if not te.any():
                    continue
                yk, e = fit_predict(m, D.take(~te), D.take(te), opts)
                yhat[te] = yk; ex.append(e)
            preds[m] = yhat; extras[m] = ex
            log(f"{name} {m} R2 {tm.r2(D.logt, yhat, D.pid):+.4f} ({time.time() - t0:.0f}s)")
        E = D
    else:
        D, V = T.take(~vault[T.pid]), T.take(vault[T.pid])
        for m in models:
            t0 = time.time()
            yhat, e = fit_predict(m, D, V, opts)
            preds[m] = yhat; extras[m] = [e]
            log(f"{name} {m} vault R2 {tm.r2(V.logt, yhat, V.pid):+.4f} ({time.time() - t0:.0f}s)")
        E = V
    out["r2"] = {m: tm.r2(E.logt, preds[m], E.pid) for m in models}
    out["extras"] = extras
    return out, E, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--stage", choices=["dev", "vault"], required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--models", nargs="*", default=MODELS_R1)
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    a.out.mkdir(parents=True, exist_ok=True)
    S = load_all(a.dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
    logf = open(a.out / "log.txt", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"round {a.round} stage {a.stage} models {a.models} eps {a.eps}")
    for name in a.sets:
        T = tm.from_timeset(S[name], SCALE[name])
        res, E, preds = run_set(name, T, a.models, a.stage, {"eps": a.eps}, log)
        np.savez_compressed(a.out / f"{name}_pred.npz", logt=E.logt, pid=E.pid, n=E.n, choice=E.choice,
                            **{m: v for m, v in preds.items()})
        json.dump(res, open(a.out / f"{name}_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
