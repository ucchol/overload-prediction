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
    RACES = {"LBA": "lba", "MLBA_LBA": "mlba", "Wald": "wald", "RWald": "rwald",
             "LBA_n": "lba_n", "MLBA_LBA_n": "mlba_n", "Wald_n": "wald_n", "RWald_n": "rwald_n"}
    if name.replace("_cal", "") in RACES:
        base = name.replace("_cal", "")
        kind = RACES[base]
        dl = opts.get("deadline")
        if opts.get("warm") == "repair" and kind.endswith("_n"):
            # deviation of 5 Nov 2026, keep the original cold fit when it passes the nesting check, otherwise the
            # warm start fit from the converged fixed threshold fit with gamma = 0.02, if that passes and is finite
            R0 = tm.Race(kind[:-2]).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
            R = tm.Race(kind).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
            nll_fixed = R0.nll
            extra_fit = {"nll_cold": R.nll, "chosen": "cold"}
            if R.nll > R0.nll + 1.0:
                th0 = np.r_[R0.theta[:-1], np.log(0.02), R0.theta[-1]]
                Rw = tm.Race(kind, th0).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
                extra_fit["nll_warm"] = Rw.nll
                if Rw.nll <= R0.nll + 1.0:
                    R = Rw; extra_fit["chosen"] = "warm"
        elif opts.get("warm") and kind.endswith("_n"):
            # warm start (deviation of 5 Nov 2026), from the converged fixed threshold fit with gamma = 0.02
            R0 = tm.Race(kind[:-2]).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
            th0 = np.r_[R0.theta[:-1], np.log(0.02), R0.theta[-1]]
            R = tm.Race(kind, th0).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
            nll_fixed = R0.nll
            if opts.get("warm") == "best":
                # the original cold fit as well, keeping whichever has the lower negative log likelihood
                Rc = tm.Race(kind).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
                extra_fit = {"nll_cold": Rc.nll, "nll_warm": R.nll}
                if Rc.nll <= R.nll:
                    R = Rc
                extra_fit["chosen"] = "cold" if R is Rc else "warm"
        else:
            R = tm.Race(kind).fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
            nll_fixed = None
        extra = {"theta": R.theta.tolist(), "nll": R.nll}
        if nll_fixed is not None:
            extra["nll_fixed"] = nll_fixed
            extra["converged"] = bool(R.nll <= nll_fixed + 1.0)
            if opts.get("warm") in ("best", "repair"):
                extra.update(extra_fit)
        if opts.get("bits", True) and not name.endswith("_cal"):
            extra["bits_te"] = (R.choice_log2(te) + np.log2(te.n)).tolist()
        def E(t):
            if dl is None:
                return R.expected_logt(t)
            return R.expected_logt_deadline(t, tm.relative_time(t) * dl / np.exp(t.logt))
        b = tm.lsq_center(E(tr)[:, None], tr.logt, tr.pid)
        extra["slope"] = float(b[0])
        et = E(te)
        if opts.get("warm") == "repair" and kind.endswith("_n"):
            extra["finite"] = bool(np.all(np.isfinite(et)) and np.all(np.isfinite(E(tr))))
            extra["converged"] = bool(extra["converged"] and extra["finite"])
        extra["cal_pred"] = et * b[0]  # calibrated version from the same fit
        return et, extra
    if name in ("S2+TLS", "S2+SR", "S2+CLH"):
        if name == "S2+CLH":
            lam = tm.fit_lambda(tm.scores(tr, "CL"), tr.choice)
            q = lambda t: tm.entropy_bits(tm.softmax_masked(lam * t.x))
        else:
            lam = tm.fit_sr_lambda(tr)
            g = tm.tls_expected_looks if name == "S2+TLS" else tm.sr_expected_looks
            q = lambda t: np.log(g(lam, t))
        F = lambda t: np.column_stack([tm.structure(t), q(t)])
        b = tm.lsq_center(F(tr), tr.logt, tr.pid); return F(te) @ b, {"lambda": lam, "coef": b.tolist()}
    if name == "S2+TLS_tfit":
        best = None
        for lam in np.geomspace(0.5, 200, 30):
            F = np.column_stack([tm.structure(tr), np.log(tm.tls_expected_looks(lam, tr))])
            b = tm.lsq_center(F, tr.logt, tr.pid)
            r = tm.r2(tr.logt, F @ b, tr.pid)
            if best is None or r > best[0]:
                best = (r, lam, b)
        _, lam, b = best
        Fte = np.column_stack([tm.structure(te), np.log(tm.tls_expected_looks(lam, te))])
        return Fte @ b, {"lambda": float(lam), "coef": b.tolist()}
    if name == "S2+Wald_n":
        dl = opts.get("deadline")
        R = tm.Race("wald_n").fit(tr, starts=opts.get("starts", 3), eps=opts.get("eps", 0.0), deadline_s=dl)
        def E(t):
            if dl is None:
                return R.expected_logt(t)
            return R.expected_logt_deadline(t, tm.relative_time(t) * dl / np.exp(t.logt))
        F = lambda t: np.column_stack([tm.structure(t), E(t)])
        b = tm.lsq_center(F(tr), tr.logt, tr.pid)
        return F(te) @ b, {"theta": R.theta.tolist(), "coef": b.tolist()}
    if name == "TLS_tfit":
        grid = np.geomspace(0.5, 200, 30)
        best = None
        for lam in grid:
            L = np.log(tm.tls_expected_looks(lam, tr))[:, None]
            b = tm.lsq_center(L, tr.logt, tr.pid)
            r = tm.r2(tr.logt, L @ b, tr.pid)
            if best is None or r > best[0]:
                best = (r, lam, b)
        _, lam, b = best
        return np.log(tm.tls_expected_looks(lam, te))[:, None] @ b, {"lambda": float(lam), "coef": b.tolist()}
    if name == "Ceiling2":
        from sklearn.ensemble import HistGradientBoostingRegressor
        def Fc(t):
            F = np.nan_to_num(tm.rich_features(t), nan=-1.0)
            return np.column_stack([F, np.column_stack([tm.center(F[:, j], t.pid) for j in range(F.shape[1])])])
        Ftr, Fte, ytr = Fc(tr), Fc(te), tm.center(tr.logt, tr.pid)
        # inner choice of complexity on two folds of the training participants
        people = np.unique(tr.pid); half = np.isin(tr.pid, people[::2])
        best = None
        for it, lr, leaf in ((100, 0.05, 80), (200, 0.05, 80), (300, 0.03, 120), (150, 0.1, 200)):
            sc = 0.0
            for a in (half, ~half):
                g = HistGradientBoostingRegressor(random_state=0, max_iter=it, learning_rate=lr, min_samples_leaf=leaf)
                g.fit(Ftr[a], ytr[a]); sc += tm.r2(tr.logt[~a], g.predict(Ftr[~a]), tr.pid[~a])
            if best is None or sc > best[0]:
                best = (sc, it, lr, leaf)
        _, it, lr, leaf = best
        g = HistGradientBoostingRegressor(random_state=0, max_iter=it, learning_rate=lr, min_samples_leaf=leaf).fit(Ftr, ytr)
        return g.predict(Fte), {"hp": [it, lr, leaf]}
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
                yhat[te] = yk
                if "cal_pred" in e:
                    yc = preds.setdefault(f"{m}_cal", np.full(len(D.n), np.nan)); yc[te] = e.pop("cal_pred")
                ex.append(e)
            preds[m] = yhat; extras[m] = ex
            log(f"{name} {m} R2 {tm.r2(D.logt, yhat, D.pid):+.4f} ({time.time() - t0:.0f}s)")
            if f"{m}_cal" in preds:
                log(f"{name} {m}_cal R2 {tm.r2(D.logt, preds[f'{m}_cal'], D.pid):+.4f} slopes {[round(x['slope'], 3) for x in ex]}")
        E = D
    else:
        D, V = T.take(~vault[T.pid]), T.take(vault[T.pid])
        for m in models:
            t0 = time.time()
            yhat, e = fit_predict(m, D, V, opts)
            if "cal_pred" in e:
                preds[f"{m}_cal"] = e.pop("cal_pred")
            preds[m] = yhat; extras[m] = [e]
            log(f"{name} {m} vault R2 {tm.r2(V.logt, yhat, V.pid):+.4f} ({time.time() - t0:.0f}s)")
        E = V
    out["r2"] = {m: tm.r2(E.logt, preds[m], E.pid) for m in preds}
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
    ap.add_argument("--warm", nargs="?", const="warm", default="",
                    help="warm start the rising threshold races, or 'best' to keep the better of the cold and warm fits "
                         "(deviation of 5 Nov 2026)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    np.seterr(all="ignore")
    a.out.mkdir(parents=True, exist_ok=True)
    S = load_all(a.dev, Path("data/snack"), Path("data/reutskaja"), Path("data/cdm"), Path("data/gaze"))
    logf = open(a.out / "log.txt", "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"round {a.round} stage {a.stage} models {a.models} eps {a.eps} warm {a.warm}")
    for name in a.sets:
        T = tm.from_timeset(S[name], SCALE[name])
        opts = {"eps": a.eps, "deadline": 3.0 if name == "reutskaja" else None, "warm": a.warm}
        res, E, preds = run_set(name, T, a.models, a.stage, opts, log)
        old = a.out / f"{name}_pred.npz"
        if old.exists():  # keep predictions of earlier models of the same round
            prev = dict(np.load(old)); preds = {**{k: v for k, v in prev.items() if k not in ("logt", "pid", "n", "choice")}, **preds}
        np.savez_compressed(a.out / f"{name}_pred.npz", logt=E.logt, pid=E.pid, n=E.n, choice=E.choice,
                            **{m: v for m, v in preds.items()})
        json.dump(res, open(a.out / f"{name}_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
