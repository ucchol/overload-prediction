"""Decision time from trial structure, across every dataset with times, ANALYSIS_PLAN.md Amendment 12.

Log time is centered within participant, so the laws predict which trials take a given person longer. Features are
log2 n, the best value, the gap between the best and second best, and the mean of the other options, values rescaled
to 0 to 1 by each dataset's scale and centered within participant.

Usage
    python -I -m overload.experiments.time_structure --data DEV_CSV --out results/time_structure
    python -I -m overload.experiments.time_structure --data DEV_CSV --simulate {structure,count} --replicates 50 ...
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, roots_laguerre

from ..bootstrap import holm, summarize
from ..data import _match, load_task
from ..engine import provenance, write_json
from ..gaze import load_gaze
from ..snack import load_snack

SETS = ("dev_file", "dev_laptop", "snack", "reutskaja", "cdm", "krajbich2010", "krajbich2011", "folke2016")
SCALE = {"dev_file": (0.0, 100.0), "dev_laptop": (0.0, 100.0), "snack": (-3.0, 3.0), "reutskaja": (-5.0, 5.0),
         "cdm": (1.0, 35.0), "krajbich2010": (0.0, 10.0), "krajbich2011": (0.0, 10.0), "folke2016": (0.0, 10.0)}
FEATURES = ("log2n", "best", "gap", "mean_rest")
LAWS = ("S1", "S2", "S3", "S4")
SEED_OFFSET = 120000
FOLD_SEED = 20261014  # the folds of Amendment 6
FOLDS = 10
_X, _W = roots_laguerre(32)  # exact for the race integrals up to 63 options
_LOGW = np.log(_W) + np.log(_X)


# ----------------------------------------------------------------------------- data
@dataclass
class Block:
    values: np.ndarray  # trials by n, raw values
    choice: np.ndarray
    logt: np.ndarray  # log seconds
    participant: np.ndarray


@dataclass
class TimeSet:
    name: str
    blocks: list[Block]
    participants: int

    def stack(self, attr: str) -> np.ndarray:
        return np.concatenate([getattr(b, attr) for b in self.blocks])


def _from_rows(name: str, values: list, choice: list, seconds: list, person: list) -> TimeSet:
    ids = sorted(set(person))
    index = {p: i for i, p in enumerate(ids)}
    by_n: dict[int, list[int]] = {}
    for r, v in enumerate(values):
        by_n.setdefault(len(v), []).append(r)
    blocks = []
    for n in sorted(by_n):
        rows = by_n[n]
        blocks.append(Block(values=np.array([values[r] for r in rows], float),
                            choice=np.array([choice[r] for r in rows], int),
                            logt=np.log(np.array([seconds[r] for r in rows], float)),
                            participant=np.array([index[person[r]] for r in rows], int)))
    return TimeSet(name, blocks, len(ids))


def load_dev(csv: Path) -> dict[str, TimeSet]:
    df = pd.read_csv(csv, low_memory=False)
    out = {}
    for task, stim in (("file", None), ("laptop", "laptop_trials.json")):
        t = load_task(df, task, stim)
        values, choice, seconds, person = [], [], [], []
        for r, (_, row) in enumerate(df.iterrows()):
            for tr in t.trials:
                i = _match(row[tr.column], {o.lower(): k for k, o in enumerate(tr.options)})
                s = pd.to_numeric(row.get(f"{tr.column}_timer"), errors="coerce")
                if i is None or not np.isfinite(s) or s <= 0:
                    continue
                values.append(tr.U); choice.append(i); seconds.append(float(s)); person.append(r)
        out[f"dev_{task}"] = _from_rows(f"dev_{task}", values, choice, seconds, person)
    return out


def load_snack_times(folder: Path) -> TimeSet:
    d = load_snack(folder)
    blocks = []
    for b in d.blocks:
        ok = np.isfinite(b.rt) & (b.rt > 0)
        blocks.append(Block(b.values[ok], b.choice[ok], np.log(b.rt[ok] / 1000.0), b.participant[ok]))
    return TimeSet("snack", blocks, len(d.participants))


def load_reutskaja(folder: Path) -> TimeSet:
    r = pd.read_stata(Path(folder) / "fixations_data.dta")
    g = r.groupby(["subject", "trialid"], sort=True).first().reset_index()
    values, choice, seconds, person = [], [], [], []
    for _, row in g.iterrows():
        n = int(row["numalt"])
        v = np.array([row[f"rating{i}"] for i in range(1, n + 1)], float)
        c = row["chosendestring"]
        if not np.isfinite(v).all() or not np.isfinite(c) or not (1 <= c <= n) or not row["rt"] > 0:
            continue
        values.append(v); choice.append(int(c) - 1); seconds.append(float(row["rt"]) / 1000.0)
        person.append(int(row["subject"]))
    return _from_rows("reutskaja", values, choice, seconds, person)


def load_cdm_times(folder: Path) -> TimeSet:
    from ..cdm import load_cdm
    d = load_cdm(folder)
    values, choice, seconds, person = [], [], [], []
    for k in range(d.rounds):
        if not d.final_time[k] > 0:
            continue
        values.append(d.values[k, : d.n[k]]); choice.append(int(d.choice[k]))
        seconds.append(float(d.final_time[k])); person.append(int(d.participant[k]))
    return _from_rows("cdm", values, choice, seconds, person)


def load_gaze_times(folder: Path) -> dict[str, TimeSet]:
    out = {}
    for name, gs in load_gaze(folder).items():
        out[name] = TimeSet(name, [Block(gs.values, gs.choice, np.log(gs.seconds), gs.participant)],
                            len(gs.participants))
    return out


def load_all(dev: Path, snack: Path, reutskaja: Path, cdm: Path, gaze: Path) -> dict[str, TimeSet]:
    s = {**load_dev(dev), "snack": load_snack_times(snack), "reutskaja": load_reutskaja(reutskaja),
         "cdm": load_cdm_times(cdm), **load_gaze_times(gaze)}
    for k, (lo, hi) in SCALE.items():
        v = s[k].stack("values") if len({b.values.shape[1] for b in s[k].blocks}) == 1 else \
            np.concatenate([b.values.ravel() for b in s[k].blocks])
        if v.min() < lo - 1e-9 or v.max() > hi + 1e-9:
            raise ValueError(f"{k} has values outside its scale")
    return {k: s[k] for k in SETS}


# ----------------------------------------------------------------------------- features and models
def raw_features(ts: TimeSet) -> np.ndarray:
    lo, hi = SCALE[ts.name]
    rows = []
    for b in ts.blocks:
        v = (np.sort(b.values, axis=1)[:, ::-1] - lo) / (hi - lo)
        n = v.shape[1]
        rows.append(np.column_stack([np.full(len(v), np.log2(n)), v[:, 0], v[:, 0] - v[:, 1], v[:, 1:].mean(axis=1)]))
    return np.concatenate(rows)


def center(x: np.ndarray, person: np.ndarray, P: int) -> np.ndarray:
    """Subtract each participant's mean (unweighted over their trials, so duplicates do not shift it)."""
    x2 = x if x.ndim == 2 else x[:, None]
    cnt = np.bincount(person, minlength=P).astype(float)
    means = np.stack([np.bincount(person, weights=x2[:, j], minlength=P) for j in range(x2.shape[1])], axis=1)
    means /= np.maximum(cnt, 1.0)[:, None]
    out = x2 - means[person]
    return out if x.ndim == 2 else out[:, 0]


def _q(lam: float, v: np.ndarray) -> np.ndarray:
    z = lam * v
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sr_logp_chosen(lam: float, v: np.ndarray, c: np.ndarray) -> np.ndarray:
    """log P(chosen) of sample until return, by Gauss Laguerre quadrature of the threshold two race."""
    q = _q(lam, v)
    L = np.log1p(q[:, :, None] * _X[None, None, :]).sum(axis=1)  # trials by nodes
    qi = q[np.arange(len(c)), c]
    L = L - np.log1p(qi[:, None] * _X[None, :])
    return 2 * np.log(np.maximum(qi, 1e-300)) + logsumexp(L + _LOGW[None, :], axis=1)


def sr_looks(lam: float, v: np.ndarray) -> np.ndarray:
    q = _q(lam, v)
    L = np.log1p(q[:, :, None] * _X[None, None, :]).sum(axis=1)
    return np.exp(logsumexp(L + np.log(_W)[None, :], axis=1))


def cl_logp_chosen(lam: float, v: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = lam * v
    return z[np.arange(len(c)), c] - logsumexp(z, axis=1)


def cl_entropy(lam: float, v: np.ndarray) -> np.ndarray:
    q = _q(lam, v)
    return -(q * np.log2(np.maximum(q, 1e-300))).sum(axis=1)


def fit_lambda(model: str, blocks: list[Block], w: list[np.ndarray], lo: float, hi: float) -> float:
    f = sr_logp_chosen if model == "SR" else cl_logp_chosen
    span = hi - lo

    def nll(t):
        lam = np.exp(t) / span
        return -sum(float(wi @ f(lam, b.values, b.choice)) for b, wi in zip(blocks, w) if wi.sum() > 0)

    r = minimize_scalar(nll, bounds=(-6.0, 6.0), method="bounded", options={"xatol": 1e-7})
    return float(np.exp(r.x) / span)


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    s = np.sqrt(w)
    return np.linalg.lstsq(X * s[:, None], y * s, rcond=None)[0]


def r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    return float(1.0 - np.sum(w * (y - yhat) ** 2) / np.sum(w * y ** 2))


# ----------------------------------------------------------------------------- per dataset
@dataclass
class Prepared:
    ts: TimeSet
    F: np.ndarray  # centered features
    y: np.ndarray  # centered log time
    person: np.ndarray
    fold: np.ndarray  # per trial
    bounds: list[tuple[int, int]]  # rows of each block


def prepare(ts: TimeSet, y_override: np.ndarray | None = None) -> Prepared:
    person = ts.stack("participant")
    y = ts.stack("logt") if y_override is None else y_override
    perm = np.random.default_rng(FOLD_SEED).permutation(ts.participants)
    fold_p = np.zeros(ts.participants, int)
    fold_p[perm] = np.arange(ts.participants) % FOLDS
    bounds, start = [], 0
    for b in ts.blocks:
        bounds.append((start, start + len(b.choice))); start += len(b.choice)
    return Prepared(ts, center(raw_features(ts), person, ts.participants), center(y, person, ts.participants),
                    person, fold_p[person], bounds)


def signals(p: Prepared, train_mask: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centered log SR expected looks and CL entropy, with sensitivities fitted on the training trials."""
    lo, hi = SCALE[p.ts.name]
    wb = [w[a:b] * train_mask[a:b] for a, b in p.bounds]
    lam_sr = fit_lambda("SR", p.ts.blocks, wb, lo, hi)
    lam_cl = fit_lambda("CL", p.ts.blocks, wb, lo, hi)
    looks = np.concatenate([np.log(sr_looks(lam_sr, b.values)) for b in p.ts.blocks])
    ent = np.concatenate([cl_entropy(lam_cl, b.values) for b in p.ts.blocks])
    P = p.ts.participants
    return center(looks, p.person, P), center(ent, p.person, P)


def within(p: Prepared, w: np.ndarray) -> dict[str, float]:
    """Held out R2 of each law over ten folds of participants, and the coefficients of S2 on all trials."""
    pred = {k: np.zeros_like(p.y) for k in LAWS}
    for k in range(FOLDS):
        test = p.fold == k
        if not np.any(test & (w > 0)):
            continue
        tr = ~test
        look, ent = signals(p, tr, w)
        X = {"S1": p.F[:, :1], "S2": p.F, "S3": np.column_stack([p.F, look]), "S4": np.column_stack([p.F, ent])}
        for law in LAWS:
            beta = wls(X[law][tr], p.y[tr], w[tr])
            pred[law][test] = X[law][test] @ beta
    out = {f"R2|{law}": r2(p.y, pred[law], w) for law in LAWS}
    out["T1"] = out["R2|S2"] - out["R2|S1"]
    out["T3"] = out["R2|S3"] - out["R2|S2"]
    out["S4_minus_S2"] = out["R2|S4"] - out["R2|S2"]
    beta = wls(p.F, p.y, w)
    out.update({f"coef|{f}": float(beta[j]) for j, f in enumerate(FEATURES)})
    return out


def transfer(prep: dict[str, Prepared], w: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for held in SETS:
        Xs, ys, ws = [], [], []
        for k in SETS:
            if k == held:
                continue
            Xs.append(prep[k].F); ys.append(prep[k].y); ws.append(w[k] / w[k].sum())
        beta = wls(np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
        out[f"{held}|T2"] = r2(prep[held].y, prep[held].F @ beta, w[held])
        out.update({f"{held}|transfer_coef|{f}": float(beta[j]) for j, f in enumerate(FEATURES)})
    return out


def time_growth(p: Prepared, w: np.ndarray) -> dict[str, float]:
    """Growth of mean (uncentered) log time per doubling, observed, for sets with several set sizes."""
    if len(p.ts.blocks) < 2:
        return {}
    logt = p.ts.stack("logt")
    raw_n = np.concatenate([np.full(len(b.choice), np.log2(b.values.shape[1])) for b in p.ts.blocks])
    xm, ym = np.average(raw_n, weights=w), np.average(logt, weights=w)
    return {"growth_log_time_observed": float(np.sum(w * (raw_n - xm) * (logt - ym)) / np.sum(w * (raw_n - xm) ** 2))}


def measures(prep: dict[str, Prepared], draw: dict[str, np.ndarray] | None) -> dict[str, float]:
    w = {}
    for k, p in prep.items():
        if draw is None:
            w[k] = np.ones_like(p.y)
        else:
            mult = np.bincount(draw[k], minlength=p.ts.participants).astype(float)
            w[k] = mult[p.person]
    out = {}
    for k, p in prep.items():
        out.update({f"{k}|{m}": v for m, v in within(p, w[k]).items()})
        out.update({f"{k}|{m}": v for m, v in time_growth(p, w[k]).items()})
    out.update(transfer(prep, w))
    return out


def draws(prep: dict[str, Prepared], b: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_OFFSET + b)
    return {k: rng.integers(0, prep[k].ts.participants, size=prep[k].ts.participants) for k in SETS}


# ----------------------------------------------------------------------------- simulation
def simulated_times(prep: dict[str, Prepared], kind: str, seed: int = 12) -> dict[str, Prepared]:
    """Log times from S2 with known coefficients ('structure') or from log2 n alone ('count'), plus a person
    speed and noise, on every dataset's real trials."""
    rng = np.random.default_rng(seed)
    coef = {"structure": np.array([0.25, -0.6, -0.8, 0.3]), "count": np.array([0.25, 0.0, 0.0, 0.0])}[kind]
    out = {}
    for k, p in prep.items():
        raw = raw_features(p.ts)
        speed = rng.normal(0, 0.4, p.ts.participants)[p.person]
        y = 1.0 + raw @ coef + speed + rng.normal(0, 0.3, len(raw))
        out[k] = prepare(p.ts, y_override=y)
    return out


# ----------------------------------------------------------------------------- run
def run(sets: dict[str, TimeSet], out: Path, replicates: int, simulate: str | None = None, chunk: int = 10) -> dict:
    t0 = time.time()
    prep = {k: prepare(sets[k]) for k in SETS}
    if simulate:
        prep = simulated_times(prep, simulate)
    point = measures(prep, None)
    part = out / "partial"
    part.mkdir(parents=True, exist_ok=True)
    reps: list[dict] = []
    for start in range(0, replicates, chunk):
        f = part / f"{start:05d}.json"
        if f.exists():
            reps += json.loads(f.read_text()); continue
        block = [{"b": b, **measures(prep, draws(prep, b))} for b in range(start, min(start + chunk, replicates))]
        tmp = f.with_suffix(".tmp"); tmp.write_text(json.dumps(block)); tmp.rename(f)
        reps += block
    summary = {k: summarize(v, [r[k] for r in reps]) if reps else {"point": v} for k, v in point.items()}
    for test in ("T1", "T2", "T3"):
        fam = {k: summary[f"{k}|{test}"]["p"] for k in SETS} if reps else {}
        adj = holm(fam) if fam else {}
        for k in SETS:
            e = summary[f"{k}|{test}"]
            e["p_holm"] = adj.get(k)
            e["supported"] = bool(reps and e["clear"] and e["point"] > 0 and adj[k] < 0.05)
    summary["provenance"] = provenance({"replicates": replicates, "seconds": round(time.time() - t0, 1),
                                        "simulated": simulate})
    write_json(out / "point.json", point)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "summary.json", summary)
    (out / "table_time_structure.md").write_text(table(summary))
    shutil.rmtree(part)
    return summary


def _f(e: dict, d: int = 3) -> str:
    if "percentile" not in e:
        return f"{e['point']:+.{d}f}"
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" + (" *" if e["clear"] else "")


NAMES = {"dev_file": "Our study, files", "dev_laptop": "Our study, laptops", "snack": "Snacks at a glance",
         "reutskaja": "Snacks in three seconds", "cdm": "Sums worked out", "krajbich2010": "Krajbich 2010, two foods",
         "krajbich2011": "Krajbich 2011, three foods", "folke2016": "Folke 2016, three snacks"}


def table(s: dict) -> str:
    L = ["# Decision time from trial structure (Amendment 12)", "",
         "Held out share of variance of log time centered within person.", "",
         "| Dataset | Counting options (S1) | Structure (S2) | T1, structure minus counting | T2, law from the other sets "
         "| T3, search model's looks beyond structure | CL entropy beyond structure |", "|---|---|---|---|---|---|---|"]
    for k in SETS:
        sup = lambda t: " (supported)" if s[f"{k}|{t}"].get("supported") else ""
        L.append(f"| {NAMES[k]} | {_f(s[f'{k}|R2|S1'])} | {_f(s[f'{k}|R2|S2'])} | {_f(s[f'{k}|T1'])}{sup('T1')} | "
                 f"{_f(s[f'{k}|T2'])}{sup('T2')} | {_f(s[f'{k}|T3'])}{sup('T3')} | {_f(s[f'{k}|S4_minus_S2'])} |")
    L += ["", "Coefficients of S2 on all trials (centered log time per unit of each feature, values on a 0 to 1 scale).", "",
          "| Dataset | log2 n | Best value | Gap to runner up | Mean of the rest | Growth of log time per doubling |",
          "|---|---|---|---|---|---|"]
    for k in SETS:
        g = s.get(f"{k}|growth_log_time_observed")
        L.append(f"| {NAMES[k]} | " + " | ".join(_f(s[f'{k}|coef|{f}']) for f in FEATURES) + f" | {_f(g) if g else ''} |")
    L += ["", "Brackets give 95% percentile intervals from the participant bootstrap. A star marks a value whose "
          "percentile and basic intervals both exclude zero. Supported means clearly positive with Holm's correction "
          "across the eight datasets."]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="development data CSV")
    ap.add_argument("--snack", type=Path, default=Path("data/snack"))
    ap.add_argument("--reutskaja", type=Path, default=Path("data/reutskaja"))
    ap.add_argument("--cdm", type=Path, default=Path("data/cdm"))
    ap.add_argument("--gaze", type=Path, default=Path("data/gaze"))
    ap.add_argument("--out", type=Path, default=Path("results/time_structure"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--simulate", choices=("structure", "count"))
    a = ap.parse_args()
    sets = load_all(a.data, a.snack, a.reutskaja, a.cdm, a.gaze)
    s = run(sets, a.out, a.replicates, a.simulate)
    print((a.out / "table_time_structure.md").read_text())
    print("seconds", s["provenance"]["seconds"])


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
