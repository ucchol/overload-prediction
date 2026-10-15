"""Exploratory. Does an absolute standard let the search model time single decisions? (snack data)

The confirmatory tests of Amendment 5 left one clear gap. Sample until return chooses by values relative to each other,
so it cannot say why a display with an excellent snack is decided quickly, and the strongest sign of a slow decision
was a low rating of the best snack on display. Two look satisficing (``overload.search``) adds an absolute standard
without a new free parameter. A look registers with a probability that rises with the snack's rating relative to the
top of the scale, and the first snack whose looks register twice is chosen. Its choices are those of sample until
return at the same sensitivity, so only time and search can show the difference.

These data suggested the model, so everything here is exploratory. Its test is frozen in advance on new data.

E1  Time, held out. log response time = log(t0 + tau effort), fitted on three set sizes, applied to the fourth, as in
    test S2 of Amendment 5. Effort is n for CL and the expected number of looks for sample until return and for two
    look satisficing.
E2  Slow decisions. Within participant and set size, the rank correlation between a forecast and the response time,
    averaged over set sizes and participants, as in S3. Forecasts are the two search models' expected looks and the
    rating of the best snack on display.
E3  Search. The expected number of distinct snacks looked at, against the observed number, and its growth with set
    size (elasticity).
A sensitivity reports the variant in which attention ignores value and only the registration of a look depends on it.

Resampling and sensitivities follow Amendment 5 (seeds 40000 + b, leave one set size out, every fit redone in every
replicate), so every replicate pairs with the corresponding replicate of Amendment 5.

Usage
    python -m overload.experiments.two_look_snack --snack data/snack --out results/two_look_snack --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..snack import load_snack
from ..snack_search import LEVELS, expected_looks, fit_search, two_look_levels
from .search_confirmatory import draw, elasticity, fit_time_law, prepare, spearman, wmean

V_TOP = float(LEVELS[-1])
MODELS = ("CL", "SR", "TLS")
FORECASTS = ("looks_TLS", "looks_SR", "top_rating")
_STATE: dict = {}


def _uniform_attention(lam: float, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sensitivity variant. Attention spread evenly, a look registers with exp(lam (v - v_top)). Looks and distinct."""
    from ..search import race_with_registration

    looks, distinct = np.zeros(counts.shape[0]), np.zeros(counts.shape[0])
    for t, c in enumerate(counts):
        v = np.repeat(LEVELS, c.astype(int))
        _, looks[t], distinct[t] = race_with_registration(np.full(v.size, 1.0 / v.size), np.exp(lam * (v - V_TOP)))
    return looks, distinct


def run_replicate(b: int) -> dict:
    data, prep = _STATE["data"], _STATE["prep"]
    blocks = prep["blocks"]
    K = len(blocks)
    d = draw(prep["participants"], b)
    w = data.weights(d)
    mult = np.ones(prep["participants"]) if d is None else np.bincount(d, minlength=prep["participants"]).astype(float)
    rec: dict = {"b": b}

    lam, sr_looks, tls_looks, tls_distinct = [], [], [], []
    for k in range(K):
        lk = fit_search([(blocks[j]["levels"], w[j]) for j in range(K) if j != k])
        lam.append(lk)
        sr_looks.append([expected_looks(lk, blocks[j]["levels"]) for j in range(K)])
        two = [two_look_levels(lk, blocks[j]["levels"], V_TOP) for j in range(K)]
        tls_looks.append([x[0] for x in two])
        tls_distinct.append([x[1] for x in two])
    rec["lambda"] = lam

    # E1, time held out
    effort = {
        "CL": lambda k, j: np.full(blocks[j]["values"].shape[0], float(blocks[j]["n"])),
        "SR": lambda k, j: sr_looks[k][j],
        "TLS": lambda k, j: tls_looks[k][j],
    }
    rec["E1"] = {}
    for m in MODELS:
        errors, params = [], []
        for k in range(K):
            train = [j for j in range(K) if j != k]
            e = np.concatenate([effort[m](k, j)[blocks[j]["timed"]] for j in train])
            y = np.concatenate([blocks[j]["log_seconds"][blocks[j]["timed"]] for j in train])
            ww = np.concatenate([w[j][blocks[j]["timed"]] for j in train])
            t0, tau = fit_time_law(e, y, ww)
            th = blocks[k]["timed"]
            pred = np.log(t0 + tau * effort[m](k, k)[th])
            errors.append(wmean(np.abs(blocks[k]["log_seconds"][th] - pred), w[k][th]))
            params.append((t0, tau))
        rec["E1"][m] = {"by_size": errors, "error": float(np.mean(errors)), "params": params}

    # E2, slow decisions
    forecast = {
        k: {
            "looks_TLS": tls_looks[k][k],
            "looks_SR": sr_looks[k][k],
            "top_rating": blocks[k]["features"]["top_rating"],
        }
        for k in range(K)
    }
    per: dict[str, dict[int, list[float]]] = {name: {} for name in FORECASTS}
    for k, p, rows in prep["groups"]:
        if mult[p] == 0:
            continue
        y = blocks[k]["seconds"][rows]
        for name in FORECASTS:
            r = spearman(forecast[k][name][rows], y)
            if np.isfinite(r):
                per[name].setdefault(p, []).append(r)
    rec["E2"] = {}
    for name in FORECASTS:
        ps = sorted(per[name])
        r = np.array([np.mean(per[name][p]) for p in ps])
        rec["E2"][name] = float(mult[ps] @ r / mult[ps].sum())

    # E3, search
    observed = [wmean(blocks[k]["distinct_seen"], w[k]) for k in range(K)]
    predicted = [wmean(tls_distinct[k][k], w[k]) for k in range(K)]
    predicted_sr = [wmean(sr_looks[k][k] - 1.0, w[k]) for k in range(K)]
    rec["E3"] = {
        "observed": observed,
        "predicted": predicted,
        "predicted_SR": predicted_sr,
        "observed_elasticity": elasticity(observed),
        "predicted_elasticity": elasticity(predicted),
    }

    if b < 0:  # sensitivity, uniform attention, point estimates only
        u_looks = [[_uniform_attention(lam[k], blocks[j]["levels"].counts)[0] for j in range(K)] for k in range(K)]
        u_dist = [_uniform_attention(lam[k], blocks[k]["levels"].counts)[1] for k in range(K)]
        errors = []
        for k in range(K):
            train = [j for j in range(K) if j != k]
            e = np.concatenate([u_looks[k][j][blocks[j]["timed"]] for j in train])
            y = np.concatenate([blocks[j]["log_seconds"][blocks[j]["timed"]] for j in train])
            t0, tau = fit_time_law(e, y, np.concatenate([w[j][blocks[j]["timed"]] for j in train]))
            th = blocks[k]["timed"]
            errors.append(wmean(np.abs(blocks[k]["log_seconds"][th] - np.log(t0 + tau * u_looks[k][k][th])), w[k][th]))
        corr: dict[int, list[float]] = {}
        for k, p, rows in prep["groups"]:
            r = spearman(u_looks[k][k][rows], blocks[k]["seconds"][rows])
            if np.isfinite(r):
                corr.setdefault(p, []).append(r)
        dist = [wmean(u_dist[k], w[k]) for k in range(K)]
        rec["uniform_attention"] = {
            "error": float(np.mean(errors)),
            "correlation": float(np.mean([np.mean(v) for v in corr.values()])),
            "distinct": dist,
            "elasticity": elasticity(dist),
        }
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(b) for b in bs]


def summarize_all(point: dict, reps: list[dict]) -> dict:
    R = reps
    out: dict = {"lambda": point["lambda"], "E1": {"error": {}, "TLS_minus": {}}, "E2": {"correlation": {}}}
    for m in MODELS:
        out["E1"]["error"][m] = summarize(point["E1"][m]["error"], [r["E1"][m]["error"] for r in R])
        out["E1"]["error"][m]["by_size"] = point["E1"][m]["by_size"]
        out["E1"]["error"][m]["params"] = point["E1"][m]["params"]
    for m in ("CL", "SR"):
        out["E1"]["TLS_minus"][m] = summarize(
            point["E1"]["TLS"]["error"] - point["E1"][m]["error"],
            [r["E1"]["TLS"]["error"] - r["E1"][m]["error"] for r in R],
        )
    for name in FORECASTS:
        out["E2"]["correlation"][name] = summarize(point["E2"][name], [r["E2"][name] for r in R])
    out["E2"]["TLS_minus_SR"] = summarize(
        point["E2"]["looks_TLS"] - point["E2"]["looks_SR"], [r["E2"]["looks_TLS"] - r["E2"]["looks_SR"] for r in R]
    )
    # how much of the best rating's link with time the model carries (correlations of opposite sign, so a sum)
    out["E2"]["TLS_plus_top_rating"] = summarize(
        point["E2"]["looks_TLS"] + point["E2"]["top_rating"],
        [r["E2"]["looks_TLS"] + r["E2"]["top_rating"] for r in R],
    )
    E3 = point["E3"]
    out["E3"] = {
        "observed": E3["observed"],
        "predicted": E3["predicted"],
        "predicted_SR": E3["predicted_SR"],
        "observed_over_predicted": [o / p for o, p in zip(E3["observed"], E3["predicted"], strict=True)],
        "observed_interval": np.percentile([r["E3"]["observed"] for r in R], [2.5, 97.5], axis=0).tolist(),
        "predicted_interval": np.percentile([r["E3"]["predicted"] for r in R], [2.5, 97.5], axis=0).tolist(),
        "observed_elasticity": summarize(E3["observed_elasticity"], [r["E3"]["observed_elasticity"] for r in R]),
        "predicted_elasticity": summarize(E3["predicted_elasticity"], [r["E3"]["predicted_elasticity"] for r in R]),
        "predicted_minus_observed_elasticity": summarize(
            E3["predicted_elasticity"] - E3["observed_elasticity"],
            [r["E3"]["predicted_elasticity"] - r["E3"]["observed_elasticity"] for r in R],
        ),
    }
    out["uniform_attention"] = point["uniform_attention"]
    return out


def run(folder: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    data = load_snack(folder)
    _STATE.update(data=data, prep=prepare(data))
    t0 = time.time()
    point = run_replicate(-1)
    order = list(range(replicates))
    if jobs > 1 and replicates:
        size = max(1, replicates // (4 * jobs))
        groups = [order[i : i + size] for i in range(0, len(order), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            reps = [r for chunk in pool.map(_worker, groups) for r in chunk]
    else:
        reps = _worker(order)
    reps.sort(key=lambda r: r["b"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "point.json", point)
    summary = summarize_all(point, reps)
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
    )
    write_json(out / "summary.json", summary)
    (out / "table_two_look_snack.md").write_text(table(summary))
    return summary


def _f(e: dict, digits: int = 3, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    lo, hi = e["percentile"]
    star = " *" if e.get("clear") else ""
    return f"{fmt.format(e['point'])} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def table(s: dict) -> str:
    L = ["# Two look satisficing on the snack data (exploratory)", ""]
    L += ["## E1, time held out (mean absolute error of log response time)", ""]
    L += ["| Model | Error | 9 | 16 | 25 | 36 |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        e = s["E1"]["error"][m]
        L.append(f"| {m} | {_f(e)} | " + " | ".join(f"{x:.3f}" for x in e["by_size"]) + " |")
    L += [
        "",
        f"- Two look satisficing minus CL {_f(s['E1']['TLS_minus']['CL'], 4, True)}, "
        f"minus sample until return {_f(s['E1']['TLS_minus']['SR'], 4, True)}.",
        "",
        "## E2, slow decisions (mean within participant rank correlation with response time)",
        "",
        "| Forecast | Correlation |",
        "|---|---|",
    ]
    for name, e in s["E2"]["correlation"].items():
        L.append(f"| {name} | {_f(e, 3, True)} |")
    L += [
        "",
        f"- Two look satisficing minus sample until return {_f(s['E2']['TLS_minus_SR'], 3, True)}. "
        f"Two look satisficing plus best rating {_f(s['E2']['TLS_plus_top_rating'], 3, True)} "
        "(zero would mean the model carries all of the best rating's link with time).",
        "",
        "## E3, distinct snacks looked at",
        "",
        "| | 9 | 16 | 25 | 36 | Elasticity |",
        "|---|---|---|---|---|---|",
        "| People | "
        + " | ".join(f"{x:.2f}" for x in s["E3"]["observed"])
        + f" | {_f(s['E3']['observed_elasticity'], 2)} |",
        "| Two look satisficing | "
        + " | ".join(f"{x:.2f}" for x in s["E3"]["predicted"])
        + f" | {_f(s['E3']['predicted_elasticity'], 2)} |",
        "| Sample until return | " + " | ".join(f"{x:.2f}" for x in s["E3"]["predicted_SR"]) + " | |",
        "",
        f"- Predicted minus observed elasticity {_f(s['E3']['predicted_minus_observed_elasticity'], 2, True)}. "
        "People over model, " + ", ".join(f"{x:.2f}" for x in s["E3"]["observed_over_predicted"]) + ".",
        "",
        "## Sensitivity, attention that ignores value (point estimates)",
        "",
        f"- Time error {s['uniform_attention']['error']:.3f}, rank correlation with time "
        f"{s['uniform_attention']['correlation']:+.3f}, distinct snacks "
        + ", ".join(f"{x:.2f}" for x in s["uniform_attention"]["distinct"])
        + f", elasticity {s['uniform_attention']['elasticity']:.2f}.",
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile "
        "and basic intervals both exclude zero. Exploratory, these data suggested the model.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snack", type=Path, default=Path("data/snack"))
    ap.add_argument("--out", type=Path, default=Path("results/two_look_snack"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.snack, args.out, args.replicates, args.jobs)
    print((args.out / "table_two_look_snack.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
