"""Forecasting choice overload, ANALYSIS_PLAN.md Amendment 8.

Choice overload means that more options make choices worse. Here a choice is worse when it misses the best option on
display, and overload is the growth of misses with set size. Every forecast comes from fits to the other set sizes
of the same task, so a model must foresee how much worse choices get in a set size it has never seen, and, within a
set size, on which trials people will miss.

On the snack data, where these tests were built, every fixed model foresaw misses rising with set size while people's
misses stayed flat, and CL with sensitivity rising with set size at the rate of H1 (compensation) foresaw them. The
frozen claim is that compensation needs cheap evaluation. Where every value has to be worked out (Caplin, Dean and
Martin) or decided within three seconds (Reutskaja and colleagues), misses should grow as the fixed models foresee.

O1  Growth. People's growth of misses (slope on log2 of the set size) against each model's forecast. Overload when
    people's misses grow clearly more than compensation foresees, compensation when they grow clearly less than CL
    foresees, partial compensation when both hold.
O2  Premonition. Within each set size, whether sample until return's held out probability of a miss picks out the
    trials with a miss better than the count of options near the best (area under the curve).

Usage
    python -m overload.experiments.overload_premonition --cdm data/cdm --reutskaja data/reutskaja \
        --out results/overload_premonition --jobs 2
    python -m overload.experiments.overload_premonition --snack data/snack --out results/overload_snack   (exploratory)
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import summarize
from ..cdm import load_cdm
from ..overload_forecast import MODELS, Panel, cdm_panels, load_reutskaja, panel_measures

SEED_OFFSET = 80000
FEATURES = ("near_best", "minus_gap")
_STATE: dict = {}


def snack_panel(folder: Path) -> tuple[Panel, int]:
    from ..snack import load_snack

    d = load_snack(folder)
    m = max(b.n for b in d.blocks)
    vals, n, ch, part = [], [], [], []
    for b in d.blocks:
        v = np.full((b.trials, m), np.nan)
        v[:, : b.n] = b.values
        vals.append(v)
        n.append(np.full(b.trials, b.n))
        ch.append(b.choice)
        part.append(b.participant)
    panel = Panel("snack", np.vstack(vals), np.concatenate(n), np.concatenate(ch), np.concatenate(part), shift=4.0)
    return panel, len(d.participants)


def groups(args) -> dict[str, tuple[list[Panel], int]]:
    """Datasets in resampling order, each a list of panels sharing participants."""
    out: dict[str, tuple[list[Panel], int]] = {}
    if args.get("cdm"):
        out.update(cdm_panels(load_cdm(args["cdm"])))
    if args.get("reutskaja"):
        panel, ids = load_reutskaja(args["reutskaja"])
        out["reutskaja"] = ([panel], len(ids))
    if args.get("snack"):
        panel, P = snack_panel(args["snack"])
        out["snack"] = ([panel], P)
    return out


def datasets_of(G: dict) -> dict[str, list[str]]:
    """Panels pooled into each dataset's decision."""
    out = {}
    for g, (panels, _) in G.items():
        ds = "cdm" if g.startswith("cdm") else g
        out.setdefault(ds, []).extend(p.name for p in panels)
    return out


def run_replicate(b: int) -> dict:
    G = _STATE["groups"]
    rng = None if b < 0 else np.random.default_rng(SEED_OFFSET + b)
    rec: dict = {"b": b, "panels": {}}
    for _, (panels, P) in G.items():
        mult = np.ones(P) if rng is None else np.bincount(rng.integers(0, P, size=P), minlength=P).astype(float)
        for p in panels:
            rec["panels"][p.name] = panel_measures(p, mult[p.participant])
    rec["datasets"] = {}
    for ds, names in _STATE["datasets"].items():
        R = [rec["panels"][x] for x in names]
        g = {k: float(np.mean([r["growth"][k] for r in R])) for k in ("observed", *MODELS)}
        err = {m: float(np.mean([abs(r["growth"][m] - r["growth"]["observed"]) for r in R])) for m in MODELS}
        auc = {k: float(np.mean([r["auc"][k] for r in R])) for k in (*MODELS, *FEATURES)}
        rec["datasets"][ds] = {"growth": g, "error": err, "auc": auc}
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(b) for b in bs]


def _s(point: dict, reps: list[dict], get) -> dict:
    return summarize(get(point), [get(r) for r in reps])


def summarize_all(point: dict, reps: list[dict]) -> dict:
    out: dict = {"datasets": {}, "panels": {}}
    for ds in point["datasets"]:

        def S(get, ds=ds):  # noqa: N802
            return _s(point, reps, lambda r: get(r["datasets"][ds]))

        D = point["datasets"][ds]
        B: dict = {
            "growth": {k: S(lambda x, k=k: x["growth"][k]) for k in D["growth"]},
            "error": {m: S(lambda x, m=m: x["error"][m]) for m in MODELS},
            "auc": {k: S(lambda x, k=k: x["auc"][k]) for k in D["auc"]},
        }
        B["comp_minus_cl_error"] = S(lambda x: x["error"]["CL_comp"] - x["error"]["CL"])
        B["observed_minus_comp"] = S(lambda x: x["growth"]["observed"] - x["growth"]["CL_comp"])
        B["observed_minus_cl"] = S(lambda x: x["growth"]["observed"] - x["growth"]["CL"])
        B["sr_auc_minus_half"] = S(lambda x: x["auc"]["SR"] - 0.5)
        B["sr_auc_minus_near_best"] = S(lambda x: x["auc"]["SR"] - x["auc"]["near_best"])
        above = B["observed_minus_comp"]["clear"] and B["observed_minus_comp"]["point"] > 0
        below = B["observed_minus_cl"]["clear"] and B["observed_minus_cl"]["point"] < 0
        verdict = (
            "partial compensation"
            if above and below
            else "overload"
            if above
            else "compensation"
            if below
            else "undecided"
        )
        B["O1"] = {"verdict": verdict}
        d = B["sr_auc_minus_near_best"]
        B["O2"] = {"supported": bool(d["clear"] and d["point"] > 0)}
        out["datasets"][ds] = B
    for name, P in point["panels"].items():
        out["panels"][name] = {"sizes": P["sizes"], "rate": P["rate"], "lambda": P["lambda"]}
        out["panels"][name]["rate_interval"] = {
            k: np.percentile([r["panels"][name]["rate"][k] for r in reps], [2.5, 97.5], axis=0).tolist()
            for k in P["rate"]
        }
        if "miss_best_unseen" in P:
            out["panels"][name]["miss_best_unseen"] = P["miss_best_unseen"]
    return out


def run(sources: dict, out: Path, replicates: int = 1000, jobs: int = 1, status: str = "") -> dict:
    G = groups(sources)
    _STATE.update(groups=G, datasets=datasets_of(G))
    t0 = time.time()
    point = run_replicate(-1)
    order = list(range(replicates))
    if jobs > 1 and replicates:
        size = max(1, replicates // (4 * jobs))
        chunks = [order[i : i + size] for i in range(0, len(order), size)]
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            reps = [r for c in pool.map(_worker, chunks) for r in c]
    else:
        reps = _worker(order)
    reps.sort(key=lambda r: r["b"])
    from ..engine import provenance, write_json

    out.mkdir(parents=True, exist_ok=True)
    with (out / "replicates.jsonl").open("w") as fh:
        for r in reps:
            fh.write(json.dumps(r) + "\n")
    write_json(out / "point.json", point)
    summary = summarize_all(point, reps)
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": status}
    )
    write_json(out / "summary.json", summary)
    (out / "table_overload.md").write_text(table(summary))
    return summary


def _f(e: dict, d: int = 3, sign: bool = True) -> str:
    fmt = f"{{:{'+' if sign else ''}.{d}f}}"
    lo, hi = e["percentile"]
    return f"{fmt.format(e['point'])} [{lo:.{d}f}, {hi:.{d}f}]{' *' if e.get('clear') else ''}"


def table(s: dict) -> str:
    L = ["# Forecasting choice overload (Amendment 8)", ""]
    for ds, B in s["datasets"].items():
        L += [f"## {ds}", "", "| Measure | Value |", "|---|---|"]
        for k, e in B["growth"].items():
            L.append(f"| Growth of misses per doubling of the set, {k} | {_f(e)} |")
        for m in MODELS:
            L.append(f"| Error of the forecast growth, {m} | {_f(B['error'][m], sign=False)} |")
        L.append(f"| O1, people's growth minus compensation's | {_f(B['observed_minus_comp'])} |")
        L.append(f"| O1, people's growth minus CL's | {_f(B['observed_minus_cl'])} |")
        L.append(f"| Compensation's error minus CL's (descriptive) | {_f(B['comp_minus_cl_error'])} |")
        L.append(f"| O1 verdict | {B['O1']['verdict']} |")
        for k, e in B["auc"].items():
            L.append(f"| Within set size, area under the curve, {k} | {_f(e, sign=False)} |")
        L.append(f"| O2, SR minus the near best count | {_f(B['sr_auc_minus_near_best'])} |")
        L.append(f"| SR minus one half (descriptive) | {_f(B['sr_auc_minus_half'])} |")
        L.append(f"| O2 supported | {B['O2']['supported']} |")
        L.append("")
    L += ["## Miss rates by set size (original sample)", ""]
    for name, P in s["panels"].items():
        L.append(f"- {name}, sizes {P['sizes']}")
        for k, r in P["rate"].items():
            L.append(f"  - {k}: " + ", ".join(f"{x:.3f}" for x in r))
        if "miss_best_unseen" in P:
            L.append(
                "  - misses whose best option was never fixated: "
                + ", ".join(f"{x:.3f}" for x in P["miss_best_unseen"])
            )
    L += [
        "",
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose "
        "percentile and basic intervals both exclude zero.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdm", type=Path)
    ap.add_argument("--reutskaja", type=Path)
    ap.add_argument("--snack", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    status = "exploratory" if a.snack else "confirmatory, Amendment 8"
    summary = run({"cdm": a.cdm, "reutskaja": a.reutskaja, "snack": a.snack}, a.out, a.replicates, a.jobs, status)
    print((a.out / "table_overload.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
