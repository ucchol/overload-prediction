"""Confirmatory tests of the search model on the search and satisficing data of Caplin, Dean and Martin (2011),
ANALYSIS_PLAN.md Amendment 7.

Here every selection a person made on the way to a choice was recorded, so the end of a search can be watched rather
than inferred. The original authors reported that people stopped once their selection was worth more than a
reservation amount (satisficing). Sample until return makes claims these data can test head on.

C1  Stopping. A search that ends on its own clock, as at a first return, ends with a selection according to how many
    better options remain to be found. A search that ends at an absolute standard ends according to what the
    selection is worth. Held out, the dollar value of a selection should then foresee whether it ends the search no
    better than the share of options in the set worth more.
C2  Order of search. A value is unknown until it is worked out, so a reader blind to the screen picks the first
    option to work out at random, and in the model attention goes by value. In the choice process experiment, where
    holding nothing paid nothing, the first selection is the first option worked out. It should then lie no higher
    on the screen than one half of the list, or than the model's own attention predicts.
C3  Choices. As well as CL, within 2 percent of the held out likelihood of the choice.

Descriptive. How the time to the final selection grows with the number of options, next to the model's expected
looks. The data hold no submission times, so when a search ended is not known, and this is not a test.

Every prediction comes from fits to other people (ten folds of participants within each experiment). C1 and C3 are
the means of their values in the two experiments, C2 is taken in the choice process experiment.

Usage
    python -m overload.experiments.stopping_cdm --cdm data/cdm --out results/stopping_cdm --jobs 2
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..bootstrap import summarize
from ..cdm import (
    CELLS,
    COMPLEXITIES,
    EXPERIMENTS,
    SIZES,
    CDMData,
    Displays,
    chosen_log_softmax,
    fit_logistic,
    fit_softmax,
    fit_sr_displays,
    levels,
    load_cdm,
    logistic_log2,
    softmax_masked,
    sr_levels,
    static_scores,
)
from .search_confirmatory import fit_time_law
from .search_confirmatory import noninferiority as _noninferiority

SEED_OFFSET = 70000
FOLD_SEED = 20261015
FOLDS = 10
STATIC = {"CL": "CL", "DbS": "DbS_avg", "MLBA": "MLBA_avg"}
CHOICE_MODELS = (*STATIC, "SR")
RULES = ("value", "share", "value_time", "share_time")
LN2 = np.log(2.0)
_STATE: dict = {}


def participants_of(data: CDMData, e: int) -> np.ndarray:
    """Sorted participant indices of one experiment."""
    return np.unique(data.participant[data.experiment == e])


def folds(n_participants: int) -> np.ndarray:
    perm = np.random.default_rng(FOLD_SEED).permutation(n_participants)
    fold = np.zeros(n_participants, dtype=int)
    fold[perm] = np.arange(n_participants) % FOLDS
    return fold


def draws(data: CDMData, b: int) -> dict[int, np.ndarray | None]:
    if b < 0:
        return {e: None for e in range(len(EXPERIMENTS))}
    rng = np.random.default_rng(SEED_OFFSET + b)
    out = {}
    for e in range(len(EXPERIMENTS)):
        P = participants_of(data, e).size
        out[e] = rng.integers(0, P, size=P)
    return out


def _elasticity(x: np.ndarray, w: np.ndarray, n: np.ndarray, c: np.ndarray) -> float:
    """Least squares slope of the log weighted geometric mean of x on log set size, averaged over complexities."""
    slopes = []
    for cc in COMPLEXITIES:
        gm = []
        for nn in SIZES:
            r = (n == nn) & (c == cc) & (w > 0)
            gm.append(float(w[r] @ np.log(x[r]) / w[r].sum()))
        slopes.append(float(np.polyfit(np.log(SIZES), gm, 1)[0]))
    return float(np.mean(slopes))


def _rule_columns(rule: str, value: np.ndarray, share: np.ndarray, log_time: np.ndarray) -> np.ndarray:
    return {
        "value": value[:, None],
        "share": share[:, None],
        "value_time": np.column_stack([value, log_time]),
        "share_time": np.column_stack([share, log_time]),
    }[rule]


def _experiment(data: CDMData, e: int, d: np.ndarray | None) -> dict:
    people = participants_of(data, e)
    P = people.size
    local = np.full(len(data.participants), -1)
    local[people] = np.arange(P)
    mult = np.ones(P) if d is None else np.bincount(d, minlength=P).astype(float)
    rows = np.flatnonzero(data.experiment == e)
    part = local[data.participant[rows]]
    w = mult[part]
    fold_of = folds(P)[part]
    values = data.values[rows]
    choice = data.choice[rows]
    n, c = data.n[rows], data.complexity[rows]
    R = rows.size
    lv = levels(values, choice)
    disp = Displays.of(lv)
    scores = {m: static_scores(values, fam) for m, fam in STATIC.items()}

    # selections of this experiment
    sel = np.flatnonzero(np.isin(data.sel_round, rows))
    round_index = np.full(data.rounds, -1)
    round_index[rows] = np.arange(R)
    s_round = round_index[data.sel_round[sel]]
    s_value = data.sel_value[sel]
    s_share = data.sel_better[sel] / data.n[data.sel_round[sel]]
    s_logt = np.log(data.sel_time[sel])
    s_stop = data.sel_stop[sel]
    s_open = data.sel_better[sel] > 0  # the set's best ends every search with recall, so it tells nothing
    s_w = w[s_round]
    s_fold = fold_of[s_round]
    s_cell = np.array([CELLS.index((a, b)) for a, b in zip(n[s_round], c[s_round], strict=True)])

    logq = {m: np.zeros(R) for m in CHOICE_MODELS}
    looks = np.zeros(R)
    first_pred = np.zeros(R)
    rel_pos = np.arange(values.shape[1])[None, :] / (n[:, None] - 1.0)
    log_seconds = np.log(data.final_time[rows])
    time_err = {m: np.zeros(R) for m in ("static", "SR")}
    rule_score = {r: np.full(sel.size, np.nan) for r in RULES}
    lam: dict[str, list[float]] = {m: [] for m in CHOICE_MODELS}
    for f in range(FOLDS):
        test, train = fold_of == f, fold_of != f
        for m in STATIC:
            lm = fit_softmax(scores[m][train], choice[train], w[train])
            lam[m].append(lm)
            logq[m][test] = chosen_log_softmax(lm, scores[m][test], choice[test])
        ls = fit_sr_displays(disp, np.where(train, w, 0.0))
        lam["SR"].append(ls)
        logp_u, lk_u = sr_levels(ls, disp.unique)
        logp, lk = logp_u[disp.index], lk_u[disp.index]
        logq["SR"][test] = np.maximum(logp[np.flatnonzero(test), lv.chosen[test]], np.log(1e-300))
        looks[test] = lk[test]
        q = softmax_masked(ls * values[test])
        first_pred[test] = np.nansum(q * rel_pos[test], axis=1)
        effort = {"static": n.astype(float), "SR": lk}
        for m in ("static", "SR"):
            t0, tau = fit_time_law(effort[m][train], log_seconds[train], w[train])
            time_err[m][test] = np.abs(log_seconds[test] - np.log(t0 + tau * effort[m][test]))
        for k in range(len(CELLS)):
            tr = (s_fold != f) & (s_cell == k) & s_open
            te = (s_fold == f) & (s_cell == k) & s_open
            if not te.any():
                continue
            for rule in RULES:
                X = _rule_columns(rule, s_value, s_share, s_logt)
                beta, mu, sd = fit_logistic(X[tr], s_stop[tr], s_w[tr])
                rule_score[rule][te] = logistic_log2(beta, mu, sd, X[te], s_stop[te])

    rec: dict = {"lambda": lam}
    rec["L"] = {m: float(w @ ((np.log(n) + logq[m]) / LN2) / w.sum()) for m in CHOICE_MODELS}
    ok = s_open & (s_w > 0)
    rec["stop"] = {r: float(s_w[ok] @ rule_score[r][ok] / s_w[ok].sum()) for r in RULES}
    fs = data.sel_first[sel]
    first_rel = np.zeros(R)
    first_rel[s_round[fs]] = data.sel_position[sel][fs] / (n[s_round[fs]] - 1.0)
    obs, pred = float(w @ first_rel / w.sum()), float(w @ first_pred / w.sum())
    rec["first"] = {"observed": obs, "predicted": pred, "minus_half": obs - 0.5, "minus_predicted": obs - pred}
    rec["growth"] = {
        "observed": _elasticity(data.final_time[rows], w, n, c),
        "predicted": _elasticity(looks, w, n, c),
        "observed_by_size": [
            float(np.exp(w[(n == nn) & (w > 0)] @ log_seconds[(n == nn) & (w > 0)] / w[(n == nn) & (w > 0)].sum()))
            for nn in SIZES
        ],
        "looks_by_size": [float(w[n == nn] @ looks[n == nn] / w[n == nn].sum()) for nn in SIZES],
    }
    rec["time"] = {m: float(w @ time_err[m] / w.sum()) for m in time_err}
    return rec


def run_replicate(b: int) -> dict:
    data: CDMData = _STATE["data"]
    d = draws(data, b)
    rec: dict = {"b": b, "experiments": {EXPERIMENTS[e]: _experiment(data, e, d[e]) for e in range(len(EXPERIMENTS))}}
    E = rec["experiments"]
    rec["pooled"] = {
        "L": {m: float(np.mean([E[x]["L"][m] for x in EXPERIMENTS])) for m in CHOICE_MODELS},
        "stop": {r: float(np.mean([E[x]["stop"][r] for x in EXPERIMENTS])) for r in RULES},
        "growth": {k: float(np.mean([E[x]["growth"][k] for x in EXPERIMENTS])) for k in ("observed", "predicted")},
        "time": {m: float(np.mean([E[x]["time"][m] for x in EXPERIMENTS])) for m in ("static", "SR")},
    }
    return rec


def _worker(bs: list[int]) -> list[dict]:
    return [run_replicate(b) for b in bs]


def _percent(dl: float) -> float:
    return 100.0 * (2.0**dl - 1.0)


def _percent_summary(e: dict) -> dict:
    return {
        "point": _percent(e["point"]),
        "percentile": [_percent(x) for x in e["percentile"]],
        "basic": [_percent(x) for x in e["basic"]],
        "p": e["p"],
        "clear": e["clear"],
    }


def _s(point: dict, reps: list[dict], get) -> dict:
    return summarize(get(point), [get(r) for r in reps])


def _block(point: dict, reps: list[dict], get) -> dict:
    out: dict = {}
    out["stop"] = {r: _s(point, reps, lambda x, r=r: get(x)["stop"][r]) for r in RULES}
    out["stop"]["value_minus_share"] = _s(point, reps, lambda x: get(x)["stop"]["value"] - get(x)["stop"]["share"])
    out["stop"]["value_minus_share_with_time"] = _s(
        point, reps, lambda x: get(x)["stop"]["value_time"] - get(x)["stop"]["share_time"]
    )
    if "first" in get(point):
        out["first"] = {k: _s(point, reps, lambda x, k=k: get(x)["first"][k]) for k in get(point)["first"]}
    g = {k: _s(point, reps, lambda x, k=k: get(x)["growth"][k]) for k in ("observed", "predicted")}
    g["one_minus_observed"] = _s(point, reps, lambda x: 1.0 - get(x)["growth"]["observed"])
    g["predicted_minus_observed"] = _s(
        point, reps, lambda x: get(x)["growth"]["predicted"] - get(x)["growth"]["observed"]
    )
    out["growth"] = g
    out["choices"] = {"times_chance": {m: 2.0 ** get(point)["L"][m] for m in CHOICE_MODELS}}
    for m in STATIC:
        e = _s(point, reps, lambda x, m=m: get(x)["L"]["SR"] - get(x)["L"][m])
        out["choices"][f"SR_minus_{m}_percent"] = _percent_summary(e)
    out["time"] = {m: _s(point, reps, lambda x, m=m: get(x)["time"][m]) for m in ("static", "SR")}
    out["time"]["SR_minus_static"] = _s(point, reps, lambda x: get(x)["time"]["SR"] - get(x)["time"]["static"])
    return out


def summarize_all(point: dict, reps: list[dict]) -> dict:
    out = {"pooled": _block(point, reps, lambda r: r["pooled"])}
    for name in EXPERIMENTS:
        B = _block(point, reps, lambda r, name=name: r["experiments"][name])
        B["lambda"] = point["experiments"][name]["lambda"]
        B["growth"]["observed_by_size"] = point["experiments"][name]["growth"]["observed_by_size"]
        B["growth"]["looks_by_size"] = point["experiments"][name]["growth"]["looks_by_size"]
        out[name] = B
    P = out["pooled"]
    c1 = P["stop"]["value_minus_share"]
    P["C1"] = {
        "absolute": bool(c1["clear"] and c1["point"] > 0),
        "relative": bool(c1["clear"] and c1["point"] < 0),
    }
    f = out["process"]["first"]
    P["C2"] = {
        "order_matters": bool(
            f["minus_half"]["clear"]
            and f["minus_half"]["point"] < 0
            and f["minus_predicted"]["clear"]
            and f["minus_predicted"]["point"] < 0
        )
    }
    P["C3"] = _noninferiority(P["choices"]["SR_minus_CL_percent"])
    P["C3"]["supported"] = P["C3"]["noninferior"]
    return out


def run(folder: Path, out: Path, replicates: int = 1000, jobs: int = 1) -> dict:
    _STATE.update(data=load_cdm(folder))
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
    from ..engine import provenance, write_json

    write_json(out / "point.json", point)
    summary = summarize_all(point, reps)
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "confirmatory, Amendment 7"}
    )
    write_json(out / "summary.json", summary)
    (out / "table_stopping_cdm.md").write_text(table(summary))
    return summary


def _f(e: dict, digits: int = 3, sign: bool = False) -> str:
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    lo, hi = e["percentile"]
    star = " *" if e.get("clear") else ""
    return f"{fmt.format(e['point'])} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def table(s: dict) -> str:
    names = {
        "pooled": "Mean of the two experiments",
        "standard": "Experiment 1, standard choice",
        "process": "Experiment 2, choice process",
    }
    P = s["pooled"]
    f = s["process"]["first"]
    c1 = P["C1"]
    verdict = "absolute" if c1["absolute"] else "relative" if c1["relative"] else "undecided"
    L = [
        "# The search model's stopping rule on the data of Caplin, Dean and Martin (Amendment 7, confirmatory)",
        "",
        "## Decisions",
        "",
        f"- C1, stopping. Value rule minus share rule {_f(P['stop']['value_minus_share'], 4, True)} bits per "
        f"selection. Verdict {verdict}.",
        f"- C2, order of search (Experiment 2). First selection's place {_f(f['observed'], 3)}, minus one half "
        f"{_f(f['minus_half'], 3, True)}, minus the model's {_f(f['minus_predicted'], 3, True)}. Order matters "
        f"{P['C2']['order_matters']}.",
        f"- C3, choices. Sample until return minus CL {_f(P['choices']['SR_minus_CL_percent'], 2, True)} percent. "
        f"Non inferior within 2 percent {P['C3']['noninferior']}, better {P['C3']['better']}, worse "
        f"{P['C3']['worse']}. Supported {P['C3']['supported']}.",
        "",
    ]
    for key in ("pooled", *EXPERIMENTS):
        B = s[key]
        L += [f"## {names[key]}", "", "| Measure | Value |", "|---|---|"]
        for r in RULES:
            L.append(f"| Stopping, {r} rule, log2 probability per selection | {_f(B['stop'][r], 4)} |")
        L.append(f"| Stopping, value minus share | {_f(B['stop']['value_minus_share'], 4, True)} |")
        L.append(
            f"| Stopping, value minus share, time in both | {_f(B['stop']['value_minus_share_with_time'], 4, True)} |"
        )
        if "first" in B:
            L.append(f"| First selection's place (0 top, 1 bottom) | {_f(B['first']['observed'], 3)} |")
            L.append(f"| The model's predicted place | {_f(B['first']['predicted'], 3)} |")
        g = B["growth"]
        L.append(f"| Growth of the final selection time with set size (elasticity) | {_f(g['observed'], 2, True)} |")
        L.append(f"| Growth of the model's expected looks (elasticity) | {_f(g['predicted'], 2, True)} |")
        if "observed_by_size" in g:
            obs = ", ".join(f"{x:.1f}" for x in g["observed_by_size"])
            lk = ", ".join(f"{x:.1f}" for x in g["looks_by_size"])
            L.append(f"| Final selection time at 10, 20 and 40 options, seconds | {obs} |")
            L.append(f"| The model's expected looks at 10, 20 and 40 options | {lk} |")
        tc = B["choices"]["times_chance"]
        L.append(
            "| Choices, times chance, "
            + ", ".join(CHOICE_MODELS)
            + " | "
            + ", ".join(f"{tc[m]:.3f}" for m in CHOICE_MODELS)
            + " |"
        )
        for m in STATIC:
            L.append(f"| Choices, SR minus {m}, percent | {_f(B['choices'][f'SR_minus_{m}_percent'], 2, True)} |")
        L.append(f"| Time error, static law (number of options) | {_f(B['time']['static'], 3)} |")
        L.append(f"| Time error, SR's expected looks | {_f(B['time']['SR'], 3)} |")
        L.append(f"| Time error, SR minus static | {_f(B['time']['SR_minus_static'], 4, True)} |")
        L.append("")
    L += [
        "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile "
        "and basic intervals both exclude zero.",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cdm", type=Path, default=Path("data/cdm"))
    ap.add_argument("--out", type=Path, default=Path("results/stopping_cdm"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.cdm, args.out, args.replicates, args.jobs)
    print((args.out / "table_stopping_cdm.md").read_text())
    print(f"written in {summary['provenance']['seconds']} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
