"""Which trials overload people? The overload index of the development data (exploratory).

Question. After every trial the study asked six questions about the choice just made. Do they measure one
thing, in which trials were people overloaded, and did overload go with worse choices and longer decisions?

Why it matters. To foresee overload before anyone chooses (the premonition analysis), we first need a
trustworthy measure of it and a map of where it occurred. Choice overload is usually tested as one contrast
between a small and a large set. Here every participant rated twelve trials, so overload can be located
trial by trial, against each person's own baseline.

How.
1. Measurement. The six answers are aligned so that higher always means more overloaded (satisfaction and
   certainty reversed, see ``overload.responses``). Their agreement is Cronbach's alpha on answers centered
   within each participant, so that it reflects changes from trial to trial rather than differences between
   people, together with the share of that within person variance carried by the first principal component.
2. Location. The overload index of a trial is the mean aligned answer, in scale points from 1 to 5. A trial
   overloaded people when its index clearly exceeds the participants' own average over all twelve trials
   (both bootstrap intervals above zero), and relieved them when it clearly falls below.
3. Set size and information. Within each task, the change from 8 to 32 options (averaged over the two
   information levels) and from brief to full information (averaged over set sizes), for the overload index,
   regret and decision time (geometric mean, as a percent change).
4. Feelings, errors and effort. Correlations of the index with regret and with log decision time after
   removing each participant's average and each trial's average. They answer one question. When a person
   felt more overloaded than usual on a trial, compared with others on the same trial, had they chosen worse
   or taken longer?

Uncertainty. 1000 participant resamples, each participant drawn with all twelve trials (seed 30000 + b).
These are development data, so every result here is exploratory.

Usage
    python -m overload.experiments.overload_index --data PATH_TO_CSV --out results/overload
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..bootstrap import summarize
from ..engine import provenance, write_json
from ..responses import ITEMS, QUESTIONS, aligned, load_responses

SEED_OFFSET = 30000
TASK_ORDER = ("file", "laptop")


# ----------------------------------------------------------------------------- arrays
def to_arrays(R) -> dict:
    """Participant by trial arrays (twelve trials, files first), every answer, regret and log time."""
    R = R.sort_values(
        ["person", "task", "trial"], key=lambda s: s.map({"file": 0, "laptop": 1}) if s.name == "task" else s
    )
    persons = np.sort(R["person"].unique())
    n_p, n_t = len(persons), 12
    A = aligned(R).to_numpy(float).reshape(n_p, n_t, len(ITEMS))
    raw = R[list(ITEMS)].to_numpy(float).reshape(n_p, n_t, len(ITEMS))
    out = {
        "aligned": A,
        "raw": raw,
        "overload": A.mean(axis=2),
        "regret": R["regret"].to_numpy(float).reshape(n_p, n_t),
        "logtime": np.log(R["seconds"].to_numpy(float)).reshape(n_p, n_t),
    }
    keys = R.drop_duplicates(["task", "trial"])[["task", "trial", "key", "options", "information"]]
    out["trials"] = keys.to_dict("records")
    if not np.isfinite(np.stack([out["overload"], out["regret"], out["logtime"]])).all():
        raise ValueError("every participant must have every answer, regret and time in all twelve trials")
    return out


# ----------------------------------------------------------------------------- statistics
def cronbach_alpha(X: np.ndarray) -> float:
    """Cronbach's alpha of the columns of X (observations by items)."""
    k = X.shape[1]
    return float(k / (k - 1) * (1.0 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1)))


def _within_person(X: np.ndarray) -> np.ndarray:
    """Center every participant's values over their twelve trials (axis 1)."""
    return X - X.mean(axis=1, keepdims=True)


def _two_way(X: np.ndarray) -> np.ndarray:
    """Remove participant and trial averages (participants by trials)."""
    return X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()


def measurement(arr: dict) -> dict:
    A = _within_person(arr["aligned"]).reshape(-1, len(ITEMS))
    eig = np.linalg.eigvalsh(np.corrcoef(A.T))[::-1]
    raw = _within_person(arr["raw"]).reshape(-1, len(ITEMS))
    r_raw = np.corrcoef(raw.T)
    q6 = ITEMS.index("uncertain")
    return {
        "alpha_within_person": cronbach_alpha(A),
        "alpha_raw": cronbach_alpha(arr["aligned"].reshape(-1, len(ITEMS))),
        "first_component_share": float(eig[0] / eig.sum()),
        "question6_raw_correlations": {ITEMS[k]: float(r_raw[q6, k]) for k in range(len(ITEMS)) if k != q6},
    }


def trial_means(arr: dict) -> dict:
    O, G, S = arr["overload"], arr["regret"], arr["logtime"]
    grand = O.mean()
    return {
        "overload": O.mean(axis=0),
        "overload_vs_average": O.mean(axis=0) - grand,
        "regret": G.mean(axis=0),
        "best_chosen": (G == 0).mean(axis=0),
        "geo_seconds": np.exp(S.mean(axis=0)),
        "items": arr["raw"].mean(axis=0),
    }


def contrasts(arr: dict) -> dict:
    """Within task, 32 minus 8 options and full minus brief information, for overload, regret and log time."""
    out = {}
    for k, task in enumerate(TASK_ORDER):
        cols = slice(6 * k, 6 * k + 6)
        for measure in ("overload", "regret", "logtime"):
            M = arr[measure][:, cols].mean(axis=0)  # trials 1 to 6: 8L, 8H, 16L, 16H, 32L, 32H
            size = ((M[4] + M[5]) - (M[0] + M[1])) / 2.0
            info = ((M[1] + M[3] + M[5]) - (M[0] + M[2] + M[4])) / 3.0
            if measure == "logtime":
                size, info = 100.0 * (np.exp(size) - 1.0), 100.0 * (np.exp(info) - 1.0)
            out[f"{task}|{measure}|32_minus_8"] = float(size)
            out[f"{task}|{measure}|full_minus_brief"] = float(info)
    return out


def associations(arr: dict) -> dict:
    out = {}
    for k, task in enumerate((*TASK_ORDER, "both")):
        cols = slice(0, 12) if task == "both" else slice(6 * k, 6 * k + 6)
        O, G, S = (_two_way(arr[m][:, cols]).ravel() for m in ("overload", "regret", "logtime"))
        out[f"{task}|overload~regret"] = float(np.corrcoef(O, G)[0, 1])
        out[f"{task}|overload~logtime"] = float(np.corrcoef(O, S)[0, 1])
        out[f"{task}|regret~logtime"] = float(np.corrcoef(G, S)[0, 1])
    return out


def statistics(arr: dict) -> dict:
    """Every quantity of one sample, flat, for the bootstrap."""
    stats = {f"measurement|{k}": v for k, v in measurement(arr).items() if not isinstance(v, dict)}
    tm = trial_means(arr)
    for j, t in enumerate(arr["trials"]):
        for name in ("overload", "overload_vs_average", "regret", "best_chosen", "geo_seconds"):
            stats[f"trial|{t['key']}|{name}"] = float(tm[name][j])
    stats.update({f"contrast|{k}": v for k, v in contrasts(arr).items()})
    stats.update({f"association|{k}": v for k, v in associations(arr).items()})
    return stats


def resample(arr: dict, b: int) -> dict:
    rng = np.random.default_rng(SEED_OFFSET + b)
    idx = rng.integers(0, arr["overload"].shape[0], size=arr["overload"].shape[0])
    out = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
    return out


# ----------------------------------------------------------------------------- run
def run(csv: Path, out: Path, replicates: int = 1000) -> dict:
    t0 = time.time()
    arr = to_arrays(load_responses(csv))
    point = statistics(arr)
    reps = [statistics(resample(arr, b)) for b in range(replicates)]
    summary: dict = {"point_only": {}, "measures": {}}
    for key, value in point.items():
        values = np.array([r[key] for r in reps])
        if key.startswith("trial|") and not key.endswith("overload_vs_average"):
            summary["measures"][key] = {"point": value, "percentile": list(np.percentile(values, [2.5, 97.5]))}
        else:
            summary["measures"][key] = summarize(value, values)
    m = measurement(arr)
    summary["point_only"]["question6_raw_correlations"] = m["question6_raw_correlations"]
    summary["point_only"]["item_means"] = {
        t["key"]: dict(zip(ITEMS, trial_means(arr)["items"][j].tolist(), strict=True))
        for j, t in enumerate(arr["trials"])
    }
    summary["trials"] = arr["trials"]
    summary["questions"] = [{"name": n, "direction": s, "wording": w} for n, s, w in QUESTIONS]
    summary["provenance"] = provenance(
        {"replicates": replicates, "seconds": round(time.time() - t0, 1), "status": "exploratory"}
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", summary)
    (out / "table_overload.md").write_text(table(summary))
    return summary


# ----------------------------------------------------------------------------- report
def _fmt(s: dict, scale: float = 1.0, digits: int = 2) -> str:
    p = s["point"] * scale
    lo, hi = (x * scale for x in s["percentile"])
    star = " *" if s.get("clear") else ""
    return f"{p:+.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]{star}"


def _level(s: dict, digits: int) -> str:
    lo, hi = s["percentile"]
    return f"{s['point']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


NOTE = (
    "Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a difference or "
    "correlation whose percentile and basic intervals both exclude zero. The overload index runs from 1 (none) "
    "to 5 (strong). Time is the geometric mean on the choice page. Development data, exploratory."
)


def table(summary: dict) -> str:
    M = summary["measures"]
    alpha, share = M["measurement|alpha_within_person"], M["measurement|first_component_share"]
    q6 = ", ".join(f"{k} {v:+.2f}" for k, v in summary["point_only"]["question6_raw_correlations"].items())
    lines = [
        "# Overload index, development data (exploratory)",
        "",
        "## Do the six questions measure one thing?",
        "",
        f"- Cronbach's alpha within participants {_level(alpha, 2)}, "
        f"over all answers {M['measurement|alpha_raw']['point']:.2f}.",
        f"- The first principal component carries {100 * share['point']:.0f}% of the within participant variance.",
        f"- Question 6 against the other answers, raw and within participants: {q6}.",
        "",
        "## Where were people overloaded?",
        "",
        "| Trial | Options | Information | Overload index | Against own average | Regret | Best chosen, % | Time, s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in summary["trials"]:
        k = t["key"]
        o, d, g = M[f"trial|{k}|overload"], M[f"trial|{k}|overload_vs_average"], M[f"trial|{k}|regret"]
        b, s = M[f"trial|{k}|best_chosen"], M[f"trial|{k}|geo_seconds"]
        cells = [k, str(t["options"]), t["information"], _level(o, 2), _fmt(d), _level(g, 1)]
        cells += [f"{100 * b['point']:.0f}", _level(s, 0)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## More options and more information",
        "",
        "| Task | Measure | 32 minus 8 options | Full minus brief |",
        "|---|---|---|---|",
    ]
    units = {"overload": "scale points", "regret": "points", "logtime": "percent time"}
    for task in TASK_ORDER:
        for measure in ("overload", "regret", "logtime"):
            a, c = M[f"contrast|{task}|{measure}|32_minus_8"], M[f"contrast|{task}|{measure}|full_minus_brief"]
            digits = 2 if measure == "overload" else 1
            lines.append(
                f"| {task} | {measure} ({units[measure]}) | {_fmt(a, digits=digits)} | {_fmt(c, digits=digits)} |"
            )
    lines += [
        "",
        "## Feelings, errors and effort (within participant and trial correlations)",
        "",
        "| | Overload and regret | Overload and log time | Regret and log time |",
        "|---|---|---|---|",
    ]
    for task in (*TASK_ORDER, "both"):
        cells = [
            _fmt(M[f"association|{task}|{pair}"]) for pair in ("overload~regret", "overload~logtime", "regret~logtime")
        ]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", NOTE, ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/overload"))
    ap.add_argument("--replicates", type=int, default=1000)
    args = ap.parse_args()
    summary = run(args.data, args.out, args.replicates)
    print((args.out / "table_overload.md").read_text())
    print(json.dumps({"seconds": summary["provenance"]["seconds"]}))


if __name__ == "__main__":
    main()
