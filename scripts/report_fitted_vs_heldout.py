"""Fitted and held out forecasts of human overload for every model, from saved results (exploratory).

Usage: python -I scripts/report_fitted_vs_heldout.py
Writes results/overload_fitted_vs_heldout.md. Reads only summary files, no raw data.
"""

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
MODELS = ("CL", "CL_comp", "SR", "MXL", "DbS", "MLBA")
NAMES = {"CL": "CL", "CL_comp": "CL with compensation", "SR": "Sample until return", "MXL": "Mixed logit",
         "DbS": "DbS", "MLBA": "MLBA"}
SOURCES = {
    "cdm": ("Caplin, Dean and Martin", "overload_all_models", "overload_fitted"),
    "reutskaja": ("Reutskaja and colleagues", "overload_all_models", "overload_fitted"),
    "snack": ("Snacks", "overload_snack_all", "overload_snack_fitted"),
}


def f(e, d=3):
    lo, hi = e["percentile"]
    return f"{e['point']:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]" + (" *" if e.get("clear") else "")


def main():
    L = ["# Fitted and held out forecasts of human overload, every model (exploratory)", "",
         "Growth is the rise in the share of choices that miss the best option per doubling of the set size. "
         "Held out forecasts each set size from fits to the others, fitted uses one fit to every set size. "
         "Error is the distance from people's growth. Bits per choice is the mean log2 probability of the chosen "
         "option (higher is better). Brackets give 95% percentile intervals from the participant bootstrap, a star "
         "marks a value whose percentile and basic intervals both exclude zero.", ""]
    for key, (title, ho_dir, fit_dir) in SOURCES.items():
        ho = json.loads((R / ho_dir / "summary.json").read_text())["datasets"][key]
        fi = json.loads((R / fit_dir / "summary.json").read_text())["datasets"][key]
        L += [f"## {title}", "", f"People's growth {f(ho['growth']['observed'])}.", "",
              "| Model | Growth, held out | Growth, fitted | Error, held out | Error, fitted | Bits, held out | Bits, fitted |",
              "|---|---|---|---|---|---|---|"]
        for m in MODELS:
            L.append(f"| {NAMES[m]} | {f(ho['growth'][m])} | {f(fi['growth'][m])} | {ho['error'][m]['point']:.3f} | "
                     f"{fi['error'][m]['point']:.3f} | {ho['loglik'][m]['point']:.3f} | {fi['loglik'][m]['point']:.3f} |")
        L += ["", f"O1 verdict held out: {ho['O1'] if isinstance(ho['O1'], str) else ho['O1'].get('verdict', ho['O1'])}. "
              f"Fitted: {fi['O1'] if isinstance(fi['O1'], str) else fi['O1'].get('verdict', fi['O1'])}.", ""]
    L += ["## Development data", "", "See `overload_dev/table_overload_dev.md` for every model's fitted and held out "
          "forecasts of misses, regret and felt overload by display, in file selection and laptop purchase.", ""]
    (R / "overload_fitted_vs_heldout.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
