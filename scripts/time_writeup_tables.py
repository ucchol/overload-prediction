"""LaTeX tables for the decision time write up (paper/time_section), written from stored results only.

Reads results/time_structure, results/statistics, results/time_models_dev/dev_r2_all.json and
results/time_models_vault (vault_eval.json, secondary.json, calibration slopes in paper/time_section/figures).
Writes paper/time_section/tables/*.tex. Usage, python3 -I scripts/time_writeup_tables.py
"""
import json
from pathlib import Path

R = Path("results")
OUT = Path("paper/time_section/tables")
OUT.mkdir(parents=True, exist_ok=True)
SETS = [("dev_file", "Our study, files", "\\makecell{Our\\\\files}"), ("dev_laptop", "Our study, laptops", "\\makecell{Our\\\\laptops}"),
        ("snack", "Snacks at a glance", "\\makecell{Snacks,\\\\glance}"), ("reutskaja", "Snacks in three seconds", "\\makecell{Snacks,\\\\3 s}"),
        ("cdm", "Sums worked out", "Sums"), ("krajbich2010", "Two foods", "\\makecell{Two\\\\foods}"),
        ("krajbich2011", "Three foods", "\\makecell{Three\\\\foods}"), ("folke2016", "Three snacks", "\\makecell{Three\\\\snacks}")]
SHORT = {"dev_file": "Our files", "dev_laptop": "Our laptops", "snack": "Snacks, glance", "reutskaja": "Snacks, 3 s",
         "cdm": "Sums", "krajbich2010": "Two foods", "krajbich2011": "Three foods", "folke2016": "Three snacks"}
NAMECOL = ">{\\raggedright\\arraybackslash}p{%s}"
NAME = {"S1": "Set size law", "S2": "Structure law", "CL_entropy": "CL entropy", "DbS_entropy": "DbS entropy",
        "MLBA_entropy": "MLBA entropy", "SR_looks": "Sample until return looks", "TLS_looks": "Two look satisficing looks",
        "TLS_tfit": "Two look satisficing, sensitivity fitted to time", "SR_timing": "Sample until return timing",
        "LBA": "LBA", "LBA_cal": "LBA, time slope", "LBA_n": "LBA, rising threshold",
        "LBA_n_cal": "LBA, rising threshold, time slope", "MLBA_LBA": "MLBA LBA", "MLBA_LBA_cal": "MLBA LBA, time slope",
        "MLBA_LBA_n": "MLBA LBA, rising threshold", "MLBA_LBA_n_cal": "MLBA LBA, rising threshold, time slope",
        "Wald": "Wald race", "Wald_cal": "Wald race, time slope", "Wald_n": "Wald race, rising threshold",
        "Wald_n_cal": "Wald race, rising threshold, time slope", "RWald": "Relative Wald",
        "RWald_cal": "Relative Wald, time slope", "RWald_n": "Relative Wald, rising threshold",
        "RWald_n_cal": "Relative Wald, rising threshold, time slope", "S2+TLS": "Structure plus satisficing looks",
        "S2+SR": "Structure plus sample until return looks", "S2+CLH": "Structure plus CL entropy",
        "S2+TLS_tfit": "Structure plus satisficing looks, sensitivity fitted to time",
        "S2+Wald_n": "Structure plus Wald race, rising threshold", "Ceiling": "Boosted trees ceiling",
        "Ceiling2": "Boosted trees ceiling, centered features"}
ORDER = ["S1", "S2", "CL_entropy", "DbS_entropy", "MLBA_entropy", "SR_looks", "TLS_looks", "TLS_tfit", "SR_timing",
         "LBA", "LBA_cal", "LBA_n", "LBA_n_cal", "MLBA_LBA", "MLBA_LBA_cal", "MLBA_LBA_n", "MLBA_LBA_n_cal",
         "Wald", "Wald_cal", "Wald_n", "Wald_n_cal", "RWald", "RWald_cal", "RWald_n", "RWald_n_cal",
         "S2+TLS", "S2+SR", "S2+CLH", "S2+TLS_tfit", "S2+Wald_n", "Ceiling", "Ceiling2"]


def f(x, d=3, sign=True):
    if x is None:
        return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return "$" + s.replace("-", "-") + "$"


def ci(lo, hi, d=3):
    return f"$[{lo:+.{d}f}, {hi:+.{d}f}]$"


def bf(x):
    if x is None:
        return "--"
    if x >= 100:
        return f"${x:.0f}$"
    if x >= 10:
        return f"${x:.1f}$"
    return f"${x:.2f}$"


def cell2(top, bottom):
    return "\\makecell{" + top + " \\\\ {\\scriptsize " + bottom + "}}"


def write(name, text):
    (OUT / name).write_text(text)
    print("wrote", OUT / name)


# ------------------------------------------------------------------ Table, Amendment 12 tests
def table_a12():
    t = json.load(open(R / "time_structure/summary.json"))
    bay = {r["test"]: r for r in json.load(open(R / "statistics/bayes_sensitivity.json"))["rows"]}
    rows = []
    for k, lab, _ in SETS:
        cells = [lab]
        e = t[f"{k}|R2|S2"]
        cells.append(cell2(f(e['point']), ci(*e['percentile'])))
        for test in ("T1", "T2", "T3"):
            e = t[f"{k}|{test}"]
            sup = bool(e["clear"] and e["point"] > 0 and e.get("p_holm", 1) < 0.05)
            mark = "\\textsuperscript{*}" if sup else ""
            cells.append(cell2(f(e['point']) + mark, ci(*e['percentile'])))
        b = bay.get(f"T3 {k}", {}).get("BF01")
        cells.append(bf(b))
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    write("a12_tests.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Which trials take longer, from the structure of the trial (Amendment 12, confirmatory). Held out share of within person variance in log time, ten folds of participants within each dataset, with 95 percent participant bootstrap intervals ($B = 1000$, refitted in every replicate). T1 is the structure law minus the set size law, T2 the score of a structure law fitted to the other seven datasets, T3 the gain from adding sample until return's expected looks to structure. An asterisk marks a test supported after Holm's correction across the eight datasets. BF$_{01}$ is the Bayes factor for no gain in T3.}
\label{tab:a12}
\begin{tabular}{lccccc}
\toprule
Dataset & Structure law & \makecell{T1, beyond\\set size} & \makecell{T2,\\transferred} & \makecell{T3, search\\looks added} & BF$_{01}$, T3 \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ Table, mixed model of time on structure
def table_lmm():
    m = json.load(open(R / "statistics/misc_stats.json"))["time_structure_lmm"]
    rows = []
    for k, lab, _ in SETS:
        r = m[k]
        u = r.get("percent_time_per_doubling_unadjusted")
        a = r.get("percent_time_per_doubling")
        cu = cell2(f"${u[0]:.1f}$", f"$[{u[1]:.1f}, {u[2]:.1f}]$") if u else "--"
        ca = cell2(f"${a[0]:.1f}$", f"$[{a[1]:.1f}, {a[2]:.1f}]$") if a else "--"
        tb = r["terms"].get("best", {}).get("std_beta_within")
        tg = r["terms"].get("gap", {}).get("std_beta_within")
        rows.append(" & ".join([lab, cu, ca, f(tb, 2) if tb is not None else "--", f(tg, 2) if tg is not None else "--",
                                f(r["R2_marginal"], 3, False), f(r["R2_conditional"], 3, False)]) + " \\\\")
    write("lmm_structure.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{How much time each feature of the trial adds, from a linear mixed model of log seconds on the four structure features with a random intercept and a random slope of $\log_2 n$ per participant (REML). Percent change in time per doubling of the set, without and with the value features in the model, with 95 percent Wald intervals. Standardized coefficients of the best value and the gap to the runner up are in standard deviations of within person log time per standard deviation of the feature. Marginal and conditional $R^2$ follow \citet{nakagawa2013}. The conditional $R^2$ includes how fast each person is. Features that do not vary within a dataset are left out (set size in the two and three option datasets, the best value in our study, the mean of the rest with two options).}
\label{tab:lmm}
\begin{tabular}{lcccccc}
\toprule
Dataset & \makecell{Time per\\doubling, \%} & \makecell{Adjusted for\\values, \%} & \makecell{Best value,\\standardized} & \makecell{Gap,\\standardized} & \makecell{$R^2$\\marginal} & \makecell{$R^2$\\conditional} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ Table, vault scores of the main models
MAIN = ["S1", "S2", "CL_entropy", "SR_looks", "TLS_looks", "TLS_tfit", "S2+TLS", "S2+SR", "S2+Wald_n",
        "LBA", "LBA_n_cal", "Wald_n", "Ceiling2"]


def failed_vault_fits():
    """Entries of the frozen vault run whose rising threshold fit failed (results/time_models_vault_corrected)."""
    c = json.load(open(R / "time_models_vault_corrected/corrections.json"))
    bad = set()
    for k, r in c.items():
        for m, v in r.items():
            if not v["original_converged"]:
                bad |= {(k, m), (k, m + "_cal")}
    return bad


SHORT_FIXED = {("snack", "LBA"), ("snack", "LBA_cal"), ("reutskaja", "LBA"), ("reutskaja", "LBA_cal"),
               ("snack", "MLBA_LBA"), ("snack", "MLBA_LBA_cal")}  # verification/check_race_starts.py


def vault_rows(models, ev_path="time_models_vault_corrected/vault_eval.json", dagger=None, suffix="", short=None):
    ev = json.load(open(R / ev_path))
    rows = []
    for m in models:
        cells = [NAME[m] + suffix]
        for k, _, _ in SETS:
            r = ev["sets"][k]["models"][m]
            mark = ""
            if m != "S2":
                if r["minus_S2"]["clear_above"]:
                    mark = "\\,$\\uparrow$"
                elif r["minus_S2"]["clear_below"]:
                    mark = "\\,$\\downarrow$"
            if dagger and (k, m) in dagger:
                mark += "\\textsuperscript{\\dag}"
            if short and (k, m) in short:
                mark += "\\textsuperscript{\\S}"
            cells.append(f(r["R2"]) + mark)
        cells.append("yes" if ev["verdict"][m]["satisfactory"] else "no")
        rows.append(" & ".join(cells) + " \\\\")
    return rows


def table_vault_main():
    head = " & ".join(["Model"] + [s[2] for s in SETS] + ["Satisfactory"])
    write("vault_main.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.6pt}
\caption{Held out $R^2$ of within person log time on the vault for the main models. Every model was fitted once on the participants outside the vault. Arrows mark a model clearly above or below the structure law (both 95 percent bootstrap intervals of the difference exclude zero). The structure law is clearly above the set size law in snacks at a glance, snacks in three seconds, two foods and three foods. Satisfactory follows the rule fixed before any fit. The rising threshold LBA race uses the corrected fit for snacks at a glance (Section~\ref{sec:refit}). Every model, with the frozen and the corrected fits, is in Table~\ref{tab:vaultall}.}
\label{tab:vaultmain}
\begin{tabular}{""" + NAMECOL % "3.4cm" + "c" * (len(SETS) + 1) + r"""}
\toprule
""" + head + r""" \\
\midrule
""" + "\n".join(vault_rows(MAIN)) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_vault_all():
    head = " & ".join(["Model"] + [s[2] for s in SETS] + ["Satisfactory"])
    write("vault_all.tex", r"""\begin{table}[p]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{Every frozen model on the vault, held out $R^2$ of within person log time. Arrows as in Table~\ref{tab:vaultmain}. A dagger marks a frozen fit that failed to converge, and a section sign a frozen fixed threshold fit that stopped short of its ten start optimum (by 75 and 1.2 for LBA at a glance and in three seconds, and by 72 for MLBA LBA at a glance), Section~\ref{sec:refit}. The last four rows give the final fits, which at three decimals differ from the frozen ones only where a dagger stands. In snacks in three seconds every race is conditioned on a response before the three second limit.}
\label{tab:vaultall}
\begin{tabular}{""" + NAMECOL % "3.9cm" + "c" * (len(SETS) + 1) + r"""}
\toprule
""" + head + r""" \\
\midrule
""" + "\n".join(vault_rows(ORDER, "time_models_vault/vault_eval.json", dagger=failed_vault_fits(), short=SHORT_FIXED)) + r"""
\midrule
\multicolumn{""" + str(len(SETS) + 2) + r"""}{l}{Corrected fits, refitted after the vault was opened} \\
\midrule
""" + "\n".join(vault_rows(["LBA_n", "LBA_n_cal", "MLBA_LBA_n", "MLBA_LBA_n_cal"], suffix=", corrected")) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ Table, contrasts with the structure law
def table_contrasts():
    ev = json.load(open(R / "time_models_vault_corrected/vault_eval.json"))["sets"]
    keys = ["snack", "reutskaja", "krajbich2010", "krajbich2011"]
    lab = {k: l for k, l, _ in SETS}
    rows = []
    for m in ["TLS_looks", "S2+TLS", "SR_looks", "LBA_n_cal", "Ceiling2"]:
        for i, k in enumerate(keys):
            r = ev[k]["models"][m]; d = r["minus_S2"]; b = r.get("BF01_vs_S2")
            bft = f"{bf(b['BF01'])} $[{b['half']:.1f}, {b['double']:.1f}]$" if b else "--"
            p = r["lmm"].get("p_holm")
            pt = "$<0.001$" if p is not None and p < 0.001 else (f"${p:.2f}$" if p is not None else "--")
            rows.append(" & ".join([NAME[m] if i == 0 else "", SHORT[k], f(d["point"]), ci(*d["percentile"]), pt,
                                    f(r["cohen_dz_vs_S2"], 2), f"${100 * r['share_people_better_than_S2']:.0f}$",
                                    bft]) + " \\\\")
        rows.append("\\addlinespace")
    write("vault_contrasts.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Each model against the structure law on the vault, in the four datasets where structure beats set size. Difference in held out $R^2$ with its 95 percent bootstrap interval. Holm $p$ is from a linear mixed model of the per trial difference in squared error with a random intercept per participant, corrected across the eight datasets. $d_z$ is Cohen's effect size of the per participant mean difference, and Better is the percentage of vault participants whose predictions improved. BF$_{01}$ is the Bayes factor for no gain over structure, at the prior scale of the development gain of structure over set size and, in brackets, at half and double that scale. It is not computed where a model is clearly above structure.}
\label{tab:contrasts}
\begin{tabular}{""" + NAMECOL % "3.0cm" + r"""lcccccc}
\toprule
Model & Dataset & Difference & 95\% interval & Holm $p$ & $d_z$ & \makecell{Better,\\\%} & \makecell{BF$_{01}$\\$[\tfrac12, 2]$} \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ Table, development rounds
def table_dev():
    d = json.load(open(R / "time_models_dev/dev_r2_all.json"))
    devfail = set()
    for k, _, _ in SETS:
        if k == "reutskaja":
            continue  # its round 2 fit records were overwritten by the time limit rerun
        e1 = json.load(open(R / f"time_models_dev/r1_{k}_summary.json"))["extras"]
        e2 = json.load(open(R / f"time_models_dev/r2_{k}_summary.json"))["extras"]
        for b in ("LBA", "MLBA_LBA", "Wald", "RWald"):
            if any(n["nll"] > f0["nll"] + 1.0 for f0, n in zip(e1[b], e2[b + "_n"])):
                devfail.add((k, b + "_n"))

    def mk(k, m, rnd):
        base = m.replace("_cal", "")
        if (k, base) in devfail:
            return "\\textsuperscript{\\dag}"
        if k == "reutskaja" and rnd == "2" and base in ("LBA_n", "MLBA_LBA_n", "Wald_n", "RWald_n"):
            return "\\textsuperscript{\\ddag}"
        return ""
    head = " & ".join(["Round", "Model"] + [s[2] for s in SETS])
    rows = []
    for rnd in ("1", "2", "3"):
        for m in ORDER:
            if m not in d[rnd]:
                continue
            vals = d[rnd][m]
            if rnd == "2" and m in ("LBA", "MLBA_LBA", "Wald", "RWald", "LBA_cal", "MLBA_LBA_cal", "Wald_cal", "RWald_cal"):
                label = NAME[m] + ", with the time limit"
            else:
                label = NAME[m]
            rows.append(" & ".join([rnd, label] + [f(vals[k]) + mk(k, m, rnd) if k in vals else "" for k, _, _ in SETS]) + " \\\\")
        rows.append("\\addlinespace")
    write("dev_rounds.tex", r"""\begin{table}[p]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\caption{Development scores, held out $R^2$ of within person log time over ten folds of the participants outside the vault. Round 2 reran the four fixed threshold races in snacks in three seconds conditioned on the three second limit. Each round was logged before the next was run. A dagger marks a rising threshold race whose fit failed to converge in at least one fold (its negative log likelihood stayed worse than the fixed threshold fit's by more than 1), and a double dagger one whose convergence could not be checked (Section~\ref{sec:refit}).}
\label{tab:dev}
\begin{tabular}{c""" + NAMECOL % "4.6cm" + "c" * len(SETS) + r"""}
\toprule
""" + head + r""" \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


# ------------------------------------------------------------------ Table, secondary measures
def table_secondary():
    p = R / "time_models_vault_corrected/secondary.json"
    if not p.exists():
        print("secondary.json missing, skipped")
        return
    s = json.load(open(p))
    models = ["S1", "S2", "SR_looks", "TLS_looks", "S2+TLS", "LBA", "Wald", "LBA_n_cal", "Wald_n", "Ceiling2"]
    head = " & ".join(["Model"] + [x[2] for x in SETS])
    rows = []
    for m in models:
        rows.append(" & ".join([NAME[m]] + [f(s[k]["rank_corr"][m]["mean"], 2) for k, _, _ in SETS]) + " \\\\")
    rank = "\n".join(rows)
    usets = [k for k, _, _ in SETS if s[k].get("unseen_size")]
    rows = []
    for m in models:
        cells = [NAME[m] + ("\\textsuperscript{\\dag}" if m == "LBA_n_cal" else "")]
        for k in usets:
            u = s[k]["unseen_size"][m]
            mark = "\\textsuperscript{\\ddag}" if (m in ("LBA", "LBA_n_cal") and k == "cdm") else ""
            cells.append(cell2(f(u['skill'], 2) + mark, f"$[{u['skill_ci'][0]:+.2f}, {u['skill_ci'][1]:+.2f}]$"))
        rows.append(" & ".join(cells) + " \\\\")
    lab = {k: l for k, _, l in SETS}
    unseen = "\n".join(rows)
    uhead = " & ".join(["Model"] + [lab[k] for k in usets])
    rows = []
    for m in ["CL", "SR", "LBA", "LBA_n", "MLBA_LBA", "MLBA_LBA_n", "Wald", "Wald_n", "RWald", "RWald_n"]:
        cells = [{"CL": "CL", "SR": "Sample until return and two look satisficing"}.get(m, NAME.get(m, m))]
        for k, _, _ in SETS:
            b = s[k]["bits"].get(m)
            cells.append(f(b["bits_per_choice"], 3, False) if b else "--")
        rows.append(" & ".join(cells) + " \\\\")
    bits = "\n".join(rows)
    write("secondary.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\caption{Secondary measures on the vault. (a) Mean within person rank correlation of predicted and observed time. (b) Bits per choice gained over a uniform guess on vault trials. Two look satisficing makes the same choices as sample until return.}
\label{tab:secondary}
\begin{tabular}{""" + NAMECOL % "4.4cm" + "c" * len(SETS) + r"""}
\toprule
\multicolumn{""" + str(len(SETS) + 1) + r"""}{l}{(a) Rank correlation with time} \\
\midrule
""" + head + r""" \\
\midrule
""" + rank + r"""
\midrule
\multicolumn{""" + str(len(SETS) + 1) + r"""}{l}{(b) Choices, bits per choice above a uniform guess} \\
\midrule
""" + bits + r"""
\bottomrule
\end{tabular}
\end{table}
""")
    write("unseen.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Forecasting the time cost of a set size the model never saw, on the vault, in datasets with three or more set sizes. Each set size is left out in turn and every model is refitted on the others. Skill is one minus the mean squared error of each vault participant's predicted difference between mean log time at the left out size and at the other sizes, relative to forecasting no difference, with a 95 percent bootstrap interval. Negative skill is worse than forecasting no difference. Race refits at each left out size use three starts. A dagger marks the rising threshold LBA race, whose refits only replace fits that fail the nesting check, without the fuller search of Section~\ref{sec:refit}. A double dagger marks the two LBA races in the sums. With set size 20 left out, the fixed LBA refit stopped far from its optimum (negative log likelihood 8{,}115 against 2{,}237 with ten starts), and the rising LBA refit (2{,}488) passed the nesting check only against that failed fixed fit.}
\label{tab:unseen}
\begin{tabular}{""" + NAMECOL % "4.4cm" + "c" * len(usets) + r"""}
\toprule
""" + uhead + r""" \\
\midrule
""" + unseen + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def table_convergence():
    c = json.load(open(R / "time_models_vault_corrected/corrections.json"))
    srch = json.load(open(R / "time_models_vault_corrected/lba_search.json"))
    num = lambda x: "$" + f"{x:,.0f}".replace(",", "{,}") + "$"
    rows = []
    for k, lab, _ in SETS:
        v, q = c[k]["LBA_n"], srch[k]
        fail_l = "\\textsuperscript{\\dag}" if v["nll_original"] > q["nll_fixed_ten_starts"] + 1.0 else ""
        w = c[k]["MLBA_LBA_n"]
        fail_m = "" if w["original_converged"] else "\\textsuperscript{\\dag}"
        final_m = w["nll_refit"] if w["kept"] == "refit" else w["nll_original"]
        rows.append(" & ".join([lab, num(q["nll_fixed_ten_starts"]), num(v["nll_original"]) + fail_l, num(q["nll"]),
                                f"${q['gamma']:.2f}$", num(w["nll_fixed"]), num(w["nll_original"]) + fail_m,
                                num(final_m)]) + " \\\\")
    write("convergence.tex", r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{Negative log likelihood of the vault fits of the two rising threshold races, on the training participants. Fixed is the nested fixed threshold race (ten starts for LBA, three for MLBA LBA), Frozen the fit of the frozen run and Final the fit used in this section. A rising threshold fit can never be worse than its fixed threshold version at a true optimum, so a frozen fit worse than the fixed one by more than 1 failed (dagger). For LBA, Final is the best admissible fit of the fuller search in every dataset, with its $\gamma$, which is not identified in the two and three option datasets because set size never changes there. For MLBA LBA, a frozen fit was replaced only where it failed, by a warm start refit that passed the check.}
\label{tab:convergence}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{4}{c}{LBA, rising threshold} & \multicolumn{3}{c}{MLBA LBA, rising threshold} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Dataset & Fixed & Frozen & Final & $\gamma$ & Fixed & Frozen & Final \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    table_a12(); table_lmm(); table_vault_main(); table_vault_all(); table_contrasts(); table_dev(); table_secondary()
    table_convergence()
