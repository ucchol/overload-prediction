# Independent verification brief

You are re-deriving the results of a pre-registered analysis from its specification, without the original code, so that errors in the original implementation can be caught. Independence is the point of the exercise.

## Rules

- Work only inside this folder (`verify/`). Do not open, list or search any other folder of the scratchpad, in particular `overload-prediction/` and `run/`. You have no reference results and must not look for any.
- Write your code in `verify/independent/` and your results to `verify/independent/results.json`.
- The data file holds personal identifiers. Never print, copy or write any column whose name contains "Prolific", and never write the number of participants or any count of choices in any file or message. Report shares, log scores and completeness values only.
- Python 3 with numpy, scipy and pandas is available. Use them. Write clear code from scratch.

## Inputs

- `Updated_Choice_Overload.csv`, the raw responses, one row per participant.
- `stimuli/file_trials.json` and `stimuli/laptop_trials.json`. The six trials of each task in order (trial index 0 to 5 follows the JSON order). Each trial gives the response column, the information level, the intended number of options, the rubric key and the displayed options with their raw attributes. The options listed are exactly the options displayed (the last laptop trial lists 30).
- `stimuli/rubric.json`, the rubric score U of every option, keyed by task (`file`, `laptop`), then the trial's `rubric_key`, then the option name.
- `coding_definition.py`, the coding of raw attributes into ordered levels, higher meaning better. Call `code_file_trial(options)` or `code_laptop_trial(options)` with a trial's options dictionary; each returns an array of options by attributes, with file attributes (extension, owner, modified, permissions, size) and laptop attributes (memory, graphics, storage, processor, battery, brand, price). This is the data definition and the only code you are given. Use it as it is, but tell us if any coding rule looks wrong against the stimuli.
- `ANALYSIS_PLAN.md`, the specification. You need Sections 2 to 5, Section 7 for R0 and H1 only, and Amendments 1 and 2.

## Details the plan leaves implicit

1. Choices. A response matches an option when the response text, stripped of surrounding whitespace and lowercased, equals the option name lowercased. Non-string and unmatched responses are dropped. A task's panel is the list of respondents with at least one matched choice in that task, in the order of the data file. Choice counts of a trial count every matched response.
2. Rubric shares r. Stack the six trials of a task. Within each trial, center U and center each coded attribute column. Fit `scipy.optimize.nnls` of centered U on the centered attributes, using the first 5 file attributes and the first 6 laptop attributes (laptop price is left out and gets share 0), and normalize the weights to sum to one.
3. Scores follow the table in Section 3 of the plan, with x_ik the coded levels and r the shares (price share 0, so price never matters). The number of options n is the number displayed.
4. Fitting. Maximum likelihood of the single parameter on the pooled choices of the training trials, softmax sensitivity lambda >= 0 (lambda = 0 when the derivative of the log likelihood at 0 is not positive), EBA lapse epsilon in [0, 1]. Find the optimum to high precision (relative tolerance 1e-10 or better).
5. Held out: trial c is predicted with the parameter fitted to the other five trials of its task. Fitted: one parameter fitted to all six trials.
6. Completeness subsets: all six trials, the three low information trials, the three high information trials, and each intended set size (8, 16, 32, two trials each). The last laptop trial counts as a 32 option trial for grouping, but its n is 30 everywhere in the formulas.
7. Bootstrap. B = 1000 replicates. Replicate b uses `rng = numpy.random.default_rng(10000 + b)`, draws the file panel indices with `rng.integers(0, P_file, size=P_file)` and then the laptop panel indices with `rng.integers(0, P_laptop, size=P_laptop)` from the same generator, where P_task is the length of that task's panel. A respondent drawn k times contributes their choices k times. Every model is refitted in every fold of every replicate. The benchmark L* of each trial stays at its full sample value in every replicate (Amendment 1).
8. Summaries of a replicate distribution v with point estimate t. Percentile interval `numpy.percentile(v, [2.5, 97.5])` with the default linear rule. Basic interval [2t - hi, 2t - lo]. Clear when both intervals exclude zero on the same side. p = min(1, 2 min(mean(v <= 0), mean(v >= 0))). Holm within the two primary tests of each task.
9. B1 and Amendment 2. The sampled effective number E_samp(q) at N choices uses `rng = numpy.random.default_rng((role, task_index, trial_index, b + 1))`, with role 1 for a model prediction and 2 for observed shares, task_index 0 for files and 1 for laptops, b = -1 for the original sample, then `s = rng.multinomial(N, q / q.sum(), size=2000) / N`, the effective number exp(-sum s ln s over s > 0) of every draw, and their mean. N is the number of matched choices of that trial in that sample (original or replicate). q is the held out CL prediction (beta = 0) of that sample. The slope is the least squares slope of G_c on log2 n_c over the six trials, n_c displayed. For the centered intervals, the percentile interval is the point estimate plus the 2.5 and 97.5 percentiles of the errors, the basic interval is the point estimate minus them (upper becomes lower), and p uses the values point estimate plus error in the formula of item 8.
10. A1b groups trials by intended set size. Delta_c uses held out log scores in bits of EBA and of CL at beta = 0.

## What to compute

Both tasks, everything below.

R0, for QR, CL, DbS_avg, MLBA_avg, DbS_sum, MLBA_sum:
- observed shares of every option of every trial (as fractions),
- rubric shares r,
- L* per trial,
- held out and fitted log score per trial (bits), pooled lambda, held out overlap per trial,
- held out completeness overall, low, high; fitted completeness overall; held out overlap overall (mean of trials),
- bootstrap: each model's held out completeness (point, percentile, basic), and each model's difference from CL in held out completeness and in held out overlap (point, percentile, basic, p, clear),
- the held out completeness of CL, DbS_avg and MLBA_avg in replicates 0, 1, 499 and 999.

H1, for CL, DbS_avg and MLBA_avg at beta in {-1, -0.5, 0, 0.5, 1}, and EBA (median aspects) and EBA_best:
- A1a: held out completeness (point, percentile, basic), completeness at 8, 16 and 32 options, held out overlap, and for beta != 0 the difference from beta = 0 in completeness (point, percentile, basic, p, clear) and in overlap (point only), the two primary decisions with Holm p, and whether A1a supports H1 by the rule in Section 7 and Amendment 2,
- A1b: for EBA and EBA_best, the fitted lapse, held out completeness, completeness minus CL (point, percentile, basic, p, clear), Delta in bits at 8, 16 and 32 options, the 32 minus 8 contrast (point, percentile, basic, p, clear),
- B1: per trial the observed effective share E(p)/n, the model effective share E_samp(q)/n and the gap, then the mean gap and the slope per doubling, each with point, centered percentile and basic intervals, p and clear, and the B1 decision.

## Output

`independent/results.json` with this layout (numbers as plain floats, completeness in percent, overlap and shares as fractions, log scores in bits):

```
{
  "file": {
    "rubric_shares": {"extension": ..., ...},
    "shares": {"T1_L8": {"option name": share, ...}, ...},
    "Lstar": {"T1_L8": ..., ...},
    "R0": {
      "CL": {"lambda": ..., "ho_bits": {trial: ...}, "fit_bits": {trial: ...}, "overlap_ho": {trial: ...},
             "completeness_ho": {"point": ..., "percentile": [lo, hi], "basic": [lo, hi]},
             "completeness_ho_low": ..., "completeness_ho_high": ..., "completeness_fit": ..., "overlap_ho_mean": ...,
             "minus_CL": {"point": ..., "percentile": [...], "basic": [...], "p": ..., "clear": true},
             "overlap_minus_CL": {"point": ..., "percentile": [...], "basic": [...], "p": ..., "clear": false},
             "replicates": {"0": ..., "1": ..., "499": ..., "999": ...}},
      "QR": {...}, ...
    },
    "H1": {
      "A1a": {"CL|beta=+0.5": {"completeness": {...}, "by_size": {"8": ..., "16": ..., "32": ...}, "overlap": ...,
                               "minus_beta0": {...}, "overlap_minus_beta0": ...}, ...,
              "primary": {"CL|beta=+0.5": {"p": ..., "p_holm": ..., "supports_H1": ...}, "CL|beta=-0.5": {...}},
              "supported": false},
      "A1b": {"EBA": {"lapse": ..., "completeness": {...}, "minus_CL": {...}, "delta_bits_by_size": {...},
                      "large_minus_small_bits": {...}}, "EBA_best": {...}, "supported": ...},
      "B1": {"per_trial": {trial: {"observed_share": ..., "model_share": ..., "gap": ...}},
             "mean_gap": {...}, "slope": {...}, "supported": ...}
    }
  },
  "laptop": {...}
}
```

Model names in A1a are `CL|beta=-1.0`, `CL|beta=-0.5`, `CL|beta=+0.0`, `CL|beta=+0.5`, `CL|beta=+1.0`, and the same for `DbS_avg` and `MLBA_avg`.

## Report back

A short message with: whether everything ran and how long it took, every ambiguity in the specification you had to resolve and how you resolved it, anything in the coding definition that looks wrong against the stimuli, and any statistical concern about the specification itself (for example about Amendments 1 and 2). Do not include participant counts in the message.
