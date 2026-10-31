# Independent verification brief, forecasting choice overload (Amendment 8, confirmatory)

Same rules as always. Work only inside `verify/`, write code in `verify/independent_overload/` and results to `verify/independent_overload/results.json`. Never write participant, trial or selection counts or identifiers. You may reuse your own earlier code (`independent_cdm/` for the Caplin, Dean and Martin loader, the softmax fits and the race on dollar levels).

**Do not run anything on the real data until the coordinator says so.** Write the code, test it on simulated data (below), and report that it is ready. Amendments 7 and 8 are frozen together after both of your implementations are ready, and the real data are then analyzed once.

## Inputs

- `verify/cdm/ProcessedCP_All_Final.csv`, as in Amendment 7. Use exactly Amendment 7's included rounds, values in dollars at positions 2 to n + 1, the choice being the last selection.
- `verify/reutskaja/fixations_data.dta` (Stata, read with pandas.read_stata), one row per fixation. Trial fields are constant within (`subject`, `trialid`). Columns you need: `subject`, `trialid`, `numalt` (set size 4, 9 or 16), `rating1` to `rating16` (ratings from -5 to 5 in screen order, 0 past the set size), `chosendestring` (chosen position, 1 based), `Roinum` (fixated position, 1 based) and `fixOrder` (fixation order within the trial). Participants are the sorted `subject` values.

Section 4 of `ANALYSIS_PLAN.md` (intervals, p values, the clear rule) applies.

## The specification, Amendment 8 as it will be frozen

1. Data. The rounds of Amendment 7, values in dollars. The replication package of Reutskaja, Nagel, Camerer and Rangel (2011), openICPSR project 112415, version 1 (CC BY 4.0), downloaded by the study's author in October 2026, file `fixations_data.dta` (SHA-256 e98da9eb2a0b2d1e1472e84246dcd0bc641dfbc5c60746171cd087eedc599bee), every trial, one choice each, ratings from -5 to 5 in screen order (`rating1` to `rating16`, the set size `numalt`, the chosen position `chosendestring`) and fixations in order (`Roinum`, `fixOrder`). Panels share a value scale and task. Caplin, Dean and Martin by experiment and complexity (four panels) and Reutskaja one panel.
2. Outcome. A miss is a choice whose value is below the largest value in its set. Options tied at the top all count as the best.
3. Models, one sensitivity each. CL, softmax(lambda v). CL with compensation, softmax(lambda f(n) v) with f(n) = ((n - 1) / 7) ^ 0.18, the elasticity of sensitivity with set size found by H1 on the snack data and fixed here. Sample until return, the race on softmax(lambda v), with Reutskaja's ratings shifted by 6 so that every value is a whole level from 1 to 35 (the race does not depend on the shift). Each set size of a panel is forecast from fits to the panel's other set sizes, by weighted maximum likelihood, CL and CL with compensation by root finding, sample until return on the grid of Amendment 5 refined by bounded Brent search (absolute tolerance 1e-10). A model's probability of a miss is one minus its probabilities of the options worth the largest value.
4. Resampling. B = 1000. Replicate b draws participants with replacement from numpy.random.default_rng(80000 + b), Caplin, Dean and Martin's Experiment 1 first, then their Experiment 2, then Reutskaja, each with integers(0, P, size=P) over its sorted participants. Every trial is weighted by its participant's draw count and every model is refitted in every fold of every replicate. No replicate leaves a set size of a panel without trials. Intervals, p values and the clear rule follow Section 4.
5. Measures, per panel, then the mean over a dataset's panels.
   - Growth. The weighted mean of the misses at each set size, and of each model's held out probability of a miss. Growth is the least squares slope of these rates on log2 of the set size, and a model's error is the absolute difference between its growth and people's.
   - Premonition. Within each set size, the weighted area under the curve of a forecast for the misses (the probability that a trial with a miss outscores one without, ties counting one half, trials weighted by their participants' draw counts), averaged over set sizes with their summed weights (a set size without a miss or without a hit is skipped). The models' probabilities are rounded to ten significant digits (Python's format .10g) before ranking, so that displays equal in exact arithmetic tie. Forecasts are the three models' probabilities of a miss and two display features, the number of options worth less than the largest value but no more than one dollar or one rating point below it, and minus the gap between the largest value and the largest value below it (an infinite gap when every option ties at the top).
6. Primary tests, for Caplin, Dean and Martin and for Reutskaja separately.
   - O1, growth. People's growth minus CL with compensation's, and people's growth minus CL's. Overload when the first is clearly positive (misses grow more than compensation allows), compensation when the second is clearly negative (misses grow less than fixed sensitivity foresees), partial compensation when both hold, otherwise undecided. Overload is predicted in both datasets.
   - O2, premonition. Sample until return's area under the curve minus the near best count's, supported when clearly positive, that is when the model picks out the trials with a miss better than the simplest display feature. Predicted in both datasets.
7. Secondary and descriptive. Each model's growth error and compensation's error minus CL's, sample until return's area under the curve minus one half, people's growth and each model's forecast growth with their intervals, every area under the curve, the rates by set size with percentile intervals, and in Reutskaja's data the share of misses whose best option was never fixated, by set size. The same code on the snack data, run before this amendment, is reported as exploratory.
8. Reporting. O1 and O2 in the two datasets are confirmatory and item 7 descriptive. Each set size's forecast extrapolates from fits to the panel's two other set sizes, and the four Caplin, Dean and Martin panels weigh equally whatever their size, as said with the results. They are reported without a correction for multiplicity. Any deviation from this amendment found after the run is reported as a deviation.

## Conventions to keep the two implementations comparable

- Panels, in this order and with these names: `cdm_standard_3`, `cdm_standard_7`, `cdm_process_3`, `cdm_process_7` (experiment and complexity), `reutskaja`. The dataset `cdm` pools its four panels with equal weight, `reutskaja` is its one panel.
- Bootstrap groups, drawn in this order from one generator per replicate: Caplin, Dean and Martin Experiment 1 participants (sorted), Experiment 2 participants (sorted), Reutskaja subjects (sorted). A participant's draw count weights every trial of theirs in every panel.
- Folds leave one set size out within a panel, the sensitivities fitted on the panel's other set sizes with the draw count weights.
- Rates by set size are weighted means over the panel's trials of that size. Growth is numpy.polyfit(log2(sizes), rates, 1) slope.
- The area under the curve is computed within each set size over trials with positive weight, skipped when a set size has no miss or no hit, and averaged over set sizes with weights equal to the summed trial weights of the set size.
- The near best count counts options with value v where top - 1 <= v < top. The gap feature is -(top - max of values strictly below top), and minus infinity when every option ties at the top.
- Sample until return's probability of choosing the best is the number of options at the top value times the race probability of one option at that value.
- comp_minus_cl_error is the summary of error(CL_comp) - error(CL), replicate by replicate. observed_minus_comp (people's growth minus CL_comp's), observed_minus_cl, sr_auc_minus_half and sr_auc_minus_near_best likewise.
- O1 verdict strings: "overload", "compensation", "partial compensation", "undecided".
- Reutskaja's share of misses whose best option was never fixated uses trials with a miss, the best being any option at the top value, fixated meaning any fixation's `Roinum` points at it.

## Output

`independent_overload/results.json`:

```
{
  "datasets": {
    "cdm": DS, "reutskaja": DS
  },
  "panels": {
    "<panel name>": {"sizes": [3], "rate": {"observed": [3], "CL": [3], "CL_comp": [3], "SR": [3]},
                      "lambda": {"CL": [3], "CL_comp": [3], "SR": [3]},
                      "miss_best_unseen": [3] (reutskaja only)}
  }
}
DS = {
  "growth": {"observed": S, "CL": S, "CL_comp": S, "SR": S},
  "error": {"CL": S, "CL_comp": S, "SR": S},
  "auc": {"CL": S, "CL_comp": S, "SR": S, "near_best": S, "minus_gap": S},
  "comp_minus_cl_error": S, "observed_minus_comp": S, "observed_minus_cl": S,
  "sr_auc_minus_half": S, "sr_auc_minus_near_best": S,
  "O1": {"verdict": "overload" | "compensation" | "partial compensation" | "undecided"}, "O2": {"supported": bool}
}
```

S is {"point", "percentile", "basic", "p", "clear"}. Point values are the original sample. Lambdas are by left out set size, in increasing order of set size.

## Tests on simulated data before anything else

1. The miss probabilities of the three models against a direct computation for small sets, and sample until return's against a simulation of the race.
2. Recovery of each sensitivity from choices simulated at a known value, with compensation for CL_comp.
3. Synthetic data in both input formats (a click file and a Stata file built to the columns above), with choices simulated from CL with fixed sensitivity and from CL with compensation. O1 should call the first overload (or be undecided) and never compensation, and the second compensation (or undecided) and never overload. O2 (sample until return against the near best count) should be supported in both.
4. The weighted area under the curve against a brute force pairwise count with ties.

## Report back

First, when the code is ready and the simulated checks pass, every ambiguity you resolved and any concern about the amendment. Later, when told to run, how long it ran, the headline values and decisions of O1 and O2 in both datasets, and anything that surprised you.
