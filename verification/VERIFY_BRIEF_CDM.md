# Independent verification brief, the search model's stopping rule on the data of Caplin, Dean and Martin (Amendment 7, confirmatory)

Same rules as always. Work only inside `verify/`, do not look at any other folder of the scratchpad, write code in `verify/independent_cdm/` and results to `verify/independent_cdm/results.json`. Never write participant counts, trial or selection counts, or identifiers (the click file holds a participant id, never write it out). You may reuse your own earlier code (`independent_search/`, `independent_two_look/`, `independent_gaze/` for the race formulas, the time law, the summaries and the Section 4 conventions).

**Do not run anything on the real data until the coordinator says so.** Write the code, test it on simulated data (below), and report that it is ready. The amendment is frozen only after the code on both sides is ready, and the real data are then analyzed once. Until then, open the real click file only as far as loading code needs (column names), and compute nothing on it.

## Inputs

In `verify/cdm/`: `ProcessedCP_All_Final.csv` (one row per selection, a click, of every round of every experiment), its data dictionary `ProcessedCP_DataDictionary_Final.xls` (column names are lower case in the file) and `LICENSE.txt`. Columns you need: `tracking` (0 standard choice, Experiment 1, 1 choice process, Experiment 2), `treatid`, `userid`, `rndnum` (with `userid` and `tracking` it identifies a round), `rndcode`, `practice`, `submit`, `numopts` (n), `rndmaxnumoper` (complexity), `slctnum` and `numslct` (a selection's order and the round's number of selections), `timeslct` (seconds), `valnum` (the selection's value), `optord` (its position in the list, 2 to n + 1) and `val1` to `val41` (the values in list order, `val1` the worthless default at position 1).

Section 4 of `ANALYSIS_PLAN.md` (intervals, p values, the clear rule) and Amendments 5 and 6 (fits, time law, summaries) apply where the amendment points to them.

## The specification, Amendment 7 as it will be frozen

1. Data. The replication package of Caplin, Dean and Martin (2011), openICPSR project 116100, version 1 (CC BY 4.0), downloaded by the study's author in October 2026. The click file `ProcessedCP_All_Final.csv` (SHA-256 86adf9e6331192a06765224b41cc4e10d263a85e624cee4da141e684524f7d9e) is checked and copied by `scripts/prepare_cdm_data.sh`. Included rounds follow the authors' standard data set (their `Tables.do`). Sets of 10 options with round codes 1 to 6 and sets of 20 and 40 options with a round code, at most 3 or 7 operations per option, practice rounds left out, Experiment 3 (at most 9 operations) and Experiment 4 (treatments 21, 24 and 25) left out, and only rounds that were submitted. Experiment 1 (standard choice, tracking 0) and Experiment 2 (choice process, tracking 1) are analyzed separately. The authors dropped one participant as noncompliant without a stated rule, so no participant is excluded here. Positions 2 to n + 1 of a round's list hold its n options (position 1 is the worthless default), the choice is the last selection, and a selection's time is in seconds since the round began.
2. Models. CL, DbS and MLBA choose with softmax(lambda s), s the values in dollars for CL and the single attribute scores of Amendment 3 for DbS and MLBA (kappa = 1, MLBA on the values times 6 / 35, so that the dollar scale spans six units as the snack ratings did). Sample until return (SR) chooses with the race probabilities of softmax(lambda v) and takes E_SR(softmax(lambda v)) looks, and its attention lands on option j with probability softmax(lambda v)_j. One sensitivity per model and fold, fitted by weighted maximum likelihood on the training participants of the experiment, CL, DbS and MLBA by root finding on their concave likelihoods, SR on the grid of Amendment 5 refined by bounded Brent search (absolute tolerance 1e-10). Probabilities of a chosen option are floored at 1e-300 inside the log.
3. Folds and resampling. Ten folds of participants within each experiment. The experiment's sorted participants are permuted by numpy.random.default_rng(20261015).permutation, and a participant's fold is its position in that permutation modulo 10. B = 1000. Replicate b draws participants with replacement within each experiment from numpy.random.default_rng(70000 + b), Experiment 1 first, each with integers(0, P, size=P) over its sorted participants. Rounds and selections are weighted by their participant's draw count, a participant keeps its fold, and everything is refitted in every fold of every replicate. Intervals, p values and the clear rule follow Section 4.
4. Measures, in each experiment.
   - Stopping. Every selection that is not worth as much as the set's best option is scored on whether it was the last selection of its round (the best option ends any search that keeps the best option found, so it tells nothing). Two rules are fitted on the training participants' selections, separately in each of the six cells of set size and complexity. The value rule predicts the last selection from the selection's value in dollars, the share rule from the share of the set's options worth more than the selection. Each rule is a logistic regression with an intercept and its predictor standardized by the weighted mean and standard deviation of the training selections, with a standard normal prior on the standardized slope, fitted by Newton's method. When every training selection of a cell has the same outcome, the fit takes its limit and predicts that outcome with probability one, so the two rules predict alike there. A rule's score is the weighted mean, over every held out selection of the experiment, of the log2 probability it gave the observed outcome, floored at 1e-300. Also reported, both rules with the log of the selection's time added as a second standardized predictor.
   - First selection. The place of the round's first selection in the list, its position among the n options counted from 0 at the top and divided by n - 1, averaged over rounds with participant weights. The model's prediction is the mean place of its attention, sum over j of softmax(lambda v)_j j / (n - 1), at the sensitivity of the fold that holds the round out.
   - Choices. L is the weighted mean of log2(n q_chosen) over rounds, q from the fold that holds the round out.
   - Descriptive. The weighted geometric mean time of the final selection by set size and its elasticity with set size (the least squares slope of its log on log n, averaged over the two complexities), next to the same for SR's expected looks. The held out error of the time law of Amendment 6 for the time of the final selection, with effort n for the static models and SR's expected looks.
5. Primary tests.
   - C1, stopping, on the mean of the two experiments. The value rule's score minus the share rule's. When it is clearly positive, the worth of a selection told whether search ended better than the better options still to be found, as an absolute standard implies and the model's first return does not. When it is clearly negative, stopping followed how many better options remained, as the model's first return implies. Otherwise C1 is undecided.
   - C2, order of search, in Experiment 2. Holding the worthless default paid nothing there, so the first selection is the first option worked out. When the first selection's place minus one half (the expectation for any reading order blind to the screen) and its place minus the model's prediction are both clearly negative, search followed the order of the list, as the model's attention does not.
   - C3, choices, on the mean of the two experiments. SR minus CL as 100 (2^dL - 1) percent, supported (non inferior) when the lower bounds of both the percentile and the basic interval lie above -2 percent. Better and worse are reported as in S5 of Amendment 5.
6. Secondary and descriptive. Every measure by experiment, SR minus DbS and minus MLBA, the stopping rules with time added, the first selection in Experiment 1, the growth of search and the time laws, with no direction tested.
7. Reporting. C1 to C3 are confirmatory and item 6 descriptive. The original paper makes the outcomes of C1 and C2 likely, and this is said with the results. Any deviation from this amendment found after the run is reported as a deviation.

## Conventions to keep the two implementations comparable

- Pooled values are the means over the two experiments of the per experiment values, replicate by replicate (stopping scores, growth, time errors and L alike). The pooled dL is the mean of the two per experiment dL, and the pooled times chance is 2 to the pooled L.
- Weighted standard deviation in the logistic regression is the population one, sqrt(sum w (x - mean)^2 / sum w). The penalty is 0.5 times the sum of the squared standardized slopes (the intercept is not penalized), added to the negative weighted log likelihood. Fit to convergence (the objective is strictly concave).
- An option counts as worth more than a selection when its value is strictly larger. Selections with no option worth more are left out of both fitting and scoring.
- A rule's score uses held out selections of every cell together, weighted by their participants' draw counts, so cells weigh by their selections.
- The model's predicted first place uses SR's sensitivity of the fold that holds the round out, attention softmax(lambda v) over the round's n options.
- SR_minus_X_percent is the summary of dL = L_SR - L_X with point and both intervals transformed by 100 (2^x - 1), p and clear kept. For C3, non inferior when min(percentile lower, basic lower) > -2, better when both lower bounds are above zero, worse when both upper bounds are below zero.
- Elasticities. For each complexity, the least squares slope of the log of the weighted geometric mean (final selection time) or of the log of the weighted geometric mean of SR's expected looks, over the three set sizes against log n, then the mean of the two slopes. one_minus_observed is 1 minus the observed elasticity and predicted_minus_observed their difference, replicate by replicate.
- observed_by_size is the weighted geometric mean final selection time at each set size (complexities together), looks_by_size the weighted arithmetic mean of SR's held out expected looks, original sample only.
- The time law is that of Amendment 6 (t0 >= 1e-9, tau >= 0, global minimum of the weighted squared error of log seconds), fitted on the training rounds and applied to the held out ones, with effort n (static) or SR's expected looks at the fold's sensitivity. SR_minus_static is the difference of the two errors.

## Output

`independent_cdm/results.json`:

```
{
  "pooled": BLOCK + {"C1": {"absolute": bool, "relative": bool}, "C2": {"order_matters": bool},
                     "C3": {"margin": 2.0, "noninferior": bool, "better": bool, "worse": bool, "supported": bool}},
  "standard": BLOCK + EXPERIMENT,
  "process": BLOCK + EXPERIMENT
}
BLOCK = {
  "stop": {"value": S, "share": S, "value_time": S, "share_time": S, "value_minus_share": S, "value_minus_share_with_time": S},
  "growth": {"observed": S, "predicted": S, "one_minus_observed": S, "predicted_minus_observed": S},
  "choices": {"times_chance": {"CL": x, "DbS": x, "MLBA": x, "SR": x},
              "SR_minus_CL_percent": S, "SR_minus_DbS_percent": S, "SR_minus_MLBA_percent": S},
  "time": {"static": S, "SR": S, "SR_minus_static": S}
}
EXPERIMENT = {
  "first": {"observed": S, "predicted": S, "minus_half": S, "minus_predicted": S},
  "lambda": {"CL": [10], "DbS": [10], "MLBA": [10], "SR": [10]},
  and inside "growth": "observed_by_size": [3], "looks_by_size": [3]
}
```

S is {"point", "percentile", "basic", "p", "clear"}. C2 is decided on "process" (Experiment 2). The original sample is the point estimate, lambdas by fold 0 to 9.

## Tests on simulated data before anything else

1. Your race formulas on dollar valued sets of 10, 20 and 40 options against your earlier code and a simulation of the process, and the DbS and MLBA scores for sets of different sizes against your snack code.
2. Recovery of every sensitivity from choices simulated at a known value.
3. The logistic fit is the maximum of its penalized likelihood, and its score is finite under complete separation.
4. Synthetic click files in the input format, built from simulated searches, at least these two. (a) Satisficing in list order, with a reservation amount per cell and noisy evaluation, where C1 should come out absolute and C2 should find the list order. (b) A search that reads options at random with replacement, keeps the best option found as its selection and stops at its first return to an option already read, where C1 should not come out absolute and C2 should not find the list order. In both, the selection made at the first option worked out is the first selection.
5. Every fitted sensitivity is a maximum of its likelihood and every time law a global minimum.

## Report back

First, when the code is ready and the simulated checks pass, every ambiguity you resolved and any concern about the amendment. Later, when told to run, how long it ran, the headline values and decisions of C1 to C3, the descriptive values and anything that surprised you.
