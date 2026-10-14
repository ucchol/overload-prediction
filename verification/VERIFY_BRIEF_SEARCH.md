# Independent verification brief, the search model on the snack data (Amendment 5)

Same rules as before. Work only inside `verify/`, do not look at any other folder of the scratchpad, write code in `verify/independent_search/` and results to `verify/independent_search/results.json`. Never write participant counts or trial counts anywhere. Report shares, means, correlations, log scores and estimates only.

**Do not run anything on the real snack data until the coordinator says so.** Write the code, test it on simulated data (below), and report that it is ready. The amendment is frozen only after the code on both sides is ready, and the real data are then analyzed once.

## Inputs

- `snack/setsize-{9,16,25,36}_desc-data.csv` as in `VERIFY_BRIEF_SNACK.md` (your `independent_snack/` code reads them). Additional column used here, `rt`, the response time in milliseconds.
- `ANALYSIS_PLAN.md`. The specification will be Amendment 5. Its text is reproduced in the section "Amendment 5" below, and it builds on Amendment 3 and on `VERIFY_BRIEF_SNACK.md` (trial inclusion, participants, scores, held out folds, weights, summaries).

## The search model, sample until return (SR)

Attention looks at item i with probability q_i = softmax(lambda v)_i at every step, with replacement, where v are the ratings. The first item looked at a second time is chosen.

- P(i) = q_i^2 times the integral over t from 0 to infinity of t e^(-t) prod over j != i of (1 + q_j t) dt, normalized to sum to one.
- E[looks] = integral of e^(-t) prod over all j of (1 + q_j t) dt. Every look before the last falls on a new item, so the expected number of distinct items looked at is E[looks] - 1.

Compute these any way you like (an exact sum of elementary symmetric polynomials, P(i) = q_i^2 sum_k (k + 1)! e_k(q without i) and E[looks] = sum_k k! e_k(q), is one independent route). The likelihood need not be concave in lambda, so find its global maximum over lambda >= 0, to a relative precision of 1e-8 or better.

## Amendment 5

1. Data and inclusion. As in Amendment 3, items 1 and 2. Response time is `rt`, in milliseconds. The distinct items looked at in a trial are the displayed items with positive cumulative gaze, as in Amendment 3. A trial enters the time analyses (S2, S3 and the time per step) when its response time is positive and finite.
2. Models. CL, DbS_avg and MLBA_avg over all items with beta = 0, as in Amendment 3, and SR with the rating as the value v. One sensitivity each, fitted by maximum likelihood on the pooled weighted trials of three set sizes and used for the fourth (leave one set size out).
3. Model quantities, each from the sensitivity of the fold in which the trial's set size is held out. SR's E[looks] and its expected distinct items E[looks] - 1. Each model's predicted probability that the chosen item has the highest rating of the display (the sum of the predicted probabilities of the items that share the highest rating). The entropy in bits of each static model's predicted choice distribution. For the time law of fold k, SR's E[looks] of every trial of every set size is computed with fold k's sensitivity.
4. Resampling. B = 1000, seeds 40000 + b, participants drawn as in Amendment 3 item 4 (`numpy.random.default_rng(40000 + b).integers(0, P, size=P)` over the sorted participants). Every trial is weighted by the number of times its participant was drawn, and every model is refitted in every fold of every replicate.
5. Primary tests.
   - S1. Observed elasticity, the least squares slope of log(weighted mean distinct items looked at per set size) on log n over the four set sizes, all included trials. Predicted elasticity, the same slope for SR's E[looks] - 1. Supported when one minus the observed elasticity is clear and positive and the percentile interval of predicted minus observed elasticity lies within plus or minus 0.1.
   - S2. Log response time in seconds is predicted as log(t0 + tau effort), with t0 >= 1e-9 and tau >= 0 minimizing the weighted sum of squared differences of log response times over the timed trials of three set sizes, applied to the fourth. Fold error, the weighted mean absolute difference between observed and predicted log response time over the timed trials of the held out set size. Model error, the mean over the four folds. Effort, n for CL, n (n - 1) for DbS_avg and MLBA_avg, E[looks] for SR. Supported when SR's error minus CL's and SR's error minus DbS_avg's are both clear and negative.
   - S3. For every participant and set size with at least three timed trials, the Spearman correlation (average ranks for ties, undefined and skipped when either variable is constant) across those trials between a forecast and the response time. Average over the participant's set sizes, then over participants, each participant weighted by the number of times it was drawn (participants not drawn drop out). Forecasts, SR's E[looks] and CL's entropy. Supported when SR's mean correlation is clear and positive. Also report SR's minus CL's.
   - S4. For each model and set size, the weighted mean over trials of y log2 p + (1 - y) log2(1 - p), with y = 1 when the chosen item has the highest rating of its display, p the model's held out probability of that event, clipped to [1e-15, 1 - 1e-15]. The constant forecast uses the weighted base rate of y over the trials of the other three set sizes. A model's score is the mean over the four set sizes. Report every model's score, every model minus the constant, and SR minus CL.
   - S5 (added 14 October, before any run). Choices, SR against CL. dL = L_SR - L_CL with L as in item 6, reported as 100 (2^dL - 1) percent (point, percentile, basic, p, clear, all transformed from dL). Non inferior when the lower bounds of both the percentile and the basic interval exceed -2 percent. Better when both lower bounds exceed 0, worse when both upper bounds are below 0. Supported when non inferior.
6. Secondary.
   - Choices. L of a model is the mean over set sizes of the weighted mean of log2(n q_chosen) over the held out set size's trials. Report 2^L (with the percentile interval of L transformed), the four set size values as 2^(by size), and SR minus each static model as 100 (2^dL - 1) percent (point, percentile, basic, p, clear, all transformed from dL).
   - Implied elasticity. For each set size, the CL sensitivity that maximizes the weighted expected log likelihood sum over trials and items of P_SR(i) log softmax(lambda v)_i, with P_SR the held out SR probabilities of that set size. The slope of log sensitivity on log n.
   - Per step time. Median response time of the timed trials per set size (unweighted, original sample only) divided by the mean effort per set size, for CL (n), DbS_avg (n (n - 1)) and SR (mean held out E[looks]).
   - Spearman correlations with response time as in S3 for DbS_avg's and MLBA_avg's entropy and for three display features, the highest rating, the standard deviation of the ratings (population, ddof 0) and the number of items sharing the highest rating.

Summaries as in `VERIFY_BRIEF_SNACK.md` item 9 (percentile, basic, p, clear).

## Tests on simulated data before anything else

1. SR probabilities and E[looks] against the closed forms for two items (P(1) = q1^2 (1 + 2 q2), E[looks] = 2 + 2 q1 q2) and against the birthday problem for equal ratings.
2. A simulation of the SR process (draw looks until the first repeat) against your probabilities and E[looks].
3. Parameter recovery. Simulate choices from SR at a known lambda on random displays of 9 to 36 items with ratings from -3 to 3, and recover lambda.
4. A synthetic data set in which choices, distinct items looked at and response times come from the SR process (for example rt = 1000 (1 + 0.4 looks) times lognormal noise). Your pipeline should recover lambda, a predicted search elasticity close to the observed one, a time error lower for SR than for CL and DbS_avg, and a positive S3 correlation for SR.
5. Every fitted lambda is a maximum of its likelihood.

## Output

`independent_search/results.json`:

```
{
  "lambda": {"CL": [4], "DbS_avg": [4], "MLBA_avg": [4], "SR": [4]},
  "choices": {"times_chance": {MODEL: {"point", "percentile", "by_size": [4]}},
              "SR_minus_percent": {"CL": S, "DbS_avg": S, "MLBA_avg": S}},
  "S1": {"observed_by_size": [4], "predicted_by_size": [4], "observed_over_predicted_by_size": [4],
         "observed_elasticity": S, "predicted_elasticity": S,
         "one_minus_observed": S, "predicted_minus_observed": S, "within_margin": bool, "supported": bool},
  "S2": {"error": {MODEL: S plus "by_size": [4]}, "SR_minus": {"CL": S, "DbS_avg": S, "MLBA_avg": S}, "supported": bool},
  "S3": {"correlation": {"looks_SR", "entropy_CL", "entropy_DbS_avg", "entropy_MLBA_avg", "top_rating", "rating_sd",
                         "top_count": S}, "SR_minus_CL": S, "supported": bool},
  "S4": {"score": {MODEL or "constant": S}, "minus_constant": {MODEL: S}, "SR_minus_CL": S},
  "S5": {"SR_minus_CL_percent": S, "margin": 2.0, "noninferior": bool, "better": bool, "worse": bool, "supported": bool},
  "implied_elasticity": S, "implied_cl_sensitivity": [4],
  "median_seconds": [4], "per_step_seconds": {"CL": [4], "DbS_avg": [4], "SR": [4]}
}
```

Lists run over the set sizes 9, 16, 25 and 36, and lambdas are the held out sensitivities of the original sample. S is {"point", "percentile", "basic", "p", "clear"}.

## Report back

First, when the code is ready and the simulated checks pass (without touching the real data). Later, when told to run, the headline values and decisions of S1 to S5, how long it ran, every ambiguity you resolved and any concern about Amendment 5.

## Clarifications of 14 October, after your preparation report

- S2 is the global minimum over t0 >= 1e-9 seconds and tau >= 0, as you implemented it.
- The time per step divides by the mean effort over the same timed trials as the median, as you implemented it.
- S5 also reports "worse", true when the upper bounds of both the percentile and the basic interval of SR minus CL lie below 0.
- S1 also reports "observed_over_predicted_by_size", the observed mean distinct items over SR's predicted E[looks] - 1, per set size, original sample.
- Your other resolutions (S1 bounds included, S3 counting each trial once and dropping undefined correlations, S4 on the display's top rating with the constant clipped, S5 strict inequalities, times_chance with point and percentile only, the implied sensitivity at lambda >= 0) match the specification.
