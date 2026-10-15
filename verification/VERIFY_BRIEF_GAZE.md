# Independent verification brief, two look satisficing on three gaze datasets (Amendment 6, confirmatory)

Same rules as always. Work only inside `verify/`, do not look at any other folder of the scratchpad, write code in `verify/independent_gaze/` and results to `verify/independent_gaze/results.json`. Never write participant counts, trial counts or identifiers. You may reuse your own earlier code (`independent_search/` for the time law, the Spearman conventions and the summaries, `independent_two_look/` for two look satisficing).

**Do not run anything on the real gaze data until the coordinator says so.** Write the code, test it on simulated data (below), and report that it is ready. The amendment is frozen only after the code on both sides is ready, and the real data are then analyzed once. Until then, do not open the gaze files beyond what loading code needs (column names), and do not compute anything on them.

## Inputs

In `verify/gaze/`:

- `krajbich2010.csv`, `krajbich2011.csv`, `folke2016.csv`. Columns `subject`, `trial`, `rt` (milliseconds), `choice` (the index of the chosen item, 0 based), `item_value_0` to `item_value_{n-1}`, and gaze shares (`gaze_0`, ...), which are not used.
- `folke2016_original.csv`. Columns `Conf` (confidence after the choice) and `Choice_RT`, in the same row order as `folke2016.csv`. Attach confidence by row only after checking that the two files have the same number of rows and equal response times on every row, and stop with an error otherwise.
- `LICENSE` and `COMMIT` (provenance).

Section 4 of `ANALYSIS_PLAN.md` (intervals, p values, the clear rule) and Amendment 5 (time law, Spearman conventions) apply where the amendment points to them.

## The specification, Amendment 6 as it will be frozen

### Amendment 6, 14 October 2026, confirmatory test of two look satisficing on three open gaze datasets

1. Data. The prepared files of Thomas, Molter, Krajbich, Heekeren and Mohr (2019) at commit 4a4f230fb5aa2500d81bfc16b44da3478519ca4f of athms/gaze-bias-differences (MIT license), fetched by `scripts/fetch_gaze_data.sh`. Krajbich, Armel and Rangel (2010), two foods. Krajbich and Rangel (2011), three foods. Folke, Jacobsen, Fleming and De Martino (2016), three snacks. The perceptual dataset of the same collection is left out, since its values are not preferences. A trial is included when every item value is finite, the choice is one of the items and the response time is positive and finite. Response time in seconds is rt / 1000. Folke's confidence ratings come from the original file, attached by row after checking that the two files have the same rows and equal response times on every row. The analysis stops if an included value lies above 10.
2. Values. The item values as given, and v_top = 10 for every dataset, the top of each value scale.
3. Models. CL chooses with softmax(lambda v). DbS and MLBA choose with softmax(lambda s), with the single attribute scores of Amendment 3 (DbS_avg and MLBA_avg, kappa = 1). Since kappa = 1 was set on snack ratings that span six units, MLBA scores the values multiplied by 0.6, which puts the ten unit scale on that span. CL and DbS do not depend on the unit. Sample until return (SR) chooses with race_probabilities(softmax(lambda v)) and takes E_SR(softmax(lambda v)) looks, E_SR(p) = sum_k k! e_k(p). Two look satisficing (TLS) chooses as SR at the same lambda, so one fit serves both, and takes E_SR(rho / P) / P looks, with rho = softmax(lambda v / 2) exp(lambda (v - 10) / 2) and P = sum rho. Sensitivities are fitted by weighted maximum likelihood on the training participants, CL, DbS and MLBA by root finding on their concave likelihoods, and SR on the grid of Amendment 5 (zero and 21 values spaced evenly in log between 0.01 and 100) refined by bounded Brent search (absolute tolerance 1e-10) between the neighbors of the best grid value. Probabilities of a chosen item are floored at 1e-300 inside the log.
4. Folds. Ten folds of participants within each dataset. The sorted participants are permuted by numpy.random.default_rng(20261014).permutation, a seed apart from the resampling seeds, and a participant's fold is its position in that permutation modulo 10. Each fold is predicted from fits to the other nine.
5. Resampling. B = 1000. Replicate b draws participants with replacement, as many as the dataset has, from numpy.random.default_rng(60000 + b), Krajbich 2010 first, then Krajbich 2011, then Folke, each with integers(0, P, size=P) over its sorted participants. Every trial is weighted by its participant's draw count, a participant keeps its fold, and every model is refitted in every fold of every replicate. Intervals, p values and the clear rule follow Section 4.
6. Measures, for each dataset and pooled as the mean over the three datasets.
   - Time. log seconds is predicted as log(t0 + tau effort), t0 >= 1e-9 seconds and tau >= 0 at the global minimum of the weighted squared error over the training participants' trials, and applied to the held out fold. Effort is the number of items for the static models (CL, DbS and MLBA), so their law is a constant, and the expected number of looks with the fold's sensitivity for SR and TLS. The error is the weighted mean absolute log difference over every trial, each predicted by the fold that holds it out.
   - Slow decisions. Within each participant with at least three trials, the Spearman correlation (average ranks for ties, undefined and skipped when either variable is constant) between a forecast and the response time, averaged over participants with their draw counts as weights. Forecasts are TLS's and SR's expected looks from the fold that holds the participant out, and three display features, the highest value, the mean value and the gap between the two highest values. Forecasts are rounded to ten significant digits (Python's format .10g) before they are ranked, so that forecasts equal in exact arithmetic tie, such as SR's looks on two displays whose values differ by a constant.
   - Choices. L is the weighted mean of log2(n q_chosen) over every trial, with q from the fold that holds it out, for CL, DbS, MLBA and SR (which is also TLS's).
7. Primary tests, on the pooled values.
   - G1, time. Supported when TLS's error minus SR's and TLS's error minus the static law's are both clear and negative.
   - G2, slow decisions. Supported when TLS's correlation is clear and positive and TLS's minus SR's is clear and positive.
   - G3, choices. TLS minus CL as 100 (2^dL - 1) percent, dL the difference in L. Supported (non inferior) when the lower bounds of both the percentile and the basic interval lie above -2 percent. Better and worse are reported as in S5 of Amendment 5.
8. Secondary and descriptive, with no direction tested.
   - Every measure for each dataset alone, and the fitted sensitivities.
   - Choices, TLS minus DbS and TLS minus MLBA, in percent as in G3.
   - Slow decisions, TLS's correlation plus each display feature's correlation. Each feature is expected to go with faster decisions, so a positive sum means TLS's looks foresaw slow decisions better than the feature did.
   - Time per look. For SR and TLS in each dataset, the law fitted to every trial with the sensitivity fitted to every trial, t0 as start up seconds and tau as seconds per look, the held out expected looks averaged over trials, and the geometric mean response time. The law describes how long a look would take if the model were right, and is read that way only when G1 is supported. Since it is fitted to expected rather than actual looks, tau understates the time per look somewhat even when the model is right (by about a seventh in simulations of the verification).
   - Confidence. In Folke's data, the within participant correlation of confidence with TLS's and SR's expected looks and with the entropy in bits of CL's held out choice probabilities, and TLS's minus SR's, computed as in item 6 on the trials that carry a confidence rating. More looks or more spread are expected to go with lower confidence.
9. Reporting. G1 to G3 are confirmatory and item 8 descriptive. The three tests answer different questions and are reported without a correction for multiplicity. Any deviation from this amendment found after the run is reported as a deviation.

## Conventions to keep the two implementations comparable

- Pooled values are the means over the three datasets of the per dataset values, replicate by replicate (time errors, correlations and L alike). The pooled dL is therefore the mean of the three per dataset dL, and the pooled times chance is 2 to the pooled L.
- The original sample is the point estimate. Its lambdas, by fold 0 to 9, are reported per dataset.
- TLS_minus_CL_percent is the summary of dL (point, percentile, basic, p, clear) with point and both intervals transformed by 100 (2^x - 1), p and clear kept.
- For G3, non inferior when min(percentile lower, basic lower) > -2, better when both lower bounds are above zero, worse when both upper bounds are below zero.
- TLS_minus_DbS_percent and TLS_minus_MLBA_percent are built like TLS_minus_CL_percent. TLS_plus_<feature> is the summary of TLS's correlation plus the feature's, replicate by replicate.
- The folds use a fresh numpy.random.default_rng(20261014) for every dataset.

## Output

`independent_gaze/results.json`:

```
{
  "pooled": BLOCK + {"G1": {"supported": bool}, "G2": {"supported": bool},
                     "G3": {"margin": 2.0, "noninferior": bool, "better": bool, "worse": bool, "supported": bool}},
  "krajbich2010": BLOCK + DATASET,
  "krajbich2011": BLOCK + DATASET,
  "folke2016": BLOCK + DATASET + {"confidence": {"looks_TLS": S, "looks_SR": S, "entropy_CL": S, "TLS_minus_SR": S}}
}
BLOCK = {
  "time": {"static": S, "SR": S, "TLS": S, "TLS_minus_SR": S, "TLS_minus_static": S},
  "slow": {"looks_TLS": S, "looks_SR": S, "top_value": S, "mean_value": S, "top_gap": S, "TLS_minus_SR": S,
           "TLS_plus_top_value": S, "TLS_plus_mean_value": S, "TLS_plus_top_gap": S},
  "choices": {"times_chance": {"CL": x, "DbS": x, "MLBA": x, "SR": x},
              "TLS_minus_CL_percent": S, "TLS_minus_DbS_percent": S, "TLS_minus_MLBA_percent": S}
}
DATASET = {
  "lambda": {"CL": [10], "DbS": [10], "MLBA": [10], "SR": [10]},
  "law": {"SR": {"t0": S, "tau": S}, "TLS": {"t0": S, "tau": S}},
  "looks": {"SR": S, "TLS": S},
  "typical_seconds": S
}
```

S is {"point", "percentile", "basic", "p", "clear"}. "SR" under times_chance is also TLS's, since the two choose alike. law holds the law fitted to every trial of the sample with the SR sensitivity fitted to every trial (no folds), refitted in every replicate. looks is the weighted mean over trials of the held out expected looks, and typical_seconds is exp of the weighted mean log seconds.

## Tests on simulated data before anything else

1. Your small set race formulas (choice probabilities and expected looks of sample until return, expected looks of two look satisficing) against a simulation of the processes and against your earlier general code.
2. Recovery of lambda for CL and SR from choices simulated at a known lambda, two and three items, integer values 0 to 10.
3. A synthetic set in the input format (all four files), with choices from two look satisficing and response times growing with its looks and a participant speed factor. Your pipeline should give TLS a lower time error than SR and than the static law, a positive TLS correlation with time above SR's, and, with a synthetic confidence that falls with looks, a negative correlation of TLS looks with confidence.
4. Every fitted lambda is a maximum of its likelihood, and every time law a global minimum (check against a brute force grid on a few folds).

## Report back

First, when the code is ready and the simulated checks pass, every ambiguity you resolved and any concern about the amendment. Later, when told to run, how long it ran, the headline values and decisions of G1 to G3, the confidence results and anything that surprised you.
