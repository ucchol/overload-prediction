# Independent verification brief, two look satisficing on the snack data (exploratory)

Same rules as always. Work only inside `verify/`, do not look at any other folder of the scratchpad, write code in `verify/independent_two_look/` and results to `verify/independent_two_look/results.json`. Never write participant counts, trial counts or identifiers. You may reuse your own code in `independent_snack/` and `independent_search/` (data loading, inclusion, the sample until return fit, folds, weights, the time law and the grouped Spearman correlations).

This analysis is exploratory (the snack data suggested the model), so there is nothing to freeze. You may run on the real snack data once your simulated checks pass.

## The model

Two look satisficing (TLS). Ratings v run from -3 to 3, and v_top = 3. Looks fall on item i with probability q_i = softmax(lam v / 2)_i, with replacement. A look at item i registers with probability h_i = exp(lam (v_i - v_top) / 2). The first item with two registered looks is chosen. With rho_i = q_i h_i and P = sum rho,

- choice probabilities are those of sample until return at sensitivity lam, the race on rho / P (which equals softmax(lam v)), so TLS's sensitivity in a fold is exactly your SR sensitivity of that fold;
- E[looks] = E_SR[looks](rho / P) / P, where E_SR[looks](p) = integral of e^(-t) prod_j (1 + p_j t) dt is the sample until return length;
- E[distinct items looked at] = sum_i q_i integral from 0 to infinity of e^(-(q_i + P - rho_i) t) prod over j != i of (1 + rho_j t) dt.

Derive or check these yourself (for example by simulating the process: draw looks from q, register each with probability h, stop at the first item with two registrations, count looks and distinct items). Any exact route is fine.

## Quantities, B = 1000 participant resamples, seeds 40000 + b, exactly as in Amendment 5

Leave one set size out. In fold k, lam_k is the SR (= TLS) sensitivity fitted on the other three set sizes.

- E1, time held out. As S2 of Amendment 5 (log seconds, log(t0 + tau effort), t0 >= 1e-9, tau >= 0, global minimum of the weighted squared error on the timed trials of the three training set sizes, error = weighted mean absolute log difference on the held out set size, model error = mean over folds). Effort is n for CL, SR's E[looks] for SR and TLS's E[looks] for TLS, every trial of every set size computed with lam_k in fold k. Report each model's error (point, percentile, basic, p, clear) with by_size and params, and TLS minus CL and TLS minus SR.
- E2, slow decisions. As S3 of Amendment 5 (participant and set size groups with at least three timed trials, Spearman with average ranks, undefined skipped, averaged over set sizes then participants, participant weights). Forecasts, TLS's E[looks] and SR's E[looks] of the held out set size (fold k), and the display's highest rating. Report each, TLS minus SR, and TLS plus highest rating (the sum of the two correlations).
- E3, search. Weighted mean over the trials of each set size of the observed distinct items looked at (positive cumulative gaze), of TLS's E[distinct] in fold k, and of SR's E[looks] - 1 in fold k. Report the three by set size (original sample), the observed over TLS ratio, percentile intervals by set size for observed and TLS predicted, the two elasticities (least squares slope of log mean on log n) and predicted minus observed elasticity as summaries.
- Sensitivity, original sample only. Attention uniform (q_i = 1 / n) and registration h_i = exp(lam (v_i - v_top)), lam the fold's SR sensitivity. Report its E1 error (unweighted original sample, same law), its E2 mean correlation (unweighted), its predicted distinct items by set size and their elasticity.

## Output

`independent_two_look/results.json`:

```
{
  "lambda": [4],
  "E1": {"error": {"CL": S+, "SR": S+, "TLS": S+}, "TLS_minus": {"CL": S, "SR": S}},
  "E2": {"correlation": {"looks_TLS": S, "looks_SR": S, "top_rating": S}, "TLS_minus_SR": S, "TLS_plus_top_rating": S},
  "E3": {"observed": [4], "predicted": [4], "predicted_SR": [4], "observed_over_predicted": [4],
         "observed_interval": [[4], [4]], "predicted_interval": [[4], [4]],
         "observed_elasticity": S, "predicted_elasticity": S, "predicted_minus_observed_elasticity": S},
  "uniform_attention": {"error": x, "correlation": x, "distinct": [4], "elasticity": x}
}
```

S is {"point", "percentile", "basic", "p", "clear"}, S+ adds "by_size": [4] and "params": [[t0, tau] x 4]. Lists run over the set sizes 9, 16, 25 and 36.

## Checks before the real run

1. Your TLS E[looks] and E[distinct] against a simulation of the process, and against sample until return when every h_i = 1 (E[distinct] = E[looks] - 1).
2. TLS choice probabilities equal SR's at the same lam.
3. On a synthetic set where choices, gaze and times come from TLS, the pipeline recovers lam and gives TLS a lower time error than CL.

## Report back

How long it ran, the headline values of E1 to E3 and the sensitivity, every ambiguity you resolved, and anything that surprised you.
