# Independent verification brief, confirmatory H1 on the snack data

Same rules as before. Work only inside `verify/`, do not look at any other folder of the scratchpad, write code in `verify/independent_snack/` and results to `verify/independent_snack/results.json`. Never write participant counts or trial counts anywhere. Report shares, log scores and estimates only.

## Inputs

- `snack/setsize-{9,16,25,36}_desc-data.csv`, the open data of Thomas, Molter and Krajbich (2021), one row per trial. Columns used: `subject`, `choice` (chosen item index), `item_value_{i}` (liking rating of item i, integers -3 to 3), `cumulative_gaze_{i}` (share of trial time spent looking at item i). Item index i is also the item's position on the screen.
- `ANALYSIS_PLAN.md`, Amendment 3 at the end is the specification. Sections 3 to 5 and Amendment 2 give the shared definitions.

## Details Amendment 3 leaves implicit

1. Trials. Keep a trial when `choice` is an integer from 0 to n - 1 and every `item_value_i` of the display is present. Looked-at items: `cumulative_gaze_i > 0`, a missing value counting as zero.
2. Participants. The sorted list of distinct `subject` values over the four files. Replicate b uses `numpy.random.default_rng(20000 + b).integers(0, P, size=P)`. A trial's weight is the number of times its participant was drawn (1 for every trial in the original sample).
3. Fitting. Weighted maximum likelihood of one sensitivity lambda >= 0 for the pooled included trials, lambda = 0 when the derivative of the log likelihood at 0 is not positive, precision 1e-10 relative or better. In the looked-at version the softmax runs over the looked-at items of each trial only.
4. Scores. CL is the rating. DbS_avg of item i is the sum over the other items j of sign(v_i - v_j), divided by n - 1. MLBA_avg is the sum over j of exp(-|d|) d with d = v_i - v_j, divided by n - 1. These use every displayed item, looked at or not. Scaled scores multiply by ((n - 1) / 7)^beta.
5. Held out. For each set size, fit lambda to the included trials of the other three set sizes, then take the weighted mean of log2(n q_chosen) over the included trials of the held-out set size, n being the number of displayed items in both versions. L is the plain mean of the four set size values.
6. Included trials. Version "all": every trial. Version "seen": the trials whose chosen item was looked at. C3 uses the "seen" trials for both elasticities.
7. C1. For set size k (0 for 9 items up to 3 for 36), p is the weighted share of choices at each item index, and q the weighted mean over all trials of the held-out CL (beta = 0, version "all") probability vector. E(p) = exp(-sum p ln p). E_samp(q) at N, with N the weighted number of trials of that set size: `s = numpy.random.default_rng(seed).multinomial(N, q / q.sum(), size=2000) / N`, the effective number exp(-sum s ln s over s > 0) of every draw, then their mean. Seeds are tuples, (1, 2, k, b + 1) for q (b = -1 for the original sample) and (2, 2, k, 0) for p of the original sample. G_k = E(p)/n - E_samp(q)/n. The slope is the least squares slope of G on log2 n. Centered intervals as in Amendment 2: the error of a replicate is its gap minus the original sample's world gap E_samp(p)/n - E_samp(q)/n, the percentile interval is the point estimate plus the 2.5 and 97.5 percentiles of the errors, the basic interval the point estimate minus them, and p uses point estimate plus error.
8. C3. Joint maximum likelihood of lambda and a continuous beta in [-3, 3] for CL, version "all" and version "seen", on the "seen" trials. Report beta to 1e-7 or better.
9. Summaries as before: percentile interval `numpy.percentile(v, [2.5, 97.5])`, basic interval [2t - hi, 2t - lo], clear when both exclude zero on the same side, p = min(1, 2 min(mean(v <= 0), mean(v >= 0))), Holm within the two primary tests of C2.

## What to compute

- C2. For CL, DbS_avg and MLBA_avg, every beta in {-1, -0.5, 0, 0.5, 1}, both versions: held-out L (point, percentile, basic), the four set size values, the four held-out lambdas, and for beta != 0 the difference from beta = 0 of the same family and version (point, percentile, basic, p, clear). The two primary decisions with Holm p, and whether C2 supports H1.
- C1. Per set size E(p)/n, E_samp(q)/n and the gap, then the mean gap and slope with centered intervals, p and clear, and the C1 decision.
- C3. beta_all, beta_seen and beta_all - beta_seen (point, percentile, basic, p, clear), the fitted lambdas, and the two decisions.
- Descriptives per set size: mean share of displayed items looked at, share of trials whose chosen item was looked at, and CL's lambda fitted to that set size alone for version "all" on all trials, version "all" on the "seen" trials and version "seen" on the "seen" trials, each with point and percentile interval.

## Output

`independent_snack/results.json`:

```
{
  "C2": {"cells": {"CL|beta=+0.5|all": {"L": {...}, "by_size": [..4..], "lambda_held_out": [..4..],
                                         "minus_beta0": {...}}, ...},
         "primary": {"CL|beta=+0.5|all": {"p": ..., "p_holm": ..., "supports_H1": ...}, "CL|beta=-0.5|all": {...}},
         "supported": ...},
  "C1": {"per_size": [{"set_size": 9, "observed_share": ..., "model_share": ..., "gap": ...}, ...],
         "mean_gap": {...}, "slope": {...}, "supported": ...},
  "C3": {"beta_all": {...}, "beta_seen": {...}, "beta_all_minus_seen": {...}, "lambda_all": ..., "lambda_seen": ...,
         "narrowing_supported": ..., "noise_supported": ...},
  "descriptives": {"share_looked_at": [{...} x4], "share_chosen_looked_at": [{...} x4],
                   "lambda_alone": {"all_items_all_trials": [{...} x4], "all_items_C3_trials": [...],
                                    "seen_items_C3_trials": [...]}}
}
```

Cell names are `{family}|beta={beta:+.1f}|{version}`, for example `MLBA_avg|beta=-1.0|seen`. Each `{...}` holds point, percentile, basic, p and clear where the brief asks for a test, otherwise point and percentile.

## Report back

Whether everything ran and how long it took, every ambiguity you resolved and how, any concern about Amendment 3 itself, and the headline values: the two C2 primary differences with intervals and decisions, the C1 mean gap and slope with intervals and decision, and beta_all, beta_seen and their difference with intervals and decisions.
