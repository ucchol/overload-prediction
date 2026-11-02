# Independent verification brief, H2 the return rule (Amendment 11, confirmatory)

Blind reimplementation in `independent_h2/`, written from `ANALYSIS_PLAN.md` (Amendment 11 and the sections it points to) without opening the package's H2 module, its simulation script, its results or the draft amendment. Run with `sh verification/run_independent_h2.sh [B]` (default B = 50). Output `independent_h2/independent_results.json`.

Where the plan left a detail open, the most literal reading was taken. Every such choice is listed here.

## Data

1. Snack trials come from the four aggregate files. A trial enters when every `item_value_i` of the display is finite and `choice` is an integer in 0 to n - 1 (Amendment 3, item 1). Gaze comes from the per subject `*_desc-gazes.csv` files, matched on (`subject`, `trial`), ordered by `gaze_num`, using `item` and `dur`. Every aggregate trial has gaze rows, and the gaze file's set size and choice agree with the aggregate file on every trial (structure check, no outcome relation computed).
2. Reutskaja trials are (`subject`, `trialid`) groups of `fixations_data.dta`, ordered by `fixOrder`. Set size `numalt`, ratings `rating1` to `rating{numalt}`, choice `chosendestring` (1 based, converted to 0 based), gazed item `Roinum` (1 based), duration `FixDuration`. Rows with a missing `Roinum` or `FixDuration` would be dropped from the gaze sequence; none are missing.
3. Consecutive gazes on one item are merged into one gaze with summed duration. In both datasets no consecutive repeats exist, so merging changes nothing here.
4. Gazes with zero duration (four in the snack gaze files) are kept in the sequence. They count for the return and last cues and add nothing to dwell.
5. A trial whose total gaze time is zero would have dwell zero on every item. A trial without gaze has every cue zero. Neither occurs.

## Cues

6. Return. Walk the merged sequence and mark the first gaze whose item was already gazed at. Only that item gets 1. A trial with no revisit has return zero everywhere.
7. Dwell. Each item's summed duration over the trial's total gaze duration.
8. Last. 1 for the item of the last merged gaze.

## Models and score

9. CL is softmax(lambda v) on the raw ratings (snack -3 to 3, Reutskaja -5 to 5, no shift). CL plus a cue is softmax(lambda v + beta cue), lambda >= 0, beta free.
10. Fits. Weighted maximum likelihood, L-BFGS-B from three (CL) or five (CL plus cue) starts with an analytic gradient, the best start polished by projected Newton steps on the analytic Hessian until the score function is near machine zero. CL alone is fitted the same way (the plan names no separate method for it in Amendment 11).
11. Folds. Leave one set size out, snack 9, 16, 25, 36 and Reutskaja 4, 9, 16. Each set size is predicted by the fit to the other set sizes with the replicate's weights.
12. Score. The weighted mean of log2 q(chosen) over all trials pooled across set sizes (item 3 says "over all trials"), not the mean of per set size means as in Amendment 3. Weights are draw counts.
13. "Bits per choice of CL" is reported twice. `CL_log2_score` is the plan's score, the mean log2 q(chosen), which is negative. `CL_bits_above_uniform` is the mean of log2(n q(chosen)), the Section 5 convention. Gains are differences of the score and are the same under either convention.
14. Gain of a cue is score(CL plus cue) minus score(CL), in bits per choice. `return_minus_dwell` is return gain minus dwell gain, computed replicate by replicate.
15. Cue weight. beta from the fit of CL plus that cue to all set sizes of the dataset with the replicate's weights. The matching lambda (`<cue>_lambda_all`) and CL's lambda on all set sizes (`CL_lambda_all`) are also reported.

## Descriptives

16. By set size, `first_return_chosen_n{n}` is the weighted share, among trials with a return, of trials whose first returned item was chosen. `no_return_n{n}` is the weighted share of all included trials of that set size without a return, trials without gaze included.

## Resampling and decisions

17. Participants are taken in order of first appearance. Snack, the aggregate files read in increasing set size (9, 16, 25, 36). The subject order is the same in every file and ascending, so this matches every other reading. Reutskaja, file order, also ascending.
18. Replicate b uses one generator `numpy.random.default_rng(110000 + b)`, first `integers(0, P_snack, size=P_snack)`, then `integers(0, P_reutskaja, size=P_reutskaja)`. A trial's weight is its participant's draw count. The point estimate uses weight 1 for every trial.
19. Intervals follow Section 4. Percentile interval, `numpy.percentile` (linear interpolation) at 2.5 and 97.5 of the replicates. Basic interval, 2 point minus the 97.5 and 2.5 percentiles. Clear when both intervals exclude zero. Clearly positive when both lower bounds lie above zero. The two sided p value is twice the smaller of the shares of replicates at or below zero and at or above zero, capped at 1.
20. H2 is supported in a dataset when the return gain and the return minus dwell gain are both clearly positive, and overall when it holds in both datasets.
21. The descriptive shares get the same summary (point, percentile, basic, clear). Their clear flags are against zero and carry no hypothesis.
22. Partial results (point estimates and one line per replicate) go to `independent_h2/partial/`, which is not committed. The runner resumes from them after an interruption.

## Note found in the run

23. Reutskaja, last cue. The last fixated item is the chosen item on every included trial, so the likelihood of CL plus last has no finite maximum. The fit stops where the optimizer stops (beta about 86, lambda about 0.24), held out log2 q(chosen) is 0 to rounding, and the last gain equals minus CL's score. The reported `last_beta` and its interval for Reutskaja are optimizer artifacts and should not be compared value by value with another implementation. The last gain is unaffected. The last cue is descriptive only in the plan.
