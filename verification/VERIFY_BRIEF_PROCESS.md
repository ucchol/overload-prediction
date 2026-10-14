# Verification brief, overload, decision time and search models (development data)

This brief is self-contained apart from what you already implemented for R0 and H1 (`VERIFY_BRIEF.md`, the
stimuli files, `coding_definition.py`, the rubric shares and the CL, DbS_avg, MLBA_avg and EBA models,
the leave-one-trial-out folds, completeness with the benchmark held at its full sample value, and the
participant bootstrap of R0). Use the storyboard laptop definitions (`stimuli/laptop_trials.json`).

Rules as before. Work only in the verify folder, write code under `verify/independent_process/` and
results to `verify/independent_process/results.json`. Never print or write identifiers, participant counts or
choice counts. Do not look at any other results.

## A. Per trial answers

For task prefix P (`ci_co_fl_` for files, `em_co_lp_` for laptops) and trial k = 1..6, the CSV has
`P{k}` (the choice), `P{k}_regret`, `P{k}_timer` (seconds on the choice page) and `P{k}_survey_1` ..
`P{k}_survey_6` (answers 1 to 5). Every CSV row is one participant and every row has all twelve trials
answered. Participants are the CSV rows in file order.

The six questions are satisfaction, certainty of having chosen best, confusion, frustration, complexity and
uncertainty about choosing best. Aligned answers: a1 = 6 - s1, a2 = 6 - s2, a3 = s3, a4 = s4, a5 = s5,
a6 = s6. The overload index of a participant on a trial is the mean of a1..a6.

## B. Overload index analysis (results key "overload")

Arrays are participants by twelve trials, file trials 1..6 then laptop trials 1..6.

1. Measurement, point values only.
   - alpha_within_person. Center each aligned item within participant (subtract the participant's mean of
     that item over all twelve trials), stack all participant-trials as rows, Cronbach's alpha with ddof 1
     variances: k/(k-1) * (1 - sum of item variances / variance of the row sum).
   - alpha_raw. The same on the uncentered aligned items.
   - first_component_share. Largest eigenvalue of the correlation matrix of the centered aligned items over
     the sum of eigenvalues.
   - question6_raw_correlations. Center the raw answers s1..s6 within participant as above and give the
     Pearson correlation of s6 with each of s1..s5.
2. Per trial (twelve trials): overload (mean index), overload_vs_average (mean index of the trial minus the
   mean index over all participants and all twelve trials), regret (mean), best_chosen (share with regret 0),
   geo_seconds (exp of the mean of log seconds).
3. Contrasts within each task, for the index, regret and log seconds, using the six trial means M1..M6 in
   trial order (8 brief, 8 full, 16 brief, 16 full, 32 brief, 32 full): 32_minus_8 = ((M5 + M6) - (M1 + M2)) / 2
   and full_minus_brief = ((M2 + M4 + M6) - (M1 + M3 + M5)) / 3. For log seconds report 100 (exp(d) - 1).
4. Associations. For a set of trials (files, laptops, or all twelve), remove participant and trial means,
   x - row mean - column mean + grand mean, on the participants by trials array of the index, of regret and
   of log seconds, then give the Pearson correlations of the flattened residuals: overload~regret,
   overload~logtime, regret~logtime, for "file", "laptop" and "both".
5. Bootstrap. 1000 replicates. Replicate b draws participant rows with
   `numpy.random.default_rng(30000 + b).integers(0, N, size=N)` where N is the number of rows, keeping all
   twelve trials of each drawn participant (a drawn participant counts as a new participant each time it is
   drawn, for the centering). Recompute everything. Report percentile 2.5 and 97.5 for every quantity of 1 to
   4. For overload_vs_average, contrasts, associations and the three measurement values also report the
   basic interval (2 point - upper, 2 point - lower) and the two sided p value
   min(1, 2 min(share <= 0, share >= 0)).

## C. Search models

Values v of an option are the same as for R0. CL values X r, DbS_avg scores and MLBA_avg scores.

1. Sample until return (SR). Let q = softmax(lambda v). Looks fall on options independently with probabilities
   q, with replacement. The first option to receive its second look is chosen.
   P(i) = q_i^2 * sum_{k=0}^{n-1} (k+1)! e_k(q without i), where e_k is the k-th elementary symmetric
   polynomial. The expected number of looks is E[looks] = sum_{k=0}^{n} k! e_k(q). You may compute these any
   way you like (for example by the integrals P(i) = q_i^2 int_0^inf t e^{-t} prod_{j != i}(1 + q_j t) dt and
   E[looks] = int_0^inf e^{-t} prod_j (1 + q_j t) dt, or by simulation to check). Models SR (CL values),
   SR_DbS (DbS_avg values) and SR_MLBA (MLBA_avg values).
2. Accept in reading order (Accept_read, CL values). Options are inspected in listed order (the order of the
   stimuli file), option i accepted with probability a_i = 1 / (1 + exp(-lambda (v_i - max v))), passes repeat
   until acceptance. P(i) = a_i prod_{j<i} (1 - a_j) normalized to sum one. Expected inspections
   sum_{k=1}^{n} prod_{j<k} (1 - a_j) divided by (1 - prod_j (1 - a_j)).
3. Fitting. lambda >= 0 maximizes the summed multinomial log likelihood of the fitted trials. The likelihood
   is not guaranteed to be concave, so search globally (for example a grid over lambda then a local refine).
   Held out prediction of a trial uses lambda fitted to the other five trials of its task, as in R0.
4. Report per task (results key "choices"): held out completeness over the six trials for QR, CL, DbS_avg,
   MLBA_avg, DbS_sum, MLBA_sum, EBA, SR, SR_DbS, SR_MLBA, Accept_read, with the R0 bootstrap (seed 10000 + b,
   file participants drawn before laptop participants, benchmark at its full sample value), percentile and
   basic intervals, p and the clear flag (both intervals exclude zero). Differences SR-CL, SR_DbS-DbS_avg,
   SR_MLBA-MLBA_avg, Accept_read-CL, and SR and SR_DbS minus each of CL, DbS_avg and MLBA_avg, with the same
   summaries. Also the held out lambdas of the original sample per model and trial.

## D. Effort and decision time (results key "time")

m is the number of attributes that differ between the trial's options and have a positive rubric share.
Effort in elementary steps per trial:
- QR and CL: n m.
- DbS_avg and MLBA_avg: n (n - 1) m.
- EBA (median aspects, the R0 screening model): aspects are the attributes that differ between the options and
  have a positive share, with weights equal to their shares. checks(S) = 0 if |S| = 1. If no aspect splits S
  (some but not all options of S have it), checks(S) = |S| A where A is the number of aspects. Otherwise
  checks(S) = |S| W / W_split + sum over splitting aspects a of (w_a / W_split) checks(S with aspect a), where
  W is the sum of all aspect weights and W_split the sum over splitting aspects. Effort = checks(all options).
- SR, SR_DbS: E[looks] m, with the held out lambda of the trial and the model's own values.
- Accept_read: expected inspections times m, with its held out lambda.

People's typical time on a trial is exp(mean log seconds) over the (resampled) participants of that task.
Per model and trial: seconds_per_step = typical time / effort.

Held out time law. For a model and task, for each trial j, fit t0 >= 0 and tau >= 0 by nonnegative least
squares of typical seconds on [1, effort] over the other five trials, predict t0 + tau effort_j, and record
|log(predicted / observed)|. A model's typical error is 100 (exp(mean of the six log errors) - 1). The
constant model predicts the mean of the other five trials' typical seconds. Report per task and model
(QR, CL, DbS_avg, MLBA_avg, EBA, SR, SR_DbS, Accept_read, constant): effort per trial, seconds_per_step per
trial with bootstrap percentile intervals, held out predicted seconds per trial, typical error with the
bootstrap summaries, and the difference of mean log errors (model minus constant) with summaries.

The bootstrap is the R0 bootstrap of C.4 (the same replicate draws). In every replicate refit every model,
recompute the effort of the search models with their refitted held out lambdas and recompute typical times
from the drawn participants of the task.

## E. Premonition (results key "premonition")

Per task, the target is the trial's mean overload index over the drawn participants. Forecasts per trial,
all computed without the trial's own choices or answers:
- options n, attributes m, cells n m,
- looks_SR, E[looks] of SR with its held out lambda,
- entropy_X for X in QR, CL, DbS_avg, MLBA_avg, SR, SR_DbS, the entropy in bits of the model's held out
  predicted choice distribution,
- effort_X for X in CL, DbS_avg, EBA, SR.

For each forecast: Spearman correlation with the target over the six trials (0 if the forecast is constant);
held out calibration error, for each trial fit target = a + b forecast by least squares on the other five
trials and record |prediction - target|, then average over the six; the constant error, the mean over trials
of |mean of the other five targets - target|; and error minus constant. Bootstrap as in D, with summaries.

## F. Additions to E (14 October)

1. Scopes. Run E for three scopes, "file" (six trials), "laptop" (six trials) and "both" (the twelve trials
   together, file trials first, one Spearman correlation and one held out calibration line over the twelve, each
   trial predicted from a line fitted on the other eleven). Store E under results["premonition"][scope].
2. Regret forecasts. For X in QR, CL, DbS_avg, MLBA_avg, EBA, SR, SR_DbS, the expected regret of X on a trial is
   the sum over options of X's held out choice probability times (100 minus the option's rubric score). The
   target is the trial's mean regret over the drawn participants. For each X and scope report the forecasts per
   trial, the error (mean over the scope's trials of |forecast - target|, no calibration), the constant error on
   regret (mean over trials of |mean of the other trials' targets - target|), the error minus the constant error,
   and the Spearman correlation of forecast and target, with the bootstrap summaries. Store them under
   "regret_forecasts", "regret_error", "regret_constant_error", "regret_error_minus_constant" and
   "regret_spearman" in the scope's block, and the targets under "regret" (and the overload targets under
   "overload").

## Report

Reply with a short report, every ambiguity you resolved and how, anything that looks wrong, and the headline
numbers: alpha within participants and the first component share; overload_vs_average for the twelve trials;
the 32 minus 8 contrasts; held out completeness of CL, DbS_avg, MLBA_avg, SR, SR_DbS and Accept_read per task
with SR-CL and SR_DbS-DbS_avg; seconds_per_step of CL, DbS_avg and SR in the 32 option trials; typical time
errors per model and task; and the Spearman correlations of options, looks_SR and entropy_DbS_avg per task.
