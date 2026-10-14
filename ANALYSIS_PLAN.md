# Analysis plan

Version 1.0, frozen on 9 October 2026, before any new test listed below was run.

This plan fixes every model, measure, fold, resampling scheme and decision rule in advance. Amendments are appended at the end with a date and a reason, and earlier text is never edited. The SHA-256 digest of this file at each freeze is recorded in `STORY.md`. Analyses run outside this plan are reported as exploratory.

## 1. Question and account

When displays grow in the number of options and in the attributes shown for each option, do people choose by integrating everything shown, or do they narrow their choices to a few options screened on a few cues? The umbrella account is the adaptive decision maker (Payne, Bettman and Johnson, 1993). Under load, people adapt how they choose, screening options on a few salient cues and choosing among what survives.

Each choice model is a formal statement about algorithmic structure. Held out prediction, measured by completeness and overlap, arbitrates between structures. Each hypothesis makes a behavioral prediction and an algorithmic prediction, and it says where a structure should help, not only whether it wins overall.

## 2. Data

### 2.1 Development data

The original study had two tasks, file selection in the role of an intruder and laptop purchase, each with six trials crossing 8, 16 and 32 options with low and high information. The high information laptop trial with an intended 32 options displayed 30, since options EE and FF were never shown.

- Choices. A response is matched to an option when its text, trimmed and lowercased, equals the option name. Unmatched responses are left out of every choice analysis. Every respondent with at least one matched choice in a task enters that task, in the order of the data file.
- Coding. Coded attribute levels follow the stimulus definitions in `src/overload/stimuli` and the coding functions in `src/overload/coding.py`, unchanged from the earlier pipeline.
- Rubric. Each option's rubric score U is the recorded score, with 100 for the best options of a trial. Rubric shares r come from nonnegative least squares of the rubric scores on the coded attributes, both centered within trials, normalized to sum to one. No choices enter the shares.
- Status. This dataset shaped the hypotheses. Results known before this plan are listed under each hypothesis in Section 7. New tests on these data are development evidence.

### 2.2 Confirmatory data

Only datasets that are openly downloadable at no cost enter the work.

- H1. Thomas, Molter and Krajbich (2021), choices of one snack from 9, 16, 25 or 36, with liking ratings, response times and gaze, MIT license.
- H2. An open dataset in which the same options are described by different numbers of attributes. The candidate is Hensher (2006, Journal of Applied Econometrics data archive), subject to confirming open access. The dataset is fixed by amendment before any of its outcomes is inspected.
- Rule. Before a confirmatory dataset's analysis is frozen by amendment, only its structure may be inspected (set sizes, trial counts, variable names and types). Its outcomes are analyzed once.

## 3. Models

Every model has exactly one fitted parameter. Softmax models map option scores v to probabilities through q_i = exp(lambda v_i) / sum_j exp(lambda v_j) with a fitted sensitivity lambda >= 0. The screening model fits a lapse instead.

| Model | Score v_i |
|---|---|
| QR, quantal response | U_i, the rubric score |
| CL, conditional logit | sum_k r_k x_ik |
| DbS_sum, decision by sampling | sum_k r_k sum_{j != i} sign(x_ik - x_jk) |
| DbS_avg | DbS_sum / (n - 1) |
| MLBA_sum, multiattribute linear ballistic accumulator drift | sum_k r_k sum_{j != i} exp(-kappa0 abs(d)) d, with d = x_ik - x_jk and kappa0 = 1 |
| MLBA_avg | MLBA_sum / (n - 1) |

Set size scaling. For CL, DbS_avg and MLBA_avg, the scaled score is v_beta = ((n - 1) / 7)^beta v with beta in {-1, -0.5, 0, 0.5, 1}. Negative beta lowers the effective sensitivity as sets grow, so choices become noisier (the noise account). Positive beta raises it, so choices concentrate (the narrowing account). With averaged comparison scores, beta = 1 reproduces the summed scores exactly, so summing against averaging is the beta = 1 against beta = 0 contrast.

Screening model. Elimination by aspects (Tversky, 1972) with rubric weights, EBA.
- Aspects. For each attribute k with r_k > 0, option i has aspect k when x_ik is at or above the median of attribute k over the displayed set. The aspects are fixed for the trial.
- Elimination. From the remaining set S, an aspect k is drawn with probability r_k divided by the sum of r over the aspects that some but not all options in S have. Options without aspect k are eliminated. The process repeats on the survivors. When no aspect splits S, the choice is uniform over S.
- Lapse. P_i = (1 - epsilon) P_EBA,i + epsilon / n, with epsilon in [0, 1] fitted.
- Robustness variant, secondary. Aspect k present when x_ik equals the best level of attribute k in the displayed set.

Fixed constants. kappa0 = 1, the median aspiration, the beta grid and the rubric shares are fixed in advance and never fitted to choices.

Not in this plan. Mixed logit, since individual heterogeneity is not part of the hypotheses.

## 4. Estimation and validation

- Estimation. Maximum likelihood of the single parameter on the pooled choices. The negative log likelihood is convex in lambda for every softmax model and in epsilon for EBA, so the estimate is the root of the score function, found by bracketed root finding to machine precision. A softmax model whose scores are constant within every fitted trial has lambda = 0.
- Fitted. Each task's model is fitted to all six trials and evaluated on them.
- Held out. Each trial is predicted from the fit to the other five trials of its task.
- Alone. Each trial is fitted to itself, used only for the completeness ladder (H3).
- Resampling. Participant bootstrap with B = 1000 replicates. Replicate b uses a generator seeded with 10000 + b and draws file selection participants before laptop participants, each with replacement and of the original size. Every model is refitted in every fold of every replicate.
- Intervals. 95 percent percentile intervals. A difference is clear when the percentile and the basic interval both exclude zero. The two sided bootstrap p value is twice the smaller of the shares of replicates at or below zero and at or above zero. Holm's correction applies within each family named in Section 7.

## 5. Measures

- Log score of trial c, in bits per choice above a uniform guess. L_c = (1 / N_c) sum_i n_ci log2(n_c q_ci), where n_ci counts the choices of option i, N_c their total and n_c the options.
- Attainable log score. L*_c = log2 n_c - H(p_c) - (m_c - 1) / (2 N_c ln 2), with p_c the observed shares, H in bits and m_c the options chosen at least once (Miller, 1955).
- Completeness over a set of trials S. K_S = 100 sum_{c in S} L_c / sum_{c in S} L*_c. Zero is a uniform guess and about 100 is the true choice probabilities.
- Overlap. O_c = sum_i min(p_ci, q_ci), the share of choices on which prediction and observation agree, averaged over the trials of S.
- Effective number of options. E(p) = exp(H_e(p)), with H_e the entropy in nats. A model's value at the observed sample size, E_samp(q), is the mean of E over 2000 multinomial samples of N_c choices from q, with a fixed seed for each trial and model.
- Regret. 100 - U of the chosen option.

## 6. Reporting rules

- No participant counts in any table, figure or text. Shares are reported as percentages.
- Every result named in Section 7 is reported whatever its direction.
- Development and confirmatory results are labeled as such.

## 7. Hypotheses and decision rules

### R0. Baseline, descriptive

Held out and fitted completeness and held out overlap of QR, CL, DbS_avg and MLBA_avg, with the summed versions for continuity, per task, overall and within each information level, with each model's difference from CL. Known before this plan: completeness of QR, CL, DbS_sum and MLBA_sum (AAMAS version), and of the averaged versions (supplement Section S18).

### H1. Narrowing with set size

Prediction. As options grow, choices narrow rather than become noisier.

- B1, behavior. Concentration gap G_c = E(p_c) / n_c - E_samp(q_c) / n_c, with q_c the held out CL prediction. Predicted, per task, a negative mean of G over the six trials and a negative slope of G_c on log2 n_c across the six trials. Supported in a task when both bootstrap intervals exclude zero in the predicted direction. Known before this plan: participants' choices were more concentrated than every model's at both information levels (S20). Not known: the slope for CL.
- A1a, algorithm, set size scaling. Held out completeness and overlap of CL, DbS_avg and MLBA_avg at every beta. Primary family per task, Holm corrected: CL at beta = 0.5 minus beta = 0 (predicted positive) and CL at beta = -0.5 minus beta = 0 (predicted negative). Supported in a task when both differences are clear in the predicted direction and overlap moves the same way. Known before this plan: beta = 1 against beta = 0 for DbS and MLBA, where beta = 0 predicted better in both tasks. Not known: every other cell.
- A1b, algorithm, screening. Delta_c = L_c(EBA) - L_c(CL) on held out trials. Predicted positive for the mean of the two 32 option trials minus the mean of the two 8 option trials. Supported in a task when that difference is clear. Not known.

### H2. Comparison under sparse information

Prediction. With few attributes shown, people compare options on the salient ones, and the advantage of comparison disappears when many attributes are shown.

- B2, behavior. Correlation between participants' revealed weights and each model's revealed weights, within each information level. Revealed weights come from a descriptive conditional logit with every coded attribute and listing position, fitted to the observed shares and, separately, to each model's held out predicted shares. Predicted, the comparison models' correlation exceeds CL's under low information and not under high. Known before this plan for laptops (S20).
- A2, algorithm. D_mL and D_mH, the held out completeness differences of DbS_avg and MLBA_avg from CL within the low and the high information trials, and Delta_m = D_mH - D_mL. Predicted D_mL > 0 and Delta_m < 0. Known before this plan on the development data (S18). The confirmatory test uses the open H2 dataset, specified by amendment.

### H3. Inputs over rules

Prediction. What people screen on matters more than the rule that combines it, and some cues lie outside the coding.

- A3a, the completeness ladder per task and information level. What the coding allows, alone, fitted to all six trials, and held out, for CL and the comparison models. Known before this plan.
- A3b. Gain in held out completeness from adding one cue that lies outside the coding, taken from an independent source, against the spread of held out completeness across rules. The cue and its source are fixed by amendment before this test runs. The earlier filename score R1 stays exploratory because its lexicon followed inspection of these choices.
- B3. Participants' revealed weights on cues outside the coding (filename meaning, listing position, brand). Known before this plan (S20).

### H4. Cost depends on cue validity

Prediction. Overload hurts only when the cues people screen on are poor guides to value. The measure of cue validity, the regret contrast and the screening model that should reproduce its sign are fixed by amendment before this test runs.

## 8. Software and reproducibility

The package `overload` implements every model and measure, with unit tests for each measure and model, and a reproduction test against the earlier pipeline's per trial values. Results files record the plan digest, the package version and the seeds. Exact library versions used for reported results are listed in `requirements-lock.txt`.

## Amendments

### Amendment 1, 9 October 2026, after the descriptive baseline R0 and before any hypothesis test

1. Completeness in bootstrap replicates. The attainable log score of each trial is held at its full sample value in every replicate, so completeness in a replicate is a fixed rescaling of its log score. Reason. Re-estimating the Miller corrected benchmark inside a replicate inflates it, because the benchmark of the resampled world is the plug-in score of the observed shares. In R0 the mean replicate benchmark was 11.5 percent above the full sample value in both tasks, which pulled replicate completeness down and left the point estimate near the top of its interval (CL file completeness 29.5 with a percentile interval of 19.6 to 33.0, replicate median 26.6). With the benchmark fixed, the replicate median equals the point estimate (29.4 against 29.5) and the intervals equal those of the earlier pipeline. Differences between models share the benchmark, so their tests are rescaled log score tests either way.
2. Overlap. Overlap levels are reported as point estimates without intervals. Differences between models keep intervals and the clear rule of Section 4, which requires the basic interval as well as the percentile interval to exclude zero. Reason. Overlap with resampled shares is biased downward by sampling noise (CL file overlap 58.2 against a replicate median of 55.7), so a percentile interval misplaces the level. In differences the bias mostly cancels, with replicate medians within 0.5 points of the point estimates in R0, and the clear rule guards against the remainder.

### Amendment 2, 9 October 2026, before any H1 test

1. B1 intervals. The effective number of resampled shares is lowered by a second round of sampling, so a replicate's gap is compared with the gap of the bootstrap world rather than with the sample estimate. In the bootstrap world the observed shares p_c play the role of the true probabilities, so its gap is G0_c = E_samp(p_c) / n_c - E_samp(q_c) / n_c, with q_c the held out CL prediction of the full sample. A replicate's error is e_c = G*_c - G0_c, where G*_c is the replicate's own gap. The percentile interval is the sample estimate plus the 2.5 and 97.5 percentiles of the error, and the basic interval is the sample estimate minus them, for the mean gap and for the slope alike (the slope's error is the slope of e_c on log2 n_c). Every E_samp uses 2000 draws with a seed fixed by task, trial and replicate (`overload.engine.concentration_seed`).
2. A1a decision. A primary difference supports H1 when it is clear in the predicted direction and its Holm adjusted p value, within the task's two primary tests, is below 0.05. Overlap moves the same way when the point estimate of the overlap difference has the predicted sign.

### Amendment 3, 9 October 2026, the confirmatory test of H1, before any outcome of its dataset was inspected

Status. Written after the development results of H1 and after inspecting only the structure of the confirmatory data, namely the files, variable names and types, the four set sizes, the number of trials per participant, the rating scale and the fact that no participant saw the same display twice. No choice, response time or gaze outcome of these data has been examined. Known from the original publication (Thomas, Molter and Krajbich, 2021): choices of the highest rated item, choices of the best looked-at item and the number of items looked at were analyzed by set size, and models in which gaze shapes value described choices best. None of the tests below was reported there. Test C3 was formulated after the development results.

1. Data. The aggregate files for set sizes 9, 16, 25 and 36 of the repository athms/many-item-choice at commit f6ee87f, one row per trial. A trial enters when its chosen item index lies within the display and every displayed item has a rating. The value of an item is its liking rating, an integer from -3 to 3. An item counts as looked at in a trial when its cumulative gaze in that trial is positive, with a missing value counted as zero. An item's position is its index in the data, which numbers the grid slots from left to right and top to bottom. Participants saw every display once, so per display shares, the attainable log score, completeness and overlap are not defined for these data.
2. Models. CL with score v_i equal to the rating, and DbS_avg and MLBA_avg (kappa0 = 1) with the rating as the only attribute and share one, each with the set size scaling of Section 3 (beta in {-1, -0.5, 0, 0.5, 1}, n the number of displayed items) and one fitted sensitivity. Every model has two versions, one over all displayed items and one over the looked-at items, in which items never looked at have probability zero. Trials whose chosen item was never looked at are left out of every analysis that uses the looked-at version, and their share is reported for each set size. Elimination by aspects needs more than one attribute and is not used.
3. Estimation and validation. Maximum likelihood on pooled trials, as in Section 4. Held out means leave one set size out, with each set size predicted from the sensitivity fitted to the trials of the other three. The held-out log score of a set size is the mean over its trials of log2(n q), with q the predicted probability of the chosen item, and a model's held-out log score L is the mean over the four set sizes. L is reported as the held-out likelihood of the chosen item relative to a uniform guess, 2^L, and a difference dL as the percent change in that likelihood, 100 (2^dL - 1). Tests use L.
4. Resampling. Participant bootstrap with B = 1000. Replicate b seeds a generator with 20000 + b and draws P participants with replacement, `integers(0, P, size=P)`, from the P participants sorted by identifier. A drawn participant contributes all of their trials, at every set size, once per draw. Every model is refitted in every fold of every replicate. Intervals, p values, the clear rule and Holm's correction follow Section 4.
5. Tests.
   - C1, behavior, narrowing across positions. For set size n, p_n is the share of choices at each position, and q_n is the mean over the trials of that set size of the held-out CL prediction (beta = 0, all items) at each position. The concentration gap is G_n = E(p_n)/n - E_samp(q_n)/n, with E_samp at the number of trials of that set size in the sample at hand and 2000 draws from a generator seeded with (1, 2, k, b + 1), where k runs from 0 for 9 items to 3 for 36 items and b = -1 is the original sample. Predicted, a negative mean of G_n over the four set sizes and a negative least squares slope of G_n on log2 n. Intervals are centered as in Amendment 2, with the bootstrap world's gap E_samp(p_n)/n - E_samp(q_n)/n of the original sample, where E_samp(p_n) uses the seed (2, 2, k, 0). Supported when the mean and the slope are both clear and negative.
   - C2, algorithm, set size scaling. The plan's A1a, over all items and all trials. Primary family: CL at beta = 0.5 minus beta = 0 (predicted positive) and CL at beta = -0.5 minus beta = 0 (predicted negative), Holm within the pair. Supported when both are clear in the predicted direction with Holm p below 0.05. Cells for DbS_avg and MLBA_avg and for the looked-at versions are reported alongside.
   - C3, algorithm, narrowing against noise. The sensitivity elasticity is beta treated as continuous in [-3, 3] and fitted jointly with lambda by maximum likelihood to all included trials of the four set sizes. It measures how CL's sensitivity changes with set size and is not a predictive model. beta_all uses CL over all items and beta_seen CL over the looked-at items, both on the trials whose chosen item was looked at. If choice among the items people consider grows noisier with set size, beta_seen < 0. If people instead narrow, considering a shrinking share of the display while choosing among what they consider as before, sensitivity over all items falls more than among the looked-at items, beta_all - beta_seen < 0. Narrowing is supported when beta_all - beta_seen is clear and negative. Noise is supported when beta_seen is clear and negative. Both can hold, and both are reported with their size.
6. Descriptives reported with the tests. For each set size, the mean share of displayed items looked at (known), the share of trials whose chosen item was looked at, the held-out likelihood relative to chance of every model in both versions, and CL's sensitivity fitted to that set size alone in both versions.
7. Reporting. Results on these data are labeled confirmatory, with C3 marked as formulated after the development results. Any deviation from this amendment found after the run is reported as a deviation.

### Amendment 4, 10 October 2026, correction of the laptop option definitions of trials 5 and 6

Status. A correction of input data. No model, measure, test or decision rule changes. Written after R0, the development tests of H1 and an exploratory laptop check had been run with the earlier definitions, and before any of them was rerun.

1. What changed. The laptop option definitions now follow the study's storyboard (`Choice_Overload_Storyboard_established_method`), which shows every laptop display. Trials 1 to 4 match the earlier definitions exactly. In trials 5 and 6, the two 32 option trials, the earlier definitions, taken from the earlier analysis notebook, differed from the storyboard in 40 and 46 attribute values, spread over 20 and 10 options. Option names, their order and the response columns are unchanged.
2. Evidence. Under the storyboard definitions the designers' rubric scores follow the documented laptop scoring rule (weights 4, 4, 3, 3, 2, 2 and 1 for memory, graphics, storage, processor, battery, brand and price) with correlations from 0.96 to 0.99 in all six trials, against 0.73 and 0.76 in trials 5 and 6 under the earlier definitions. Participants' choices in trials 5 and 6 are also predicted better under the storyboard definitions, by CL with rubric shares estimated from trials 1 to 4 (1.13 against 1.04 bits per choice in trial 5 and 0.99 against 0.91 in trial 6). The laptop screenshots of trials 1 and 2 in an earlier draft of the paper show a different version of those displays, consistent with neither the storyboard nor the designers' scores, and are not used.
3. Consequence. Every laptop result is recomputed with the same pre-specified analyses and seeds (R0, and B1, A1a and A1b of H1). Results obtained with the earlier definitions are superseded and stay on record in `STORY.md` and the repository history. File results are unaffected. The earlier definitions remain in `src/overload/stimuli/laptop_trials_aamas.json` only for the test that the rebuilt pipeline reproduces the earlier one. The exploratory laptop check of 9 October (rubric shares from trials 1 to 4) addressed a symptom of this error and is withdrawn.

Correction to item 2, 14 October 2026. The bits quoted there are in sample fits, with CL's sensitivity fitted to each trial alone. Held out, with the sensitivity fitted to the other five laptop trials, the storyboard definitions still predict better, 1.02 against 0.94 bits per choice in trial 5 and 0.96 against 0.89 in trial 6. The independent verifier found the wording, and both sets of figures were recomputed by the package and by the verifier. Nothing else in the amendment changes.
