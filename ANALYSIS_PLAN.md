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

None yet.
