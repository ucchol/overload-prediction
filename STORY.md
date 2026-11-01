# Story

Living record of the paper's story, the status of every hypothesis and every result. Updated after each step.

## The story in one paragraph

When displays grow in the number of options and in the attributes shown for each option, people do not integrate everything shown. They adapt (Payne, Bettman and Johnson, 1993), screening options on a few salient cues and choosing among what survives, so their choices narrow rather than become noisier. Choice models are formal statements of how choices are computed, and held out prediction decides between them. Each hypothesis predicts a behavioral signature and the model structure that should predict better where that signature appears. The intruder's file choice and the buyer's laptop choice are the development data, and open datasets give the confirmatory tests.

## Hypotheses and status

| | Behavior | Algorithm |
|---|---|---|
| H1 Narrowing with set size | B1 done. Large gap in both tasks, growing with set size for laptops only. Confirmatory C1 done, no concentration on screen positions. Attention narrowed from 82 to 41 percent of the display | A1a done. Laptops support H1 after Amendment 4 (noise rejected, sharpening at beta +0.5 clear), files do not (sharpening hurts). A1b done, rubric screening predicts poorly. Confirmatory C2 done, noise rejected (-10.3%), beta +0.5 not clear, so not supported by its rule. C3 done, sensitivity rises with set size (elasticity +0.18), also among looked-at items |
| H2 Comparison under sparse information | B2 revealed weights, known for laptops | A2 known on development data, confirmatory dataset to fix |
| H3 Inputs over rules | B3 known | A3a ladder known. A3b (Amendment 9) not supported, the cue lowered held out completeness (-4.7 points against the spread across rules) |
| H4 Cost depends on cue validity | set aside for a separate paper (Amendment 9) | set aside |
| O Where overload occurs (research program) | Development done. One factor (alpha 0.87). File displays overloaded people more as they grew, laptop displays never did. Overload went with longer decisions | Premonition within files works for every forecast, set size included. No forecast foresees the gap between tasks or display level regret. Snack data (Amendment 5). The model's expected looks foresee slow decisions (+0.08, supported), but the rating of the best item foresees them far better (-0.32) |
| T Does each model's work explain decision time (research program) | People took 29 to 40 s on laptops and 42 to 71 s on files | Not fitted. Every fitted time law puts most of the time into its start up constant, and no held out law beats a constant on the development data. On the snack data no model times single decisions better than counting options (S2 not supported). The per step argument was withdrawn on 14 Oct. Gaze data (Amendment 6). Two look satisficing timed decisions no better than one constant time (G1 not supported), and sample until return beat one constant time by about 1 percent of the error (exploratory) |
| S The search model, sample until return (research program) | | Matches CL on CL values and DbS on DbS values in held out choices. Snack data (Amendment 5). Search grows like the square root of the set size as predicted (S1 supported), but people look at 3.6 times as many items. Choices as well as CL (S5 supported), clearly better than DbS and MLBA. Gaze data (Amendment 6). Choices as well as CL (G3 supported), better than DbS and MLBA. Its looks foresaw slow decisions in all three datasets (+0.21). Two look satisficing, the absolute standard built on the snack data, did no better (G1 and G2 not supported) |

## Plan versions

| Version | Date | SHA-256 of ANALYSIS_PLAN.md |
|---|---|---|
| 1.0 | 9 Oct 2026 | 6bc0e96185501c9ae5bf2fcd14a6a2fb617466ee3c740d505e4eb773e5c75cd8 |
| 1.0 with Amendment 1 | 9 Oct 2026 | 2953b6ecaf8420cacc3e7112d8effd153e986e4cae1c34887a2d1130729e6ed0 |
| 1.0 with Amendments 1 and 2 | 9 Oct 2026 | f3b624d8c2500f4e711520e60939cf18d83826f49654df232152d3da9bcbaad2 |
| 1.0 with Amendments 1 to 3 | 9 Oct 2026 | bd138487b4209818aacb5e68ffbd5cd8693e4e44461cfc819ed60e9c3ab8d519 |
| 1.0 with Amendments 1 to 4 | 10 Oct 2026 | d93b162c719978c91da8405c81abc94dbb2818a2a4dcefafaef13053a46c41e4 |
| 1.0 with Amendments 1 to 4, wording correction to Amendment 4 item 2 | 14 Oct 2026 | d3bfe629fe81207458fb58305f8f0529dfc883233e324008747e155de9b71c96 |
| 1.0 with Amendments 1 to 5 | 14 Oct 2026 | 5a073cf3759d2bdf03bd4363e304cf293e1cb426bd308d603cf9c5080aba10fa |
| 1.0 with Amendments 1 to 6 | 14 Oct 2026 | 634cdbb8aaf68267d32847fcf129cf9885e286fa2b167136725daa0927581985 |
| 1.0 with Amendments 1 to 8 | 31 Oct 2026 | 876dbadf8f6d0f54b7a5ce0b9c3b47bcdbbc2e38c03dd62f97683d60e079936a |
| 1.0 with Amendments 1 to 9 | 1 Nov 2026 | 7ebeb737a22d0a463189dcc024354a922d57793daff8769ec39c8434ae72e696 |

## Results log

### 9 Oct 2026. Pipeline rebuilt and verified

The package reproduces the earlier pipeline from the raw data. Rubric shares match exactly, and per trial held out and fitted log scores and overlaps of QR, CL, DbS and MLBA match within 1e-6 bits (largest difference 2.5e-8, from the earlier optimizer's tolerance), for the original sample and for bootstrap replicates 0, 1, 499 and 999. Published completeness values round to the same first decimal.

### 9 Oct 2026. R0, the baseline (descriptive, development data). Laptop values superseded by Amendment 4, see 14 Oct

Held out completeness, 95 percent intervals, 1000 participant resamples (`results/baseline`).

| | QR | CL | DbS_avg | MLBA_avg | DbS_sum | MLBA_sum |
|---|---|---|---|---|---|---|
| Files | 21.9 | 29.5 [20.2, 39.2] | 32.1 | 31.4 | 23.9 | 22.6 |
| Laptops | 47.2 | 50.3 [38.9, 62.0] | 55.3 | 57.2 | 49.6 | 50.0 |

- Averaged comparison scores predicted unseen trials at least as well as CL in both tasks. DbS_avg exceeded CL by 2.6 points [0.7, 4.4] for files and 5.1 [3.1, 7.3] for laptops, MLBA_avg by 1.8 [-0.9, 4.6] and 6.9 [3.4, 11.0]. For laptops, overlap rose with them, by 2.3 and 4.2 points, both clear.
- Summing the comparisons over every rival cost 5.6 and 6.9 points against CL for files and nothing clear for laptops, so how sensitivity scales with set size matters, which is test A1a of H1.
- Every published interval of the earlier pipeline is reproduced once the benchmark is held at its full sample value (Amendment 1).

Story so far. Comparing options with each other, averaged over rivals, predicts at least as well as valuing each option on its own. The open question for H1 is what happens to that comparison as sets grow.

### 9 Oct 2026. H1, narrowing with set size (development data, pre-specified tests). Laptop values superseded by Amendment 4, see 14 Oct

1000 participant resamples, plan with Amendments 1 and 2 (`results/h1`). Decisions follow Section 7 of the plan.

| Test | Files | Laptops |
|---|---|---|
| B1 mean concentration gap, points | -25.7 [-29.7, -21.8], clear | -16.7 [-20.1, -13.1], clear |
| B1 slope of the gap per doubling of options, points | +2.9 [-1.9, 6.9], not clear | -12.0 [-16.0, -8.0], clear |
| B1 decision | not supported (no slope) | supported |
| A1a CL, sensitivity rising with set size (beta +0.5) minus constant | -3.0 [-5.4, -0.8], clear, opposite to the prediction | +2.0 [-1.6, 5.5], not clear |
| A1a CL, sensitivity falling with set size (beta -0.5) minus constant | +0.4 [-2.0, 2.5], not clear | -13.6 [-18.4, -8.9], clear, as predicted |
| A1a decision | not supported | not supported (rising sensitivity not clear) |
| A1b EBA minus CL, 32 option trials minus 8 option trials, bits | +0.38 [0.19, 0.61], clear | -0.42 [-0.63, -0.23], clear, opposite |
| A1b decision | supported by the rule, but EBA trails CL by 32 points overall | not supported |

Reading.
- People narrowed. Their choices were far more concentrated than CL's held out predictions in every trial but one, by 25.7 points of the set for files and 16.7 for laptops, and for laptops the gap widened with every doubling of the options (from +2.0 and -6.2 points at 8 options to -31.9 and -18.4 at 32).
- The noise account failed. Lowering the sensitivity as sets grow cut laptop completeness in all three model families (by 13.6 points for CL) and left files unchanged.
- Uniform sharpening failed too. Raising the sensitivity with set size hurt files by 3.0 points in all three families and did not clearly help laptops. Summing comparisons over rivals (beta +1) was worse still, as in R0.
- Screening on the rubric's attributes in the rubric's order predicted badly, 32 to 34 points below CL in both tasks (and below a uniform guess for files). Its deficit shrank from 8 to 32 file options, which passes the pre-specified A1b rule for files, but it never overtook CL, so we do not read it as evidence for this screening mechanism.

Story update. Narrowing is real, but it is not a sharper version of the valuation the models use, neither a set size sensitivity nor rubric weighted screening reproduces it. People concentrated on options for reasons the coded attributes and the rubric do not carry. That moves the weight of the story to H3 (what people screen on lies partly outside the coding) and H4 (whether narrowing helps or hurts depends on how well those cues track value). The 16 option trials were the hardest to predict in both tasks (CL completeness -7.7 for files and 41.6 for laptops at 16 options), where earlier work found the filename conflict in T4_H16 and laptop H in LT3_L16.

Caution. Development data. The behavioral level of the gap was known before the plan, its slope and every algorithmic cell were not.

### 9 Oct 2026. R0 and H1 verified twice

A second run reproduced every saved value, and an independent reimplementation written from the plan without the package agreed on every compared value, with a largest difference of 2e-12 (`verification/`). The computations behind R0 and H1 are correct.

The reimplementation raised issues with the inputs (`verification/README.md`).
- The laptop screenshots of the earlier CHI draft (trials 1 and 2) differ from the analyzed laptop definitions, while the file screenshots match. The designers' scores follow the analyzed definitions. In laptop trial 1, F drew as many choices as D (29.5 percent each), which fits the screenshot version better, since under the analyzed definitions D beats F on memory and drive type for $20 more. Which displays respondents saw is open until the survey export is checked.
- The designers' laptop scores contradict the attributes in the two 32 option trials (identical laptops scored 75 and 40, or 33, 58 and 52), while they track the attributes closely elsewhere.
- The file A1b result depends on the median convention for an even number of options, so it stays fragile and is not read as support.

Status. File results stand. Laptop results are provisional until the displays and the laptop rubric are confirmed.

### 9 Oct 2026. Exploratory. Laptop rubric shares from laptop trials 1 to 4. Withdrawn by Amendment 4

Verified twice (second run identical, independent agreement on every value). `results/superseded/laptop_rubric_1to4`. It probed a symptom of the wrong laptop definitions and is kept only for the record.

- Shares without trials 5 and 6: memory 0.31, graphics 0.30, storage 0.27, processor 0.12, battery and brand 0 (before 0.28, 0.28, 0.28, 0.14, 0.03 and 0).
- Every model built on the shares predicted unseen laptop trials better, by 1.4 to 2.6 points at every set size. CL 52.9 [41.0, 65.1] against 50.3, DbS_avg 57.8 against 55.3, MLBA_avg 59.2 against 57.2. QR, which reads the rubric directly, stayed at 47.2 and now trails CL clearly (-5.7 [-9.7, -1.9]).
- The comparison models' advantage over CL held (DbS_avg +4.9 [2.8, 7.3], MLBA_avg +6.3 [2.9, 10.4], both clear).
- No H1 decision changed. B1 gap -15.5 [-18.8, -12.1] and slope -11.7 [-15.5, -7.6] per doubling, both clear. CL at beta -0.5 minus beta 0 was -14.0 [-19.2, -9.2], clear, and beta +0.5 minus beta 0 was +1.8 [-1.9, 5.5]. EBA's 32 minus 8 contrast was -0.42 bits, clear and opposite to the prediction.

Reading. The two 32 option laptop rubrics add noise rather than signal, which fits a scoring or labeling error there. The laptop conclusions do not depend on them.

### 9 Oct 2026. H1 confirmatory, open snack choice data (Amendment 3)

Run once under the frozen amendment, then verified twice (independent blind reimplementation agreeing on every value, largest difference 4e-8 for the elasticities and 2e-15 elsewhere, and an identical second run). `results/h1_confirmatory`. Differences in held-out log score are given as percent changes in the likelihood of the chosen item.

| Test | Result | Decision |
|---|---|---|
| C2, CL beta -0.5 minus beta 0 (noise) | -10.3% [-12.0, -8.7], clear, as predicted | supports H1 |
| C2, CL beta +0.5 minus beta 0 (sharpening) | -1.6% [-3.1, 0.2], not clear | does not support H1 |
| C2 overall | | not supported |
| C3, elasticity of CL's sensitivity, all items | +0.18 [0.12, 0.25], clear | sensitivity rises |
| C3, elasticity among looked-at items | +0.16 [0.09, 0.24], clear | noise rejected |
| C3, all minus looked-at | +0.02 [-0.00, 0.04], not clear | narrowing of the considered set not supported |
| C1, concentration across screen positions | mean gap -0.3 points [-0.8, 0.1], slope -0.2 [-0.7, 0.3] | not supported |

Descriptives. The share of the display looked at fell from 82.5 percent at 9 items to 41.4 percent at 36 (known from the original publication), and the chosen item had been looked at in essentially every trial. CL's sensitivity fitted to each set size alone rose from 1.68 at 9 items to 2.19 at 36. Sharpening at beta +0.5 helped the comparison models (DbS_avg +2.7% [0.9, 4.8], MLBA_avg +2.9% [1.1, 5.0], both clear). With one subjective value per item, CL predicted unseen set sizes best (3.8 times chance, against 3.5 for DbS_avg and 2.6 for MLBA_avg).

Reading.
- The noise account failed again, now in independent data with random displays and no confound between set size and content. Letting sensitivity fall with set size cost a tenth of the likelihood of every choice.
- Choices grew sharper, not noisier. Sensitivity to liking rose with set size, by about 30 percent from 9 to 36 items, while attention covered a shrinking share of the display. The pre-registered contrast at beta +0.5 overshot this modest rise for CL, which is why C2 fails by its rule. We report that decision as it stands and the elasticity beside it.
- The rise is not produced by narrowing what is considered. Among the items people looked at, sensitivity rose almost as much as over all items. People inspected less of the display but chose more selectively among what they inspected, which fits satisficing search that stops on a well-liked item (Thomas et al. found probabilistic satisficing fitted these choices well).
- Positions did not attract choices beyond chance, so the extra concentration of the development data (B1) is not a screen position effect.

Limits stated with the result. The looked-at version uses gaze recorded during the choice, so it is a measurement, not a prediction. What people look at depends partly on value, so C3 cannot fully separate narrowing from value-guided search. The C2 decision depends on the grid spacing of beta. Bootstrap p values of zero mean p < 0.002.

Story update. H1 becomes "larger sets make choices sharper, not noisier". The noise half holds in laptops and snacks. The sharpening half holds as a measured elasticity in snacks and as growing concentration in laptops, but not as the coarse pre-registered contrast.

### 14 Oct 2026. Laptop results after Amendment 4, verified twice

The laptop task was rerun with the storyboard definitions of trials 5 and 6 (Amendment 4), with the same analyses and seeds (`results/baseline`, `results/h1`). A second run reproduced every saved value. The independent implementation, rerun blind on the corrected definitions, agreed on every compared value, with a largest difference of 2e-12. File results are identical to those of 9 October. The laptop values of the 9 October entries are superseded.

R0, laptops, held out completeness.

| QR | CL | DbS_avg | MLBA_avg | DbS_sum | MLBA_sum |
|---|---|---|---|---|---|
| 47.2 | 55.4 [43.1, 67.9] | 59.3 | 59.7 | 57.9 | 56.9 |

- Comparing options with each other, averaged over rivals, still predicted unseen laptop trials better than CL. DbS_avg gained 3.9 points [1.6, 6.5], clear. MLBA_avg gained 4.3 [0.3, 8.7], not clear, since its basic interval touches zero. Overlap rose with both, by 2.3 and 3.8 points, both clear.
- QR trailed CL by 8.2 points [4.6, 12.1].

H1, laptops.

| Test | Result | Decision |
|---|---|---|
| B1 mean concentration gap | -14.1 points [-17.3, -10.9], clear | |
| B1 slope of the gap per doubling of options | -12.8 [-16.6, -8.7], clear | supported |
| A1a CL beta +0.5 minus beta 0 (sharpening) | +3.8 [0.4, 7.5], clear, Holm p 0.032, overlap the same way | |
| A1a CL beta -0.5 minus beta 0 (noise) | -14.4 [-19.4, -9.6], clear, p < 0.002 | |
| A1a decision | | supported |
| A1b EBA, 32 minus 8 options | -0.63 bits [-0.92, -0.36], clear, opposite to the prediction | not supported |

Reading.
- With the displays people actually saw, the laptop task tells the same story as the snack data. Letting sensitivity fall with set size cost 14.4 points of completeness, and letting it rise gained 3.8. The A1a decision moves from not supported to supported. The sharpening margin is modest (Holm p 0.032, with a Monte Carlo error near 0.008 at 1000 resamples).
- The comparison models gained less from sharpening, DbS_avg 2.7 [-0.4, 6.0] and MLBA_avg 0.9 [-1.8, 3.8], neither clear.
- Concentration beyond CL grew with set size, from gaps of +5.6 and -5.2 points at 8 options to -29.6 and -19.3 at 32.
- Files are unchanged. Sharpening hurt file predictions by 3.0 points and noise was not detected, so the two tasks still disagree on A1a.

Story update. In laptops, H1 now holds on both levels. Choices concentrated beyond CL's predictions more as sets grew, and sensitivity rose rather than fell with set size, as in the snack data. Files show the concentration but not the sharpening.

Correction recorded. The evidence figures in item 2 of Amendment 4 were in sample fits. Held out, the storyboard definitions still predict better, 1.02 against 0.94 bits in trial 5 and 0.96 against 0.89 in trial 6 (correction note in the plan, recomputed by the package and the verifier).

### 14 Oct 2026. Overload, decision time and the search model (development data, exploratory), verified twice

`results/overload` and `results/process_dev`, notebook `02_overload_time_search.ipynb`. Every model has one parameter and the fixed rubric weights, and every prediction is held out (leave one trial out within task). 1000 participant resamples. A second run reproduced every saved value, and an independent implementation written blind from `verification/VERIFY_BRIEF_PROCESS.md` agreed on every value (points within 1e-6, documented tolerances for rank statistics in `verification/compare_process.py`).

Overload index (mean of six aligned post-trial answers, scale points 1 to 5).
- One factor. Alpha within participants 0.87 [0.84, 0.90], first component 62 percent. Question 6 was answered as uncertainty (its printed anchors run the other way).
- Files. Against people's own average, +0.05 and +0.13 at 8 files, +0.29 and +0.14 at 16, +0.50 and +0.32 at 32. From 8 to 32 files, +0.33 [0.21, 0.45], regret +6.3 [0.8, 11.5] points, time +52 percent [32, 74].
- Laptops. Every display below people's own average (-0.33 to -0.12). From 8 to 32 laptops, overload +0.09 [-0.03, 0.21], regret -10.6 [-14.7, -7.0], time +15 percent [5, 28].
- More information per option added no overload (-0.09 and -0.02) and slowed choices by 19 and 14 percent.
- Within person and display, overload went with time (+0.20 files, +0.15 laptops, clear) and weakly with regret (+0.05, +0.13).

Time (the per step figures first logged here were withdrawn on 14 Oct, see below).
- Held out time law (start up time plus time per step, fitted on five displays). Typical error, files and laptops, constant 25.6 and 10.2 percent, CL 22.7 and 13.5, DbS and MLBA 32.0 and 12.5, EBA 22.3 and 12.5, sample until return 18.8 and 10.8. No difference from the constant is clear.

Choices, held out completeness (files, laptops). QR 21.9, 47.2. CL 29.5, 55.4. DbS_avg 32.1, 59.3. MLBA_avg 31.4, 59.7. EBA -2.7, 14.0. Sample until return on CL values 29.4, 56.2 (minus CL -0.1 [-0.3, 0.2] and +0.8 [0.3, 1.4], clear for laptops). On DbS values 31.9, 59.7 (minus DbS -0.2 and +0.4, not clear). On MLBA values 31.2, 59.7. Accept in reading order -39.5, 27.2.

Premonition (forecasts from the display and fits to other displays only).
- Rank correlation with overload within files. Options 0.96, entropy of CL 0.89, of DbS and MLBA 0.94, expected looks of sample until return 0.89. Only MLBA's entropy beat the constant in held out error (-0.064 [-0.119, -0.003]).
- Within laptops, 0.53 to 0.66, none better than the constant. Over both tasks, no model forecast better than the constant. The number of attributes shown separates the tasks only because file displays show fewer attributes.
- Expected regret against people's regret, no model better than the constant (5.8 regret points over both tasks). For laptops sample until return came closest (4.9 against 5.7, not clear).

Reading.
- Overload is display specific. The same people were overloaded by large file displays and not by large laptop displays, so set size alone cannot be the cause.
- No model's work explains decision time on these displays (corrected on 14 Oct, see below).
- Choices cannot tell processes apart. With the values fixed, the search model predicts the same choices as the static model, so process models have to be judged on time, search and overload.
- Premonition works within a task. Twelve displays cannot show more, which moves the test to the snack data, with thousands of displays.

Caution. Development data, exploratory. The trial order is unknown, so set size may be confounded with practice in time and overload. Two file displays include timeouts at the 30 minute limit (1.1 percent of answers on each), recorded with the largest regret. Leaving them out changes no result.

### 14 Oct 2026. The search model on the snack data (Amendment 5, confirmatory), run once and verified twice

Run once under the frozen amendment (plan digest 5a073cf3), `results/search_confirmatory`. A second run reproduced every saved value. The independent implementation, written blind from `verification/VERIFY_BRIEF_SEARCH.md` and tested on simulated data before the real data were opened, agreed on every value and decision, with a largest difference of 3e-7 outside the rank correlations and 4e-4 for them. 1000 participant resamples. Every prediction comes from fits to the other three set sizes.

| Test | Result | Decision |
|---|---|---|
| S1, growth of search with set size (elasticity) | people 0.51 [0.46, 0.55], model 0.48 [0.46, 0.51], difference -0.02 [-0.07, 0.03] | supported |
| S2, held out log response time, mean absolute error | model 0.552, CL 0.543, DbS and MLBA 0.550. Model minus CL +0.009 [-0.003, 0.021] | not supported |
| S3, foreseeing slow decisions, within person rank correlation | model's looks +0.084 [0.039, 0.129], CL's spread of choices +0.043 [-0.004, 0.092], difference +0.041 [0.030, 0.053] | supported |
| S4, foreseeing a miss of the best item, bits per trial above a constant | model +0.042 [0.017, 0.066], CL +0.041, DbS -0.064, MLBA -0.441. Model minus CL +0.001 [-0.002, 0.004] | no direction predicted |
| S5, choices, model minus CL | -0.3 percent [-0.7, 0.0], within the 2 percent margin, neither better nor worse | supported |

Descriptive.
- Distinct snacks looked at, 7.4, 10.4, 13.0 and 14.9 at 9, 16, 25 and 36 items. The model predicts 2.1, 2.8, 3.5 and 4.2, about 3.6 times fewer at every set size.
- Other forecasts of slow decisions. Rating of the best item -0.32 [-0.36, -0.28], spread of the ratings -0.17, MLBA's spread of choices -0.11.
- Choices, times chance. Model 3.83, CL 3.84, DbS 3.55, MLBA 2.62. Model minus DbS +8.0 percent [5.1, 11.0], minus MLBA +46.1 [30.6, 64.2].
- The elasticity of CL's sensitivity that the model implies is -0.01 [-0.03, 0.01], against the +0.18 observed in Amendment 3.

Reading.
- People search, and their search grows like the square root of the number of options, the signature of the birthday problem that the model was built on. This was predicted before the model met these data. Looking at every item would grow in proportion.
- The form of the stopping rule is right and its level is not. People look at 3.6 times as many items before they choose. Either they do not stop at the first return to an option, or their attention is less drawn to value than their choices are.
- Time was not fitted. The model's expected looks did not time single decisions better than counting options.
- What speeds a single decision is an option that is good in absolute terms. The model reads only differences in value, so it cannot produce this, and neither can CL's spread of choices.
- Choices again cannot separate the processes. The model predicted unseen choices as well as CL and better than DbS and MLBA, as the design check expected.

Story update. People choose by a search whose length grows like the square root of what is on offer, and a one parameter model of that search is as accurate as the best static model and alone predicts how search grows. No model, ours included, times single decisions. It still lacks an absolute standard. A search that stops on an option that is good enough would lengthen searches, speed decisions with a clearly good option and say when overload should arise, when nothing on display is good enough. That idea came from these data, so its test needs new data.

### 14 Oct 2026. The per step time argument withdrawn

The claim that DbS and MLBA would need 7 to 12 ms per comparison, CL about one fixation per value and the search model about 1 to 2.5 s per value read (and, on the snack data, a steady 1.1 s per look) came from dividing people's time by each model's steps. That is not a fit to people's times. It assumes that every second goes to the model's steps.

Fitted to people's typical times on all six displays of a task (start up time plus time per step, both nonnegative, least squares, checked by two methods to 4e-14), every model's law put most of the time into the start up constant.

| Model | Files, start up s | Files, ms per step | Files, variation explained | Laptops, start up s | Laptops, ms per step | Laptops, variation explained |
|---|---|---|---|---|---|---|
| CL and QR | 49.1 | 205 | 0.53 | 31.3 | 42.4 | 0.39 |
| DbS and MLBA | 52.6 | 5.69 | 0.45 | 33.1 | 0.92 | 0.22 |
| EBA | 54.9 | 30.4 | 0.21 | 32.1 | 25.2 | 0.30 |
| Sample until return | 47.6 | 797 | 0.42 | 28.4 | 272 | 0.52 |

Held out, no law predicted a display's time better than a constant (see the 14 Oct development entry), and on the snack data none timed single decisions better than counting options (S2). The work of these models does not explain how long people took, so decision time does not rank them. The per step statements are removed from the notebook, the figures and the README. The ratios stay in the summaries as recorded values only.

### 14 Oct 2026. Exploratory. A search model with an absolute standard, two look satisficing (snack data), verified twice

`results/two_look_snack`, notebook 02 section 9. The snack data suggested the model, so this is exploratory, and its test is to be frozen on new data. A second run reproduced every saved value, and an independent implementation written blind from `verification/VERIFY_BRIEF_TWO_LOOK.md` agreed on every value (largest difference 3e-7 outside rank statistics, 5e-5 for them). Resampling as in Amendment 5, so every replicate pairs with the confirmatory one.

The model. Attention is drawn to attractive options as in sample until return. A look registers, the option looks good enough at that moment, with probability exp(lambda (v - v_top) / 2), v_top the top of the rating scale, and the first option with two registered looks is chosen. Attention and registration take half the sensitivity each, so choices equal sample until return at the same sensitivity and no parameter is added. The number of looks now depends on how good the options are in absolute terms.

| Measure | Two look satisficing | Sample until return | Other |
|---|---|---|---|
| Held out log time error | 0.528 [0.484, 0.568] | 0.552, difference -0.025 [-0.044, -0.006], clear | CL (number of snacks) 0.543, difference -0.015 [-0.034, 0.002], not clear |
| Rank correlation with time, within person | +0.278 [0.241, 0.315] | +0.084, difference +0.194 [0.156, 0.233], clear | rating of the best snack -0.322, model plus it -0.044 [-0.072, -0.020] |
| Distinct snacks looked at, 9 to 36 | 3.7 to 7.4 | 2.1 to 4.2 | people 7.4 to 14.9 |
| Growth of search (elasticity) | 0.50 [0.47, 0.51] | 0.48 | people 0.51 [0.46, 0.55] |

Sensitivity. With attention that ignores value and registration alone carrying it, the correlation with time was +0.303 and the level of search matched people at 9 and 16 snacks, but search grew too fast (elasticity 0.69) and the held out time error was 0.543, no better than CL.

Reading.
- An absolute standard, added without a new parameter, lets the search model foresee slow decisions (+0.28 against +0.08) and time single decisions better than sample until return. It still does not clearly beat counting the snacks on held out time.
- It carries most, though not all, of the link between the best snack's rating and time. The verifier noted that with v_top at the top of the scale, the expected looks largely restate whether any snack on display is rated at the top.
- Searches are still about half as long as people's, at every set size, while their growth stays right.

Story update. The missing absolute standard is real and can be built into the search without a new parameter. Whether it holds beyond the data that suggested it is the next confirmatory test.

### 14 Oct 2026. Two look satisficing on three gaze datasets (Amendment 6, confirmatory), run once and verified twice

`results/two_look_gaze`, notebook 02 sections 10 and 11. Run once under the frozen amendment (plan digest 634cdbb8). A second run reproduced every saved value. The independent implementation, written blind from `verification/VERIFY_BRIEF_GAZE.md` and tested on simulated data only before the freeze, agreed on every test and on every value that enters one (largest differences 2e-8 for sensitivities, 1e-9 for time errors, 4e-8 for choices, 6e-17 for rank correlations). The comparison flagged eight descriptive entries, all in time laws whose best fit lies on a bound. There the package returns the bound exactly and the independent code a value within 1e-11 of it, so p values that count replicates at zero differ (laws of Krajbich 2011 and Folke, and two per dataset time differences in Folke's data), and the basic interval of Folke's sample until return start up time differs by 2e-6 seconds. Replicate by replicate the laws agreed within 1e-5. 1000 participant resamples, ten folds of participants.

Data. Krajbich, Armel and Rangel (2010), two foods, Krajbich and Rangel (2011), three foods, and Folke, Jacobsen, Fleming and De Martino (2016), three snacks with confidence ratings, as prepared by Thomas, Molter, Krajbich, Heekeren and Mohr (2019, MIT license).

| Test, mean of the three datasets | Result | Decision |
|---|---|---|
| G1, held out log time error, two look satisficing minus sample until return | +0.0045 [0.0001, 0.0106], not clear | not supported |
| G1, two look satisficing minus one constant time | -0.0002 [-0.0054, 0.0056] | |
| G2, rank correlation of expected looks with time, within person | two look satisficing +0.194 [0.162, 0.229], sample until return +0.210 [0.174, 0.243], difference -0.016 [-0.040, 0.011] | not supported |
| G3, choices, two look satisficing (equal to sample until return) minus CL | -0.04 percent [-0.15, 0.03] | supported, non inferior |

Secondary and descriptive.
- Choices. Two look satisficing beat DbS by +3.0 percent [1.2, 4.5] and MLBA by +6.9 percent [5.0, 8.9].
- Display features. The value of the best item correlated -0.204 with time, the gap between the two best -0.185 and the mean value -0.137. Two look satisficing plus the best item's value -0.009, clear, so the best item's value alone did slightly better.
- By dataset. Sample until return's looks foresaw slow decisions clearly better in Folke's snacks (difference -0.060), and two look satisficing leaned the other way in Krajbich's three foods (+0.034, not clear).
- Confidence (Folke). Expected looks of either model and the spread of CL's choice probabilities each correlated about -0.28 to -0.29 with confidence, with no difference between the models.
- Time laws. Two look satisficing's law put nearly all time into its start up constant (0.001 to 0.07 seconds per look), so G1 failed by its form as much as by its ranking.

Exploratory, after the tests, from the same resamples (notebook 02 section 10). Sample until return timed held out decisions better than one constant time by about 1 percent of the error, -0.0047 [-0.0080, -0.0021], clear in both Krajbich datasets and not in Folke's. Its looks foresaw slow decisions as well as the best item's value (sum +0.007, not clear) and better than the gap between the two best (+0.025, clear). Its law put all time into looks, 0.76 and 0.77 seconds per look in the Krajbich datasets, on the bound of no start up time, and was unstable in Folke's slower task. The independent implementation's resamples reproduce the time difference.

Reading.
- The absolute standard did not generalize. It helped on the snack data, where large displays often lacked a top rated snack, and not on displays of two or three foods.
- The plain search model held up on new data. Sample until return chose as well as CL and, over the three datasets taken together, better than DbS and MLBA, and its looks ranked slow decisions above fast ones within each person in every dataset.
- No model times single decisions much better than a constant once a person's overall speed is unknown.

Story update. The paper proposes sample until return. Two look satisficing is reported as an extension that was built on one dataset, frozen and rejected on three others.

### 31 Oct 2026. The search model's stopping rule on the data of Caplin, Dean and Martin (Amendment 7, confirmatory), run once and verified twice

`results/stopping_cdm`. Run once under the frozen amendment (plan digest 876dbadf). A second run reproduced every saved value. The independent implementation, written blind from `verification/VERIFY_BRIEF_CDM.md` and tested on simulated data only before the freeze, agreed on every decision and value (largest differences 3e-7 on percent scales, 5e-9 for sensitivities). One interval end of a descriptive percent differed by 1.2e-6 on a value of 31, a relative difference of 4e-8. 1000 participant resamples, ten folds of participants within each experiment.

Data. Sums of money written out as 3 or 7 additions and subtractions, in sets of 10, 20 and 40, every selection on the way to a choice recorded. Experiment 1 paid the final choice, Experiment 2 the selection held at a random moment.

| Test | Result | Decision |
|---|---|---|
| C1, what ends a search, value rule minus share rule (mean of the two experiments) | +0.016 bits per selection [-0.008, 0.044] | undecided |
| C2, place of the first selection in Experiment 2 (0 top, 1 bottom) | 0.361 [0.309, 0.412], minus one half -0.139, minus the model's 0.508 -0.146, both clear | list order matters |
| C3, choices, sample until return minus CL | +0.41 percent [-0.06, 0.84] | supported, non inferior |

Secondary and descriptive.
- Each experiment. Value minus share +0.009 (standard) and +0.022 (choice process), neither clear. First selection's place 0.38 and 0.36. Sample until return minus CL +0.16 and +0.66 percent, the second clearly better.
- Choices. Sample until return beat MLBA by 58 percent, and DbS by 6 percent (not clear).
- Search time. The time of the final selection grew with the number of options with elasticity 0.29 and 0.24, and the model's expected looks with 0.23 and 0.24. No time law beat the number of options held out.
- Unsubmitted rounds. None in Experiment 1. In Experiment 2, 1 to 20 percent by cell, most in large and complex sets, so leaving them out shortens the longest searches.

Reading.
- The test of the stopping rule came out undecided. People did not stop clearly by the worth of what they held, unlike the satisficing the original paper suggested, and the test cannot favor the model's first return (stated before the run).
- Where values must be worked out, people partly read from the top of the list, which the model's attention, drawn by value, does not capture.
- Choices again matched CL, now on a fifth dataset, where values are money worked out from arithmetic.

### 31 Oct 2026. Forecasting choice overload on Caplin, Dean and Martin and on Reutskaja and colleagues (Amendment 8, confirmatory), run once and verified twice

`results/overload_premonition`, notebook 02. Run once under the frozen amendment (plan digest 876dbadf). A second run reproduced every saved value, and the independent implementation, written blind from `verification/VERIFY_BRIEF_OVERLOAD.md` and tested on simulated data only before the freeze, agreed on every value and decision (largest difference 2e-8, areas under the curve identical). 1000 participant resamples. Every forecast of a set size comes from fits to the panel's other set sizes.

Overload is the growth of missed best options with set size, per doubling of the set.

| Dataset | People | CL | Compensation | Sample until return | O1 | Sample until return minus the near best count (area under the curve) | O2 |
|---|---|---|---|---|---|---|---|
| Caplin, Dean and Martin (sums worked out) | +0.156 [0.126, 0.183] | -0.011 | -0.082 | -0.018 | overload, beyond every forecast | -0.007 [-0.029, 0.017], all forecasts near 0.55 | not supported |
| Reutskaja (three seconds) | +0.099 [0.075, 0.122] | +0.111 | +0.057 | +0.098 | overload, foreseen by fixed sensitivity | +0.093 [0.073, 0.116], 0.70 against 0.61 | supported |

- Caplin, Dean and Martin. Misses rose from 9 to 33 percent (sums of three steps) and from 35 to 73 percent (seven steps) between 10 and 40 options, while every model foresaw flat or falling misses. People's growth exceeded compensation's (+0.238) and CL's (+0.167), both clear.
- Reutskaja. Misses rose from 28 to 47 percent between 4 and 16 snacks. Sample until return foresaw the growth almost exactly (error 0.001) and the trials of the misses better than the simplest display feature. The share of misses whose best snack was never fixated rose from 25 to 64 percent.

Exploratory, after the run (`results/overload_all_models`, same resamples). DbS and MLBA, which compare options by rank, foresaw misses growing in Caplin, Dean and Martin (+0.234 and +0.215, the closest to people's growth) but at levels far above people's (MLBA 46 to 92 percent), and in Reutskaja they foresaw twice the growth people showed (+0.21). Trial by trial they did worse than sample until return in both.

Reading.
- Overload appeared in both datasets where judging was costly, as predicted, and people did not compensate there as they did with snacks seen at a glance.
- Under time pressure the search model forecast overload, how much worse choices got with more options and on which trials. Where values had to be worked out, misses grew for reasons the values do not show, and no value based model foresaw them.

Story update. The paper has its evidence of overload prediction, with a boundary. People compensate when an option can be judged at a glance, overload appears when judging is costly, and the search model foresees it when the cost is time but not when it is working out the values.

### 1 Nov 2026. Fitted and held out forecasts of human overload, every model including mixed logit (exploratory), run twice

Every model was also fitted once to all set sizes (`results/overload_fitted`, `results/overload_snack_fitted`), and mixed logit was added throughout. The full table is `results/overload_fitted_vs_heldout.md`.

| Growth of misses per doubling | People | CL | Sample until return | Mixed logit | CL with compensation | DbS | MLBA |
|---|---|---|---|---|---|---|---|
| Caplin, Dean and Martin, held out | +0.156 | -0.011 | -0.018 | -0.004 | -0.082 | +0.234 | +0.215 |
| Caplin, Dean and Martin, fitted | | +0.036 | +0.031 | +0.037 | -0.015 | +0.220 | +0.225 |
| Reutskaja, held out | +0.099 | +0.111 | +0.098 | +0.101 | +0.057 | +0.208 | +0.209 |
| Reutskaja, fitted | | +0.113 | +0.105 | +0.106 | +0.076 | +0.183 | +0.180 |
| Snack, held out | +0.010 | +0.062 | +0.051 | +0.056 | +0.004 | +0.134 | +0.133 |
| Snack, fitted | | +0.048 | +0.040 | +0.044 | +0.004 | +0.109 | +0.113 |

- Fitting to every set size moves forecasts a little and never changes the order of the models or the O1 verdicts.
- Mixed logit predicted choices best in every dataset by a small margin (bits per choice, e.g. -2.464 against -2.509 for sample until return fitted on Caplin, Dean and Martin) but forecast overload as CL does. Differences in sensitivity between people do not explain overload.
- Verification. Second runs identical, original samples recomputed blind (`verification/independent_mxl`).

### 1 Nov 2026. H3, the independent filename cue (Amendment 9, confirmatory), run once and verified twice

Each rule (CL, DbS_avg, MLBA_avg) was given one cue outside the coding, how close a filename's words lie to "which file is most worth taking" in frozen GloVe vectors, and its held out gain was set against the spread across rules, in file selection (`results/h3_cue`).

| | All trials | Low information | High information |
|---|---|---|---|
| Spread across rules, points | +2.6 [1.3, 4.7] | +1.2 [0.3, 3.3] | +3.8 [1.8, 7.3] |
| Gain from the cue, CL, points | -2.1 [-5.5, 0.6] | +1.8 [-1.6, 5.0] | -5.7 [-11.4, -1.4] |
| A3b, gain minus spread, CL | -4.7 [-8.5, -1.9], clear | +0.6 [-3.5, 4.1] | -9.6 [-16.2, -4.8], clear |

- A3b was clearly negative, so H3 is not supported. DbS_avg and MLBA_avg gave the same picture (-4.5 and -4.3).
- The cue weight was clearly positive in the pooled fits (+1.2 per unit of cosine for CL). People did lean toward files whose names matched the task's goal, but a weight learned on five trials did not carry over to the sixth. It helped a little where only the names and extensions were shown and hurt once the full listing was shown.
- Verification. The second run was identical (point, replicates, table, summary apart from time). A blind implementation from the amendment text agreed within 1e-6 on the original sample and replicates 0 to 49.

Reading. A cue outside the coding is real but does not generalize across displays, so the choice of rule and the choice of input mattered about equally little here. Story update. H3 does not carry the paper. The weight stays on forecasting overload from search.

## Setup facts checked against the data

- The data file holds recruitment platform identifiers (a Prolific ID column and Prolific screening questions) and a nine question security knowledge quiz (SQL injection, SSL, firewall, proxy, DDoS, IDS, honeypot, TCP, Nmap). The Study Design should describe recruitment and screening from these.

## Open decisions

- Laptop displays and rubric. Resolved by Amendment 4 from the storyboard (10 Oct), results rerun and verified (14 Oct).
- Trial order. The data file does not record whether the six trials of each task were shown in a fixed order. If they were, set size is confounded with practice in decision time and overload ratings. Asked of the study team on 14 Oct.
- Post-trial question 6 asks "How uncertain are you" with anchors running from very uncertain (1) to very certain (5). Its direction is settled from its correlations with the other five questions before the overload index is built.
- H2 confirmatory dataset. Hensher (2006) was checked on 1 Nov, its archive holds no choices (only design features and counts of ignored attributes), so it cannot test H2. Candidate, the job candidate study in Psychonomic Bulletin and Review (2025), two options with 3, 4 or 5 attributes within participants, OSF data awaited from the study's author. Fix by amendment before inspecting outcomes.
- H3 independent cue. Resolved by Amendment 9, run 1 Nov, not supported.
- H4. Set aside for a separate paper (Amendment 9).
