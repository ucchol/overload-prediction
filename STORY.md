# Story

Living record of the paper's story, the status of every hypothesis and every result. Updated after each step.

## The story in one paragraph

When displays grow in the number of options and in the attributes shown for each option, people do not integrate everything shown. They adapt (Payne, Bettman and Johnson, 1993), screening options on a few salient cues and choosing among what survives, so their choices narrow rather than become noisier. Choice models are formal statements of how choices are computed, and held out prediction decides between them. Each hypothesis predicts a behavioral signature and the model structure that should predict better where that signature appears. The intruder's file choice and the buyer's laptop choice are the development data, and open datasets give the confirmatory tests.

## Hypotheses and status

| | Behavior | Algorithm |
|---|---|---|
| H1 Narrowing with set size | B1 done. Large gap in both tasks, growing with set size for laptops only. Confirmatory C1 done, no concentration on screen positions. Attention narrowed from 82 to 41 percent of the display | A1a done, noise rejected for laptops, sharpening at beta +0.5 not supported. A1b done, rubric screening predicts poorly. Confirmatory C2 done, noise rejected (-10.3%), beta +0.5 not clear, so not supported by its rule. C3 done, sensitivity rises with set size (elasticity +0.18), also among looked-at items |
| H2 Comparison under sparse information | B2 revealed weights, known for laptops | A2 known on development data, confirmatory dataset to fix |
| H3 Inputs over rules | B3 known | A3a ladder known, A3b independent cue to fix and run |
| H4 Cost depends on cue validity | to specify | to specify |

## Plan versions

| Version | Date | SHA-256 of ANALYSIS_PLAN.md |
|---|---|---|
| 1.0 | 9 Oct 2026 | 6bc0e96185501c9ae5bf2fcd14a6a2fb617466ee3c740d505e4eb773e5c75cd8 |
| 1.0 with Amendment 1 | 9 Oct 2026 | 2953b6ecaf8420cacc3e7112d8effd153e986e4cae1c34887a2d1130729e6ed0 |
| 1.0 with Amendments 1 and 2 | 9 Oct 2026 | f3b624d8c2500f4e711520e60939cf18d83826f49654df232152d3da9bcbaad2 |
| 1.0 with Amendments 1 to 3 | 9 Oct 2026 | bd138487b4209818aacb5e68ffbd5cd8693e4e44461cfc819ed60e9c3ab8d519 |

## Results log

### 9 Oct 2026. Pipeline rebuilt and verified

The package reproduces the earlier pipeline from the raw data. Rubric shares match exactly, and per trial held out and fitted log scores and overlaps of QR, CL, DbS and MLBA match within 1e-6 bits (largest difference 2.5e-8, from the earlier optimizer's tolerance), for the original sample and for bootstrap replicates 0, 1, 499 and 999. Published completeness values round to the same first decimal.

### 9 Oct 2026. R0, the baseline (descriptive, development data)

Held out completeness, 95 percent intervals, 1000 participant resamples (`results/baseline`).

| | QR | CL | DbS_avg | MLBA_avg | DbS_sum | MLBA_sum |
|---|---|---|---|---|---|---|
| Files | 21.9 | 29.5 [20.2, 39.2] | 32.1 | 31.4 | 23.9 | 22.6 |
| Laptops | 47.2 | 50.3 [38.9, 62.0] | 55.3 | 57.2 | 49.6 | 50.0 |

- Averaged comparison scores predicted unseen trials at least as well as CL in both tasks. DbS_avg exceeded CL by 2.6 points [0.7, 4.4] for files and 5.1 [3.1, 7.3] for laptops, MLBA_avg by 1.8 [-0.9, 4.6] and 6.9 [3.4, 11.0]. For laptops, overlap rose with them, by 2.3 and 4.2 points, both clear.
- Summing the comparisons over every rival cost 5.6 and 6.9 points against CL for files and nothing clear for laptops, so how sensitivity scales with set size matters, which is test A1a of H1.
- Every published interval of the earlier pipeline is reproduced once the benchmark is held at its full sample value (Amendment 1).

Story so far. Comparing options with each other, averaged over rivals, predicts at least as well as valuing each option on its own. The open question for H1 is what happens to that comparison as sets grow.

### 9 Oct 2026. H1, narrowing with set size (development data, pre-specified tests)

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

### 9 Oct 2026. Exploratory. Laptop rubric shares from laptop trials 1 to 4

Verified twice (second run identical, independent agreement on every value). `results/exploratory/laptop_rubric_1to4`.

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

## Setup facts checked against the data

- The data file holds recruitment platform identifiers (a Prolific ID column and Prolific screening questions) and a nine question security knowledge quiz (SQL injection, SSL, firewall, proxy, DDoS, IDS, honeypot, TCP, Nmap). The Study Design should describe recruitment and screening from these.

## Open decisions

- Laptop displays and rubric. Check the survey export (Qualtrics file or PDF) for the six laptop displays and the designers' laptop scoring sheet. If the displays differ from the analyzed definitions, recode the laptop task and rerun every laptop result. If only the rubric of trials 5 and 6 is wrong, rerun with corrected scores, or estimate the rubric shares from trials 1 to 4 and drop QR and regret for trials 5 and 6.
- H2 confirmatory dataset. Confirm that Hensher (2006) is openly downloadable, otherwise choose another open dataset that varies the attributes shown for the same options. Fix by amendment before inspecting outcomes.
- H3 independent cue. Choose the external source of the filename cue before running A3b.
- H4. Specify cue validity, the regret contrast and the screening model before running.
