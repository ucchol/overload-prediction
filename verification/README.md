# Verification

Every reported result passes two checks before it is shown.

1. Second run. The experiment is rerun from a fresh process, and its summary, per trial values and every bootstrap replicate must be identical to the saved ones.
2. Independent reimplementation. A separate implementation recomputes the result from the raw data. It is written from the plan and a brief alone (`VERIFY_BRIEF.md` for the development data, `VERIFY_BRIEF_SNACK.md` for the snack data), with the raw data, the stimulus files and the attribute coding, and without access to this package or to any of its results. Only after it has produced its numbers are the two compared, value by value, by `compare_independent.py`, `compare_laptop_rubric_1to4.py` and `compare_h1_confirmatory.py`.

A disagreement beyond rounding blocks the result until its cause is found. The briefs record the details the plan left open and how they were fixed, so a third implementation can start from the same text.

| Folder | Implements | Rerun |
|---|---|---|
| `independent/` | R0, H1 and the laptop rubric check on the development data | `sh verification/run_independent.sh PATH_TO_CSV`, about two minutes |
| `independent_snack/` | the confirmatory H1 test of Amendment 3 on the snack data | `sh verification/run_independent_snack.sh`, about ten minutes |
| `independent_process/` | the overload index, decision time, search models and premonition on the development data | `sh verification/run_independent_process.sh PATH_TO_CSV`, about ten minutes |
| `independent_search/` | the confirmatory search model tests of Amendment 5 on the snack data | `sh verification/run_independent_search.sh`, about twenty minutes |

Both implementations are archived exactly as they were written, with their results. The runners copy data only into temporary folders that are removed afterwards.

## R0 and H1, 9 October 2026

- Second run. Identical for the baseline and for H1 (`summary.json`, `point.json` and `replicates.jsonl`).
- Independent reimplementation. Agreement on every compared value, including choice shares, rubric shares, benchmarks, per trial log scores and overlaps, fitted parameters, completeness levels and differences, percentile and basic intervals, p values, clear flags and hypothesis decisions. The largest absolute difference is 2e-12. The reimplementation also reproduces every diagnostic quoted in Amendment 1.

## Issues the reimplementation raised

Inputs.

Issues 1 and 2 were resolved on 10 October by Amendment 4. The study's storyboard shows every laptop display, and the earlier definitions of trials 5 and 6 were wrong. See the entry of 14 October below.

1. Laptop displays. The screenshots of laptop trials 1 and 2 in the earlier CHI draft do not match the laptop definitions analyzed here, which come from the original analysis notebook. In trial 1 every price differs and option D shows 12GB of memory and a 6GB dedicated graphics card instead of 32GB and 4GB, with no drive type. In trial 2 option F shows 16GB and a 2TB HDD at $1,440 instead of 32GB and a 2TB SSD at $1,500, and the brands of D and G differ. The file screenshots match the file definitions exactly. The designers' laptop scores follow the analyzed definitions, not the screenshots. Open until the survey export settles which displays respondents saw.
2. Laptop rubric in the two 32 option trials. Laptops with identical attributes received very different rubric scores (in trial 6, L and BB differ only by $12 in price and score 75 and 40, and in trial 5, D, J and EE are identical and score 33, 58 and 52). The rubric correlates with the rubric weighted attribute score at 0.73 and 0.82 in these trials, against 0.97 or more in the other four. The rubric enters QR, the rubric shares of every model and the planned regret measure.
3. Coding notes. Screen size is shown in the high information laptop trials but is not coded, since the designers' laptop rubric does not score it. Every HDD takes the lowest storage level whatever its capacity, as in the designers' rubric. Price is coded but constant within every trial and has share zero.

Specification.

4. The median aspect of elimination by aspects uses the midpoint for an even number of options. With the lower middle value instead, the file A1b contrast is +0.094 bits [-0.053, 0.230] and no longer clear (`independent/sensitivity_median.json`). The A1b file result is therefore reported as fragile.
5. Amendment 1 was adopted after the R0 level intervals were seen. It changes only the descriptive level intervals, which become wider, and leaves every model difference and test unchanged up to a common rescaling.
6. The negative mean gap of B1 was known before the plan, so B1 tests only the slope. In the development data set size is confounded with stimulus content (one trial per set size and information level), which the confirmatory dataset addresses.
7. The two primary A1a contrasts are strongly dependent, so Holm's correction is conservative there. A bootstrap p value of zero is reported as p < 0.002.

## Laptop results after Amendment 4, 14 October 2026

R0 and H1 were rerun with the storyboard laptop definitions (`src/overload/stimuli/laptop_trials.json`).

- Second run. Identical for the baseline and for H1 (`summary.json` apart from its run time stamp, `point.json` and `replicates.jsonl`).
- Independent reimplementation. The verifier first checked that the file task and laptop trials 1 to 4 were unchanged, and that its code reproduced its own earlier laptop results bit for bit on the earlier definitions. It then reran blind on the corrected definitions (`independent/results_storyboard.json`, checks in `independent/storyboard_checks.py`). Agreement on every compared value, largest difference 2e-12. The runner `run_independent.sh` now reproduces this comparison from the repository.
- Rubric against the documented scoring rule, Pearson correlation per laptop trial. Storyboard definitions 0.974, 0.985, 0.979, 0.987, 0.991 and 0.964. Earlier definitions 0.974, 0.985, 0.979, 0.987, 0.728 and 0.763. The groups of identically coded laptops with very different rubric scores disappear.
- The verifier found that the bit figures quoted in item 2 of Amendment 4 are in sample fits. Held out they are 1.019 against 0.939 in trial 5 and 0.958 against 0.895 in trial 6, still in favor of the storyboard. The plan carries a correction note.

`independent/results.json` stays as the record of the 9 October verification on the earlier definitions.

## Overload, decision time, search models and premonition, 14 October 2026

Exploratory analyses of the development data (`results/overload`, `results/process_dev`).

- Second run. Identical (`summary.json` apart from its run time stamp, `point.json` and `replicates.jsonl`).
- Independent reimplementation, written blind from `VERIFY_BRIEF_PROCESS.md`, reusing the verifier's own R0 code. Before the run it checked the race probabilities against closed forms for two options, the birthday problem for equal options (365 options give 24.6166 expected looks), an exact elementary symmetric polynomial computation, simulations of both search models, the elimination by aspects effort by hand and every fitted sensitivity against a dense grid (each search likelihood had a single maximum in all 48,000 resampled fold fits). `compare_process.py` then compared about 3,900 values. Every point value agrees within 1e-6 (largest difference 8e-7, from the search models' sensitivities, which the two optimizers locate to about 3e-7). Interval bounds of completeness agree within 2e-6. Rank correlations agree exactly in the original sample, but in some resamples two forecasts or outcomes equal up to 1e-7 are ordered differently, which moves an interval bound by one rank step (at most 0.007) and a p value by up to 0.014. A difference of two time errors can be exactly zero in a resample, where rounding decides its sign (p values differ by 0.002). No clear flag differs among the 364 compared.

Points the verifier raised.

1. Every participant has a matched choice in both tasks, but two file displays (trials 1 and 6) each include timeouts, 1.1 percent of answers, with no matched option, the 1800 second timer cap and regret 100. They are kept as recorded. Leaving them out moves the regret errors of the premonition by at most 0.12 points and changes no decision.
2. Combining both tasks in the premonition mixes the gap between tasks into every forecast. The number of attributes shown separates the tasks only because file displays show fewer attributes. Within each task it has no value as a forecast.
3. Percentile intervals sit badly for rank correlations over six displays and for mean absolute errors against resampled outcomes (a point at the edge of its own interval). The clear rule gives partial protection, and rank correlations over six displays are read as descriptions.
4. With the number of options as displayed, laptop trial 6 has 30 options. With the intended 32, the rank correlation of options with overload is 0.48 instead of 0.53 for laptops.
5. QR and CL share one effort count, and so do DbS and MLBA, so their time results are identical pairs.

## The search model on the snack data (Amendment 5), 14 October 2026

The independent code was written from `VERIFY_BRIEF_SEARCH.md` and tested on simulated data only, then Amendment 5 was frozen, and only then were the real data analyzed, once by each side.

- Before the run. Closed forms against two items and the birthday problem, a simulation of the search process, recovery of the sensitivity, a synthetic data set in which choices, search and response times come from the model (every test passed there), a null data set in which choices come from CL and times from set size (S2 and S3 correctly not supported), loader edge cases, and checks that every fit is a maximum, including the time law against a brute force grid. The verifier's points before the freeze were settled in the amendment (the global minimum of the time law, S1 reported with the level of search, S3 not specific to the model, S5 also reporting worse).
- Second run. Identical (`summary.json` apart from its run time stamp, `point.json` and `replicates.jsonl`).
- Independent reimplementation. Agreement on every value and every decision (`compare_search_confirmatory.py`). Outside rank correlations the largest difference is 3e-7, on the percent scale of S5, from locating the model's sensitivity to about 1e-8. Rank correlations differ by up to 4e-4 and their p values by one resample, where two forecasts equal to within rounding are ranked in different orders. Rerunning the amendment's exact fitting procedure gives the same sensitivities to within 3e-8 (`independent_search/postrun_checks.json`).
- Points the verifier raised after the run. The model implies no sharpening at all (-0.01, against +0.18 observed). S2 and S3 point in different directions, and the rating of the best item forecasts slow decisions much better than the model's looks.

## Exploratory laptop rubric sensitivity, 9 October 2026 (withdrawn by Amendment 4)

Second run identical. Independent implementation (`independent/sens_laptop_1to4.py`) agrees on every compared value, largest difference 2e-12 (`compare_laptop_rubric_1to4.py`).

## H1 confirmatory on the snack data, 9 October 2026

Run once under Amendment 3. Second run identical. The independent implementation (`independent_snack/`), written from Amendment 3 and `VERIFY_BRIEF_SNACK.md`, agrees on every compared value, with a largest difference of 2e-15 for held-out scores, sensitivities and intervals and 4e-8 for the elasticities, whose profile search stops at a tolerance near 1e-9 (`compare_h1_confirmatory.py`).

Issues the reimplementation raised about Amendment 3.

1. C1 compares choice concentration across screen positions with a value model whose position profile is almost exactly uniform, so it tests position effects only, not narrowing on value.
2. The C2 decision depends on the spacing of the beta grid. The continuous elasticity of C3 peaks near +0.18, between the grid points 0 and +0.5.
3. The looked-at version uses gaze recorded during the choice, so its held-out scores are measurements rather than predictions and are not compared with the all-items version.
4. What people look at depends partly on value, so the C3 difference cannot fully separate narrowing from value-guided search.
5. Bootstrap p values of zero mean p < 0.002.
