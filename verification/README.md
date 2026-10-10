# Verification

Every reported result passes two checks before it is shown.

1. Second run. The experiment is rerun from a fresh process, and its summary, per trial values and every bootstrap replicate must be identical to the saved ones.
2. Independent reimplementation. A separate implementation recomputes the result from the raw data. It is written from the plan and a brief alone (`VERIFY_BRIEF.md` for the development data, `VERIFY_BRIEF_SNACK.md` for the snack data), with the raw data, the stimulus files and the attribute coding, and without access to this package or to any of its results. Only after it has produced its numbers are the two compared, value by value, by `compare_independent.py`, `compare_laptop_rubric_1to4.py` and `compare_h1_confirmatory.py`.

A disagreement beyond rounding blocks the result until its cause is found. The briefs record the details the plan left open and how they were fixed, so a third implementation can start from the same text.

| Folder | Implements | Rerun |
|---|---|---|
| `independent/` | R0, H1 and the laptop rubric check on the development data | `sh verification/run_independent.sh PATH_TO_CSV`, about two minutes |
| `independent_snack/` | the confirmatory H1 test of Amendment 3 on the snack data | `sh verification/run_independent_snack.sh`, about ten minutes |

Both implementations are archived exactly as they were written, with their results. The runners copy data only into temporary folders that are removed afterwards.

## R0 and H1, 9 October 2026

- Second run. Identical for the baseline and for H1 (`summary.json`, `point.json` and `replicates.jsonl`).
- Independent reimplementation. Agreement on every compared value, including choice shares, rubric shares, benchmarks, per trial log scores and overlaps, fitted parameters, completeness levels and differences, percentile and basic intervals, p values, clear flags and hypothesis decisions. The largest absolute difference is 2e-12. The reimplementation also reproduces every diagnostic quoted in Amendment 1.

## Issues the reimplementation raised

Inputs.

1. Laptop displays. The screenshots of laptop trials 1 and 2 in the earlier CHI draft do not match the laptop definitions analyzed here, which come from the original analysis notebook. In trial 1 every price differs and option D shows 12GB of memory and a 6GB dedicated graphics card instead of 32GB and 4GB, with no drive type. In trial 2 option F shows 16GB and a 2TB HDD at $1,440 instead of 32GB and a 2TB SSD at $1,500, and the brands of D and G differ. The file screenshots match the file definitions exactly. The designers' laptop scores follow the analyzed definitions, not the screenshots. Open until the survey export settles which displays respondents saw.
2. Laptop rubric in the two 32 option trials. Laptops with identical attributes received very different rubric scores (in trial 6, L and BB differ only by $12 in price and score 75 and 40, and in trial 5, D, J and EE are identical and score 33, 58 and 52). The rubric correlates with the rubric weighted attribute score at 0.73 and 0.82 in these trials, against 0.97 or more in the other four. The rubric enters QR, the rubric shares of every model and the planned regret measure.
3. Coding notes. Screen size is shown in the high information laptop trials but is not coded, since the designers' laptop rubric does not score it. Every HDD takes the lowest storage level whatever its capacity, as in the designers' rubric. Price is coded but constant within every trial and has share zero.

Specification.

4. The median aspect of elimination by aspects uses the midpoint for an even number of options. With the lower middle value instead, the file A1b contrast is +0.094 bits [-0.053, 0.230] and no longer clear (`independent/sensitivity_median.json`). The A1b file result is therefore reported as fragile.
5. Amendment 1 was adopted after the R0 level intervals were seen. It changes only the descriptive level intervals, which become wider, and leaves every model difference and test unchanged up to a common rescaling.
6. The negative mean gap of B1 was known before the plan, so B1 tests only the slope. In the development data set size is confounded with stimulus content (one trial per set size and information level), which the confirmatory dataset addresses.
7. The two primary A1a contrasts are strongly dependent, so Holm's correction is conservative there. A bootstrap p value of zero is reported as p < 0.002.

## Exploratory laptop rubric sensitivity, 9 October 2026

Second run identical. Independent implementation (`independent/sens_laptop_1to4.py`) agrees on every compared value, largest difference 2e-12 (`compare_laptop_rubric_1to4.py`).

## H1 confirmatory on the snack data, 9 October 2026

Run once under Amendment 3. Second run identical. The independent implementation (`independent_snack/`), written from Amendment 3 and `VERIFY_BRIEF_SNACK.md`, agrees on every compared value, with a largest difference of 2e-15 for held-out scores, sensitivities and intervals and 4e-8 for the elasticities, whose profile search stops at a tolerance near 1e-9 (`compare_h1_confirmatory.py`).

Issues the reimplementation raised about Amendment 3.

1. C1 compares choice concentration across screen positions with a value model whose position profile is almost exactly uniform, so it tests position effects only, not narrowing on value.
2. The C2 decision depends on the spacing of the beta grid. The continuous elasticity of C3 peaks near +0.18, between the grid points 0 and +0.5.
3. The looked-at version uses gaze recorded during the choice, so its held-out scores are measurements rather than predictions and are not compared with the all-items version.
4. What people look at depends partly on value, so the C3 difference cannot fully separate narrowing from value-guided search.
5. Bootstrap p values of zero mean p < 0.002.
