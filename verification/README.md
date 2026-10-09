# Verification

Every reported result passes two checks before it is shown.

1. Second run. The experiment is rerun from a fresh process, and its summary, per trial values and every bootstrap replicate must be identical to the saved ones.
2. Independent reimplementation. A separate implementation, written from `ANALYSIS_PLAN.md` and `VERIFY_BRIEF.md` without access to the package, recomputes the results from the raw data. `compare_independent.py` compares the two and lists every value outside its tolerance.

The independent implementation is archived in `independent/` as it was written, with its results. To rerun it and the comparison, `sh verification/run_independent.sh PATH_TO_CSV` (about two minutes). The data file is copied only into a temporary folder that is removed afterwards.

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
