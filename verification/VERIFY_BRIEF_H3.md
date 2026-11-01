# Independent verification brief, test A3b of H3 (Amendment 9)

Written blind from `ANALYSIS_PLAN.md` (Sections 3 to 5, Amendment 1, Amendment 9), `VERIFY_BRIEF.md` and the task statement, without opening the package under test or its results. Code in `independent_h3/h3.py`, rerun with `sh verification/run_independent_h3.sh PATH_TO_CSV [B]` (default B = 50, about two seconds).

## What was reused

- `verification/independent/core.py`, the earlier blind implementation, for choice matching and the panel, attribute coding, rubric shares, the CL, DbS_avg and MLBA_avg scores, the one parameter softmax fit, log scores, L*, completeness and the interval summaries.
- `core.py` imports the data definition `coding_definition.py`. As in `run_independent.sh`, the runner copies `src/overload/coding.py` under that name into a temporary folder. It is the only file of `src/overload/` used apart from the three stimulus files (`file_trials.json`, `rubric.json`, `filename_glove.json`). It was not opened.
- Check against the earlier verification. The file rubric shares, every trial's L*, the pooled CL sensitivity (1.344582) and the held out completeness of CL, DbS_avg and MLBA_avg (29.532, 32.084, 31.359) equal the values in `independent/results_storyboard.json`.

## Details the amendment left open and how they were fixed

1. Extension. Removed as the text after the last dot. A name without a dot keeps its whole text (`patcher`, `JAPPA`, `zephyr_run`, `process_1`). In `jappa_system_analysis_v1.1.pptx` only `pptx` is removed and the words are jappa, system, analysis, v.
2. Splitting. Digits and every other non letter separate words. Inside a run of letters, a word starts at each lower to upper boundary and at the last capital of a run of capitals followed by a lowercase letter (regular expression `[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+`). Words are lower cased. Single letter words are kept when they have a vector (`t` in `t1.enc` and `data_t1.zip`, `v` in the four `jappa_system_analysis` files). No name repeats a word, so token and type means coincide.
3. Coverage. The names and the anchor give 71 distinct words and 67 have vectors. The missing four are cplan, datadump, jappa and patcher, as the amendment states.
4. Mean and cosine. The mean of the raw 300 dimensional vectors (not normalized first) of a name's known words, the same for the anchor's six words, and their cosine, in float64.
5. Names with no known word. `patcher` (T1_L8), `cplan.pdf` (T2_H8) and `JAPPA` (T5_L32). Each takes the mean cue of the other names of its trial that have a known word. No trial has two such names, so whether the mean includes other imputed names never arises.
6. Scale of the cue. The raw cosine enters the model, with no standardization. A common rescaling would change only the reported gamma, while a within trial standardization would change the model, so neither was applied. Gamma is per unit of cosine.
7. Rule scores. CL, DbS_avg and MLBA_avg at beta = 0, with the rubric shares of all six file trials, fixed across replicates. The rules without the cue use the one parameter fit of Section 4.
8. Fit with the cue. Maximum likelihood of softmax(lambda s + gamma z) on the pooled choices of the training trials. Damped Newton on the concave log likelihood with analytic gradient and Hessian (checked by finite differences, relative error below 1e-9), stopped when a step moves both parameters by less than 1e-13 relative. If the unconstrained optimum had lambda < 0, the fit would move to lambda = 0 with gamma alone and check the Kuhn Tucker condition. This never happened (0 of 1071 fits, smallest lambda 0.72). In the original sample every one of the 21 fits was also run by L-BFGS-B from four starts with lambda >= 0, and the Newton optimum was never worse (largest gap in negative log likelihood 2e-13). A grid around each full sample optimum found no better point. The amendment names L-BFGS-B from four starts, and this implementation reaches the same optimum by another route.
9. Folds. Each trial is predicted by the fit to the other five, for the rules with and without the cue. Gamma is the fit to all six trials of the same sample.
10. Subsets. Low and high use the same held out predictions (folds over all six trials) and sum L and L* over the three trials of that information level. The spread of a subset is computed within that subset.
11. A3b. For each rule, gain minus the spread of the three rules without the cue, so the three A3b values share one spread.
12. Benchmark. L* of each trial is the full sample value in the original sample and in every replicate (Amendment 1).
13. Resampling. Replicate b uses `numpy.random.default_rng(10000 + b)` and draws the file panel indices first with `integers(0, P_file, size=P_file)`, as in `independent/analysis.py`. The laptop draws come after and cannot change the file indices, so they are not made here (the laptop stimuli are outside the inputs of this check). The panel is every respondent with at least one matched file choice, in file order. A respondent drawn k times counts k times.
14. Replicates. Only b = 0 to 49 were run. With 50 replicates the percentile and basic intervals are coarse and a bootstrap p value moves in steps of 0.04. `summary_B50.json` holds them for information. The decision rule of Amendment 9 is meant for B = 1000.

## Outputs

- `independent_h3/independent_results.json`, `{"point": {...}, "replicates": {"0": {...}, ...}}`, keys as `all|K|CL`, `all|K|CL+cue`, `all|gain|CL`, `all|A3b|CL`, `all|spread`, the same for `low` and `high`, and `gamma|CL`, `gamma|DbS_avg`, `gamma|MLBA_avg`. Completeness in percent.
- `independent_h3/cue.json`, every trial's cue values by file name.
- `independent_h3/summary_B50.json`, point, percentile and basic intervals, p and the clear flag of every measure over the 50 replicates.

No identifier, participant count or choice count is printed or written.
