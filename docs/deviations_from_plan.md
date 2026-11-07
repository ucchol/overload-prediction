# Changes to the analysis plan and deviations found after runs

The plan (`ANALYSIS_PLAN.md`) was frozen on 9 October 2026 and has only been extended by dated amendments. Earlier text is never edited. Each freeze is fixed by the SHA-256 digest of the whole file, recorded in `STORY.md`, and every commit is public in the repository history.

## Amendments

| Amendment | Date | Kind | What it added or changed | Data seen before it |
|---|---|---|---|---|
| 1 | 9 Oct | Method fix | Completeness in bootstrap replicates holds the attainable score at its full sample value | Development baseline R0 only |
| 2 | 9 Oct | Method fix | Interval rule for the narrowing gap B1 | R0, before any H1 test |
| 3 | 9 Oct | New test | Confirmatory narrowing test on the snack data | Structure of the snack files only |
| 4 | 10 Oct | Data correction | Laptop trials 5 and 6 follow the study's storyboard, no test changed | Development results with the old definitions |
| 5 | 14 Oct | New tests | Search model on the snack data (S1 to S5) | Snack choices (Amendment 3), set size means of times and gaze |
| 6 | 14 Oct | New tests | Two look satisficing on three gaze datasets (G1 to G3) | Column names, value ranges |
| 7 | 31 Oct | New tests | Stopping rule on the sums data (C1 to C3) | Package documentation and design variables |
| 8 | 31 Oct | New tests | Forecasting overload on the sums and three second data (O1, O2) | Same as 7 |
| 9 | 1 Nov | New test | Independent filename cue (A3b), old H4 set aside | Development choices throughout |
| 10 | 1 Nov | New test | H1 within one study, lottery data at three costs | Package structure, no choices |
| 11 | 1 Nov | New test | H2 return rule on gaze sequences | Gaze file headers, set size summaries |
| 12 | 2 Nov | New tests | Decision time from trial structure on eight datasets (T1 to T3), equivalence for snack growth (descriptive) | Times in every dataset, through the earlier time tests |

Every confirmatory amendment was run first on simulated data over the real designs, and the result of that check is written into the amendment.

## Deviations and corrections found after runs

| Where | What | Effect |
|---|---|---|
| Amendment 4, item 2 | Evidence figures were in sample fits, not held out. Corrected on 14 Oct, held out values still favour the storyboard | None on any decision |
| Development time laws | The per step time argument (7 to 12 ms per comparison) was a ratio, not a fit. Withdrawn on 14 Oct | Time no longer ranks the models |
| Amendment 9 | A sentence encoder was planned as the frozen text embedding, but its host was blocked. GloVe vectors were used and named in the amendment before the run | None, disclosed |
| Snack second runs | Summaries differ from the first run only in the plan digest, because Amendment 9 was added between runs | None |
| Amendment 10, item 7 | Stated that each person made few choices. Each person faced the same menus at all three costs, so choices are linked across costs. Ignoring the link most likely widens the intervals of differences between costs | Clear results stand |
| Amendment 11, Reutskaja last cue | The last fixated item is always the chosen one, so that fit has no finite optimum | Descriptive only, not interpreted |
| Amendment 12, item 3 | With two options the mean of the rest equals best minus gap. Least squares handles the redundancy, so predictions are unchanged | None |
| ML ceiling plan, item 1 | Seeds were planned as 160000 plus fold. The script used 160000 plus 100 times the panel's place plus fold, so each panel has its own seeds | None, fixed in the code before the run |
| ML ceiling plan, bootstrap | The first run crashed after every fit because one participant pool was drawn for both Caplin, Dean and Martin treatments, which hold different people. Participants are now drawn within each source group, as in the forecasting bootstrap, and both runs were started again. The new runs saved held out trial scores identical to those of the crashed runs | None on points, intervals follow the planned design |
| Amendment 13, item 3, end of the list | The text says that when no option reaches the threshold the best option is chosen with ties uniform. The code keeps the first best found in the order of inspection (recall). Under random order the two are the same in expectation (the separate check found FT differing by less than 0.0004), so R1 and FT are unaffected. Under screen order the code always keeps the top-most best option. Found by the blind check on 7 Nov, before any bootstrap result was seen | Secondary screen order rows only, both readings reported |
| Amendment 13, item 3, fitting | The first package run's SQUAREM step clipped negative components to exactly zero, and EM can never revive a zero component, so some fits stopped short of the maximum (largest shortfall 0.04 in log likelihood, forecast growth off by up to 0.0006). Found by comparing with the blind check, which reached higher likelihoods. Fixed on 7 Nov by halving the step toward plain EM until every component stays positive, after which every SAT fit matches the blind check's log likelihood and forecasts to six decimals. Both bootstrap runs were started again with the fixed code, and the first run's slices are kept outside the repository | None on any verdict of the first run, final numbers from the fixed runs |
| Amendment 13, FT Monte Carlo ties | The package compares the running best with t - d (k - 1) in exact floating point, the blind check with a tolerance of 1e-9, which moves a few Monte Carlo stops. FT growths agree within 0.0004, and the d chosen in every fold agrees | None |
| Amendment 13, item 4, benchmark time scale | The text sets the mean inspection time so that the benchmark's mean time of the final selection equals people's. The frozen runs divided people's mean final time by the options inspected when search stopped, which counts the options inspected after the final selection whenever the list ran out. Found by the blind check (benchmark time coefficient -0.004 against +0.005 in Experiment 2). The code now follows the text (option --bench-scale final, the default, while --bench-scale stop reproduces the stored runs), and S1 was recomputed from the stored replicates of both runs, identical, by scripts/rational_benchmark_s_text.py (results/rational_benchmark/S1_text_benchmark.json). The raw coefficients do not depend on the benchmark | S1 time difference +0.115 [+0.091, +0.142] by the text against +0.105 [+0.082, +0.132] as run, clear either way, the text version is reported |
