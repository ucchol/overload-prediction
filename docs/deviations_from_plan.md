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
