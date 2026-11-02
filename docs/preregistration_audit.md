# Preregistration audit, plan freezes against runs

Checked 2 November 2026. The plan was never registered on OSF or AsPredicted. Each freeze is a commit of ANALYSIS_PLAN.md whose SHA-256 digest is recorded in STORY.md, and every result file records the digest of the plan it ran under and the time it was created. This audit checks three things for every confirmatory test. The digest in STORY.md equals the digest of the plan at the freeze commit. The result file names that digest or a later one (the plan is append only, so a later digest contains the earlier amendment unchanged). The result was created after the freeze.

Sources of time. Commit times and the created times inside result files are written on our own machine, so on their own they are claims. GitHub records when each push arrived, which is outside our control. The repository was first pushed on 15 October 2026 at 02:50 UTC, so GitHub's record covers Amendments 7 to 12 and not Amendments 1 to 6.

| Amendment | Test | Freeze commit, UTC | Pushed to GitHub, UTC | Result created, UTC | Digest in result | Order |
|---|---|---|---|---|---|---|
| 1 and 2 | R0 and H1 on development data | 9 Oct 10:40 and 10:42 | 15 Oct 02:50 (first push) | H1 rerun 10 Oct 03:03 under Amendment 4 | d93b162c | freeze first, local times only |
| 3 | H1 confirmatory, snack choices | 9 Oct 12:03 | 15 Oct 02:50 | 9 Oct 12:28 | bd138487 | freeze first, local times only |
| 4 | Laptop definitions corrected | 10 Oct 02:58 | 15 Oct 02:50 | 10 Oct 03:03 | d93b162c | freeze first, local times only |
| 5 | Search model on snacks | 14 Oct 14:19 | 15 Oct 02:50 | 14 Oct 14:42 | 5a073cf3 | freeze first, local times only |
| 6 | Two look satisficing on gaze data | 15 Oct 01:53 | 15 Oct 02:50 | 15 Oct 02:07 | 634cdbb8 | freeze first, local times only |
| 7 | Stopping rule, Caplin, Dean and Martin | 31 Oct 09:40 | 31 Oct 09:40 | 31 Oct 09:50 | 876dbadf | freeze first, confirmed by GitHub |
| 8 | Forecasting overload | 31 Oct 09:40 | 31 Oct 09:40 | 31 Oct 10:09 | 876dbadf | freeze first, confirmed by GitHub |
| 9 | Filename cue | 1 Nov 10:00 | 1 Nov 10:00 | 1 Nov 11:01 | 7ebeb737 | freeze first, confirmed by GitHub |
| 10 | H1 within, lotteries | 2 Nov 02:41 | 2 Nov 02:41 | 2 Nov 03:05 | 04388589 (Amendment 11 appended at 03:00, Amendment 10 text unchanged) | freeze first, confirmed by GitHub |
| 11 | H2, return rule | 2 Nov 03:00 | 2 Nov 03:00 | 2 Nov 04:54 | 04388589 | freeze first, confirmed by GitHub |
| 12 | Decision time from structure | 2 Nov 09:30 | 2 Nov 09:30 | 2 Nov 14:09 | 6edfbf9f | freeze first, confirmed by GitHub |

Every recorded digest in STORY.md matches the plan at its freeze commit (13 of 13).

What the paper can and cannot claim.

- For Amendments 7 to 12, the plan text was on GitHub before the result existed, so "frozen before the run" is supported by a record we do not control.
- For Amendments 1 to 6 the order rests on our own commit and result times. They are consistent everywhere, but a reviewer has only our word for them. These tests should be described as specified in a version controlled plan before analysis, not as preregistered in the registry sense.
- A result's created time marks when the run finished, not when it started. For the long runs (Amendment 12 took several hours) the records show the result was written after the freeze, and the result names the frozen digest, so the run used the frozen plan and code. They cannot show on their own that no earlier, unsaved run took place.
- Data access before freezing. The plan allowed inspecting a dataset's structure (set sizes, counts, variable names) before its amendment. Some outcomes were known before a test, and each case is stated in its amendment (for example the snack growth of misses was known before the equivalence check, and times were used in earlier time tests before Amendment 12). Those analyses are labelled descriptive or exploratory.
- Deviations found after runs are listed in docs/deviations_from_plan.md.

Suggested Methods text. "All confirmatory analyses followed a written plan kept under version control (ANALYSIS_PLAN.md in the public repository). The plan was frozen before the first test and extended only by dated amendments appended before each new test was run. We record the SHA-256 digest of the plan at every freeze, and every result file records the digest of the plan it was run under. For the tests of Amendments 7 to 12 the plan was pushed to GitHub before the results were created, which GitHub's own record confirms. The plan was not registered with a third party registry, and we describe these tests as planned rather than preregistered. Deviations are listed in Supplementary Table X."

Recommendation. Before submission, deposit the current repository on OSF or Zenodo with a DOI, so that the whole history has an independent timestamp, and say plainly that this deposit postdates the tests.
