# Independent implementation of Amendment 12, decision time from trial structure

Written blind from ANALYSIS_PLAN.md (Amendment 12, Section 4 and Amendment 6 item 4). The package's analysis code, its results and the draft of the amendment were not opened. Only the package's data loaders (`overload.data.load_task` and `_match`, `overload.snack.load_snack`, `overload.cdm.load_cdm`, `overload.gaze.load_gaze`) and `overload.bootstrap.summarize` and `holm` are imported. Everything else is in `time_core.py`.

Run with `sh verification/run_independent_time.sh [B]` (default B = 20). It runs `checks.py`, then `run_time.py`, and writes `independent_results.json`.

## Interpretation choices

Data and inclusion

1. Sets and order. dev_file, dev_laptop, snack, reutskaja, cdm, krajbich2010, krajbich2011, folke2016, the order of item 1, used for the bootstrap draws and the transfer.
2. Development data. A participant is a CSV row. A trial enters when its response matches an option (`_match` on the trimmed lowercased text) and `f"{column}_timer"` is a positive finite number of seconds. Values are `trial.U`. A participant enters a task when at least one of its trials enters, and participants are sorted by row order. The laptop stimuli are the default `laptop_trials.json` (storyboard).
3. Snack. Trials kept by `load_snack`, time `rt / 1000` seconds, kept when positive and finite. Participants are the loader's sorted indices of those with at least one kept trial.
4. Reutskaja. One row per (subject, trialid), the first fixation row (trial fields are constant within a trial). Values `rating1` to `rating{numalt}`, choice `chosendestring - 1`, time `rt / 1000`. Kept when the choice lies in the set, every value is finite and the time is positive. Participants sorted by subject.
5. Caplin, Dean and Martin. Every round kept by `load_cdm`, both experiments pooled into one set as item 1 lists one set. Time is `final_time` (seconds of the final selection). Participants are the loader's sorted indices.
6. Gaze sets. Trials and seconds as returned by `load_gaze`.
7. Participants of a set are those with at least one included trial, sorted. P in the folds and in the draws is their number.

Features and outcome

8. Rescaling to 0 to 1 by (v - low) / (high - low), with scales dev 0 to 100, snack -3 to 3, Reutskaja -5 to 5, CDM 1 to 35, gaze 0 to 10 (the ranges named in item 1). Only the width matters, since every feature is centered within participant.
9. Best value is the largest value. The gap is the largest minus the second largest after sorting (zero under a tie at the top). The mean of the rest is the mean of the n - 1 values left after removing one best option (ties at the top keep the other tied options in the rest).
10. Outcome is the natural log of seconds. Centering subtracts the participant's unweighted mean over its included trials, for the outcome and for each feature, once on the original sample. Bootstrap duplicates keep that centering.
11. The S3 feature is the natural log of sample until return's expected looks, the S4 feature CL's entropy in bits, both at the fold's sensitivities, computed for every trial of the set and then centered within participant. Within fold k both the training and the held out rows use fold k's sensitivities.

Models and fitting

12. Sample until return. q = softmax(lambda v) on the rescaled values. sum_k (k + 1)! e_k(q without i) and sum_k k! e_k(q) are computed as the integrals int_0^inf e^-t t prod_{j != i} (1 + q_j t) dt and int_0^inf e^-t prod_j (1 + q_j t) dt, which are exact on Gauss Laguerre nodes (polynomials of degree at most n + 1, nodes m = max(8, nmax // 2 + 4)), summed in the log domain. Every term is positive, so there is no cancellation. Checked against the recursion for e_k, exact enumeration of draw sequences and simulation.
13. Sensitivities of S3 (SR) and S4 (CL) are fitted separately, each by weighted maximum likelihood of the chosen option on the training participants' trials, weights the draw counts. The search is over u = log lambda in [-7, 9] (on 0 to 1 values), a grid of 33 evenly spaced u, then bounded Brent search (xatol 1e-10) between the neighbors of the best grid value, keeping the grid point if it is better. The plan says bounded scalar search over log lambda. The grid guards against a local optimum of SR's likelihood, which need not be concave. Probabilities of a chosen option are floored at 1e-300 inside the log. The original sample's fitted sensitivities lie inside the bounds in every set.
14. Least squares without intercept, weighted by draw counts, minimum norm solution from `numpy.linalg.lstsq(sqrt(w) X, sqrt(w) y, rcond=None)`. Coefficients of sets with a constant or collinear feature (the best value in the development data, log2 n in the gaze sets, the mean of the rest in Krajbich 2010) are the minimum norm ones and need not match another solver's, while predictions do.

Validation and scoring

15. Folds. `numpy.random.default_rng(20261014).permutation(P)` over the sorted participants of each set separately, and the participant at position j of the permutation has fold j mod 10.
16. Held out R2 of each law pools every trial of the set, each predicted by the fold that holds its participant out: R2 = 1 - sum w (y - yhat)^2 / sum w y^2 with y the centered log time. Within a replicate, w is the draw count, and undrawn participants have weight zero in fitting and scoring.
17. Transfer. For a held out set, S2 is fitted on every trial of the other seven sets (no folds), each set's weights divided by their sum, and applied unchanged to every trial of the held out set, scored with its own weights. S2 coefficients reported "on all trials" are fitted on every trial of the set with unit weights.
18. T1 = R2(S2) - R2(S1), T2 = R2 of the transferred law, T3 = R2(S3) - R2(S2), and also S4 minus S2, all from held out predictions.
19. Bootstrap. Replicate b uses `numpy.random.default_rng(120000 + b)` and draws `integers(0, P, size=P)` for each set in the order of choice 1, one generator per replicate. Folds and centering stay fixed, and every sensitivity and regression is refitted in every fold of every replicate.
20. Tests. `summarize` gives the percentile and basic intervals, p value and clear flag, `holm` adjusts the p values over the eight sets within each of T1, T2, T3. Supported when both interval lower bounds are above zero and the Holm adjusted p is below 0.05. With B = 20 the intervals and p values are coarse, so the decisions in this file are not the planned B = 1000 ones. The per replicate values are written to compare one to one.
21. Not implemented. The descriptive growth of mean log time per doubling (item 7) and the snack equivalence check (item 10).
