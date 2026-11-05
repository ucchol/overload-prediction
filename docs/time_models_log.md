# Decision time from choice models, log of rounds (exploratory)

Plan in `docs/time_models_plan.md`. Each round is written here before the next starts. Development scores only, the vault stays sealed until the final list is frozen.

## Simulation checks, 3 November 2026, before any real fit

`scripts/time_models_sim.py` on the displays of snacks in three seconds (values only, 30 people), times and choices simulated from known parameters with a speed factor per person, half the people fitted and the other half predicted.

| Model | Fitted vs true expected log time, R2 | Held out R2, fitted law | Held out R2, true law | Numeric vs Monte Carlo, R2 |
|---|---|---|---|---|
| LBA | 0.983 | 0.292 | 0.289 | 0.986 |
| MLBA LBA | 0.962 | 0.197 | 0.201 | 0.977 |
| Wald race | 0.982 | 0.212 | 0.207 | 0.983 |
| Relative Wald | 0.949 | 0.086 | 0.078 | 0.954 |
| Sample until return timing | 0.924 | 0.097 | 0.085 | 0.927 |

Every process model recovers the true trial by trial expected time and reaches the held out R2 of the true law. The numerical expectation agrees with Monte Carlo up to Monte Carlo noise (100 to 200 draws per trial). Parameters are in relative time units, so they are compared through predictions, not one by one.

## Round 1, 3 November 2026, every model of the plan

Development scores, held out R2 of within person log time over ten folds of the non vault participants.

| Model | Our files | Our laptops | Snacks at a glance | Snacks in three seconds | Sums | Krajbich 2010 | Krajbich 2011 | Folke 2016 |
|---|---|---|---|---|---|---|---|---|
| S1, log2 n | +0.083 | +0.011 | +0.057 | +0.050 | +0.000 | +0.000 | +0.000 | +0.000 |
| S2, structure | +0.090 | +0.016 | +0.162 | +0.123 | -0.003 | +0.056 | +0.072 | +0.036 |
| CL entropy | +0.088 | +0.010 | +0.061 | +0.056 | -0.002 | +0.053 | +0.053 | +0.040 |
| DbS entropy | +0.082 | +0.011 | +0.052 | +0.049 | -0.000 | +0.016 | +0.000 | +0.002 |
| MLBA entropy | +0.079 | +0.011 | +0.035 | +0.041 | +0.000 | -0.001 | +0.050 | -0.004 |
| Sample until return, looks | +0.085 | +0.011 | +0.074 | +0.056 | -0.001 | +0.045 | +0.056 | +0.042 |
| Two look satisficing, looks | +0.085 | +0.015 | +0.168 | +0.128 | -0.002 | +0.033 | +0.071 | +0.018 |
| LBA | -0.087 | -0.378 | +0.010 | -0.667 | -0.084 | +0.003 | +0.033 | -0.018 |
| MLBA LBA | -0.033 | -0.347 | -0.154 | -0.577 | -0.003 | +0.049 | +0.057 | +0.040 |
| Wald race | -0.374 | -0.302 | -0.241 | -0.552 | -0.058 | +0.015 | +0.040 | -0.006 |
| Relative Wald | -0.347 | -0.168 | -0.122 | -0.165 | -0.035 | +0.044 | +0.028 | +0.009 |
| Sample until return timing | +0.088 | -0.030 | +0.073 | +0.003 | -0.081 | +0.022 | +0.037 | +0.028 |
| Ceiling, boosted trees | +0.117 | +0.001 | +0.093 | -0.052 | -0.006 | +0.034 | -0.005 | -0.088 |

What was seen. Two look satisficing's looks alone match or beat the structure law in both snack sets. The race models correlate negatively with time wherever set size varies (within person r from -0.07 to -0.30), because an independent race with more racers finishes sooner, the opposite of what people do. In the two and three option sets their correlation is positive (+0.10 to +0.25). The ceiling underperforms the linear structure law in several sets, a sign that its features were not centered within person and that it overfits.

## Round 2, changes decided from round 1 (written before round 2 ran)

1. Races whose threshold grows with set size, b n^gamma with gamma >= 0 (LBA_n, MLBA_LBA_n, Wald_n, RWald_n). A race needs a set size cost to slow down with more options, as in Hick compatible accumulator accounts. Simulation check on the snack displays (20 people), LBA_n fitted vs true expected log time R2 0.988, held out R2 0.411 against 0.414 for the true law, numeric vs Monte Carlo 0.993. Wald_n, 0.991, 0.491 against 0.491, 0.996.
2. In snacks in three seconds, every race is conditioned on a response before the three second deadline, since trials that ran out are not in the data.
3. Calibrated versions of every race (suffix _cal), one time slope on the race's expected log time fitted on the training people. The slope's sign is reported.
4. Combined laws, structure plus one model quantity (S2+TLS, S2+SR, S2+CLH), to see whether a model adds to structure.
5. Two look satisficing with its sensitivity chosen to fit training times rather than choices (TLS_tfit, grid of 30 values).
6. Ceiling2, boosted trees on the rich features and their within person centered versions, complexity chosen by two inner folds of the training people.

## Round 2 results, 3 November 2026

Development scores, held out R2 of within person log time over ten folds of the non vault participants. Round 1 rows for S1 and S2 are repeated for reference.

| Model | Our files | Our laptops | Snacks at a glance | Snacks in three seconds | Sums | Krajbich 2010 | Krajbich 2011 | Folke 2016 |
|---|---|---|---|---|---|---|---|---|
| S1, log2 n (round 1) | +0.083 | +0.011 | +0.057 | +0.050 | +0.000 | +0.000 | +0.000 | +0.000 |
| S2, structure (round 1) | +0.090 | +0.016 | +0.162 | +0.123 | -0.003 | +0.056 | +0.072 | +0.036 |
| S2 plus two look satisficing looks | +0.095 | +0.004 | +0.175 | +0.126 | -0.002 | +0.058 | +0.071 | +0.035 |
| S2 plus sample until return looks | +0.092 | +0.001 | +0.163 | +0.121 | -0.003 | +0.054 | +0.071 | +0.036 |
| S2 plus CL entropy | +0.093 | -0.000 | +0.162 | +0.122 | -0.005 | +0.057 | +0.072 | +0.035 |
| Two look satisficing, sensitivity fitted to time | +0.091 | +0.014 | +0.172 | +0.131 | -0.002 | +0.042 | +0.072 | +0.021 |
| Ceiling2, boosted trees, centered features, inner tuning | +0.120 | +0.009 | +0.154 | +0.091 | -0.002 | +0.003 | +0.028 | -0.020 |
| LBA, threshold grows with n | +0.026 | +0.002 | +0.057 | +0.112 | -0.082 | +0.003 | +0.033 | -0.018 |
| LBA, threshold grows with n, calibrated | +0.050 | +0.021 | +0.096 | +0.124 | -0.002 | +0.020 | +0.057 | +0.004 |
| Wald race, threshold grows with n | +0.075 | +0.013 | +0.070 | +0.122 | -0.005 | +0.015 | +0.040 | -0.006 |
| Wald race, threshold grows with n, calibrated | +0.075 | +0.013 | +0.070 | +0.121 | +0.001 | +0.020 | +0.057 | +0.004 |
| Relative Wald, threshold grows with n | +0.078 | +0.010 | +0.033 | +0.029 | -0.005 | +0.044 | +0.028 | +0.009 |
| Relative Wald, threshold grows with n, calibrated | +0.078 | +0.010 | +0.040 | +0.035 | -0.000 | +0.058 | +0.029 | +0.003 |
| MLBA LBA, threshold grows with n | +0.007 | -0.026 | -0.036 | +0.040 | -0.020 | +0.049 | +0.057 | +0.040 |
| MLBA LBA, threshold grows with n, calibrated | +0.017 | +0.006 | +0.012 | +0.039 | -0.000 | +0.056 | +0.058 | +0.038 |

Snacks in three seconds, fixed threshold races conditioned on the deadline. LBA +0.019 (calibrated +0.030), Wald race -0.200 (+0.000), relative Wald -0.093 (+0.009), MLBA LBA -0.041 (+0.004).

What was seen.
1. A threshold that grows with set size turns the races from negative to positive wherever n varies. In snacks in three seconds the Wald and LBA races now match structure (+0.122 and +0.124 against +0.123). In snacks at a glance they reach about half of structure. The deadline alone does not repair a fixed threshold race.
2. Two look satisficing remains the strongest process quantity. Added to structure it gives the best score in snacks at a glance, our files and Krajbich 2010. With its sensitivity fitted to time it gives the best score in snacks in three seconds.
3. Calibration slopes are positive and stable across folds for the Wald races. For LBA and MLBA LBA they vary in sign across folds in our files, our laptops and snacks at a glance, so those fits are unstable there.
4. In our laptops every model is near zero, and adding a second feature to structure lowers the score, which points to overfitting in a small set. In sums nothing predicts time, including both ceilings.
5. Ceiling2 is far better than the first ceiling and beats structure in our files only.

## Round 3, changes decided from round 2 (written before round 3 ran)

1. S2 plus two look satisficing looks with the sensitivity chosen to fit training times (S2+TLS_tfit), choosing the sensitivity on the same grid of 30 values by the training R2 of the combined law. Round 2 found the two halves best separately.
2. S2 plus the Wald race with a growing threshold (S2+Wald_n), to ask whether the best race adds anything beyond structure.

After round 3 the list is frozen as every model run in rounds 1 to 3, so the vault judges every model and nothing is selected on development scores.

## Round 3 results, 3 November 2026

| Model | Our files | Our laptops | Snacks at a glance | Snacks in three seconds | Sums | Krajbich 2010 | Krajbich 2011 | Folke 2016 |
|---|---|---|---|---|---|---|---|---|
| S2 plus two look satisficing looks, sensitivity fitted to time | +0.116 | +0.001 | +0.173 | +0.126 | -0.009 | +0.059 | +0.072 | +0.037 |
| S2 plus Wald race with growing threshold | +0.091 | +0.012 | +0.163 | +0.121 | -0.003 | +0.055 | +0.071 | +0.038 |

What was seen. Fitting the satisficing sensitivity to time together with structure lifts our files from +0.090 to +0.116, close to the best ceiling (+0.120), and keeps the gains in snacks. The Wald race adds nothing beyond structure in any set. A development bootstrap of all models (the vault evaluation script run on development predictions, as a test of the script) found that two look satisficing alone, with its sensitivity fitted to choices or to time, already met the satisfactory rule, and that S2 plus satisficing looks fell clearly below S2 in our laptops.

## Frozen list, 3 November 2026

Every model run in rounds 1 to 3, 24 fitted models and the calibrated versions of the 8 races, 32 in all. S1, S2, CL entropy, DbS entropy, MLBA entropy, sample until return looks, two look satisficing looks, LBA, MLBA LBA, Wald race, relative Wald, sample until return timing, Ceiling, S2+TLS, S2+SR, S2+CLH, TLS_tfit, Ceiling2, LBA_n, Wald_n, RWald_n, MLBA_LBA_n, S2+TLS_tfit, S2+Wald_n. In snacks in three seconds every race is conditioned on the deadline, as in round 2. The vault is opened now, twice for an identical second run, and evaluated by `scripts/time_models_vault_eval.py` with the key numbers recomputed by `verification/check_time_vault.py`.

## Vault, opened once on 3 November 2026

Both runs gave identical predictions for all 32 models in all eight sets. `verification/check_time_vault.py` recomputed every R2 and the bootstrap intervals of the key models against S2, largest difference 1e-15. Full results in `results/time_models_vault/` (second run `results/time_models_vault_run2/`, identical), table in `vault_table.md`.

Satisfactory by the rule fixed before any fit. S2, two look satisficing looks (one parameter fitted to choices), two look satisficing with the parameter fitted to time, S2+TLS, S2+SR, S2+Wald_n and the calibrated LBA race with a growing threshold. Not satisfactory. Every fixed threshold race, every entropy law, sample until return looks and timing, S2+CLH, S2+TLS_tfit (below S2 nowhere but not clearly above S1 in Krajbich 2010) and both ceilings.

Main observations.
1. Two look satisficing, one parameter fitted to choices only, predicts time as well as the four feature structure law in snacks at a glance (+0.145 against +0.145, BF01 for no difference 18.7), snacks in three seconds (+0.111 against +0.112, BF01 11.3) and Krajbich 2011 (+0.088 against +0.088, BF01 12.2).
2. Added to structure it is clearly better than structure in snacks at a glance, +0.162 against +0.145, gain +0.017 [+0.010, +0.024], Holm p 0.0002, dz 1.15, 80 percent of vault people better, 98 percent of the best ceiling.
3. A race reaches the rule only with a threshold that grows with set size and a time slope (LBA_n_cal).
4. In sums, our two sets and Folke 2016 no model, including the ceilings, explains time in the vault. Folke 2016 was positive in development and is negative for every model in its small vault.

## Deviation found after the vault was opened, 5 November 2026

A separate check of the write up found that the rising threshold LBA fit for snacks at a glance on the vault never left its starting values (theta equal to the start vector, negative log likelihood 164,548 against 17,219 for the nested fixed threshold LBA), and that the rising threshold MLBA LBA fit for the sums also failed (42,060 against 3,666). A rising threshold race nests its fixed threshold version as gamma goes to zero, so its likelihood can never be worse at a true optimum. Checking every fit this way, all other vault fits converged. In development the rising LBA fit failed in 8 of 10 folds in our files, 5 in our laptops, 8 in snacks at a glance and 1 in the sums, and the rising MLBA LBA fit in 3 folds in snacks at a glance and 5 in the sums (snacks in three seconds could not be checked, since its round 2 fit records were overwritten by the time limit rerun). The Wald and relative Wald races with a rising threshold converged in every fold. The unstable calibration slopes noted in round 2 for LBA and MLBA LBA come from these failures.

Consequences. The vault scores of LBA_n, LBA_n_cal, MLBA_LBA_n and MLBA_LBA_n_cal where the fit failed are not a test of those models, and the satisfactory verdict of LBA_n_cal rests on an unfitted model in snacks at a glance. Both are reported as they are, marked as failed fits, and are not withdrawn from the record.

Correction, decided before refitting. The two rising threshold races are refitted with a warm start, from the converged fixed threshold fit on the same training participants with gamma = 0.02, plus the usual random restarts. A fit counts as converged when its negative log likelihood is no worse than the fixed threshold fit's (within 1). Nothing else changes. The vault is rescored for these four entries only, twice, and reported beside the original results as a correction made after the vault was opened. The secondary measures for the rising LBA race are recomputed the same way. Development is not rerun.

Refinement of the correction, 5 November 2026, after the warm start refits ran (both runs identical). Every warm start fit met the convergence criterion. In two entries where the original fit had already converged, the warm start reached a slightly worse optimum (rising LBA in snacks in three seconds, 3,543 against 3,540, and rising MLBA LBA in snacks at a glance, 17,200 against 16,781), so the likelihood surface has more than one optimum. Each entry therefore uses whichever of the original and the warm start fit has the lower negative log likelihood. This replaces exactly the two failed fits (rising LBA in snacks at a glance, 17,053 against 164,548, and rising MLBA LBA in the sums, 3,665 against 42,060) and keeps every other original fit. The vault R2 of the refits had been printed before this rule was written, but the rule is set by the likelihoods alone. The same rule is applied to the refits behind the unseen set size measure of the rising LBA race, which is rerun with both fits recorded. Corrected results go to `results/time_models_vault_corrected/`, the frozen results stay in `results/time_models_vault/`.
