# Blind check: mixed logit and fitted forecasts of choice overload

Write your own code from this description only. Do not read `src/overload/mixed_logit.py`,
`overload_forecast.py`, `overload_premonition.py` or `overload_dev.py`, and do not open anything under `results/`.
You may reuse your own earlier code in `independent_overload/` and `independent_cdm/` (the loaders, CL, SR and the
miss forecasts you already wrote for the overload check), since those were checked against the main code.

Put everything in `independent_mxl/` with a runner `run_independent_mxl.sh` that writes `independent_mxl/out.json`.
Original sample only (equal weight for every participant), no resampling.

## Model: mixed logit (MXL)

People choose by CL, P(option i) = softmax(lambda * s_i) over the same scores s as CL (the option values for the
open datasets, the CL rubric scores for the development data). Sensitivity varies across people:
log lambda ~ Normal(mu, sigma). One lambda per person, shared across all that person's trials in the fit.

Fit (mu, sigma) by maximum likelihood of each person's choices taken together:
sum over people of log integral prod_t P_t(chosen | lambda) dN(log lambda; mu, sigma).
Compute the integral however you like, but make it accurate: some people have about 180 trials, so their
likelihood in log lambda is very narrow (a 20 node Gauss-Hermite rule is NOT accurate enough). Bounds:
sigma in [0.05, 1.5]. Report the maximum you find, and say how you checked it is the global one.

Prediction for any trial is the population average: integral of softmax(lambda s) dN(log lambda; mu, sigma).

## What to compute

Open datasets, use the panels you built before: CDM (cdm_standard_3, cdm_standard_7, cdm_process_3,
cdm_process_7), Reutskaja, snack.

1. Held out, as before: for every set size s, fit on the trials of other sizes, predict the trials of size s.
   Fitted: one fit on all trials, predict all trials.
2. For each panel, each mode and each model in CL, CL_comp, SR, DbS, MLBA, MXL:
   - (mu, sigma) for MXL per fold,
   - predicted miss rate by set size (probability of not choosing an option tied for the top value), and its slope on
     log2 of the set size ("growth"),
   - choice accuracy: mean over trials of log2 of the predicted probability of the chosen option (bits, not above
     uniform).
   For CL_comp, DbS, MLBA use the definitions from the earlier overload brief (`VERIFY_BRIEF_OVERLOAD.md`). The fitted
   mode is new for every model: one sensitivity on all trials.

Development data (`/mnt/user-data/uploads/Updated_Choice_Overload.csv`, two tasks, file and laptop, six displays
each): MXL only, on the CL rubric scores of each display. Fitted on all six displays, and held out for each display
(fit to the other five). Per task report (mu, sigma) per fit, the predicted miss probability per display (one minus
probability of the options with the top rubric score), and bits per choice above uniform per display,
log2(n q_chosen) averaged over that display's choices. Use your own loader. If you need the rubric score
definition, read `src/overload/data.py` and `src/overload/models.py` for the CL score only (those predate this work).

## Report

`out.json` with these numbers, and a short note of anything ambiguous and what you decided. Never output participant
identifiers, ProlificIDs or IP addresses, and never print the total participant count.
