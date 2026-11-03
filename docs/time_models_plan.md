# Decision time from choice models, plan (exploratory, written 3 November 2026 before any fit)

Asked by the study's author on 3 November 2026. Each model should predict how long a person takes on a particular trial, the models should be compared, and the work may be repeated until the fit is satisfactory. Repeating until a fit looks good can fit noise, so the work is split into a development part, where any number of rounds is allowed, and a sealed vault, opened once.

## Data

The eight sets of Amendment 12 with the same loaders and value scales (`overload.experiments.time_structure.load_all`), our file and laptop tasks, snacks at a glance, snacks in three seconds, the sums of Caplin, Dean and Martin, Krajbich 2010, Krajbich 2011 and Folke 2016. Values are rescaled to 0 to 1 by each set's scale.

## Vault

In every set, participants are ordered by numpy.random.default_rng(20261103).permutation of their indices, and the first 30 percent (rounded up) form the vault. No vault trial is used to fit, select or tune anything until the final run. The vault is opened once, for every model in the frozen list, whatever the results.

## Outcome and score

Log time in seconds, centered within participant, as in Amendment 12, so a score says which trials take a given person longer, not who is faster. A model's prediction for a trial is also centered within that participant over the same trials. The score is R2 = 1 - sum (y - yhat)^2 / sum y^2 on participants the model was not fitted to. Zero is a constant time for each person.

Development scores use ten folds of the non vault participants (numpy.random.default_rng(20261014).permutation, fold by position modulo 10). Vault scores use models fitted on all non vault participants.

Secondary. (a) The within person rank correlation of predicted and observed time. (b) For sets with three or more set sizes, forecasting an unseen set size. Each set size is left out in turn, models are fitted on the others, and the score is the error in each held out person's difference between mean log time at the left out size and at the other sizes. (c) Choice prediction of the timing models, bits per choice, to see what timing costs or gains in choices.

## Models

Baselines. S1, log2 n. S2, the four structure features of Amendment 12 (log2 n, best value, gap to the second best, mean of the rest). Both by least squares on centered data.

Laws derived from choice models, each with its choice sensitivity fitted on the training choices and one time slope fitted by least squares on centered data. CL entropy (Hick and Hyman), DbS entropy (rank based scores), MLBA entropy (averaged comparisons), sample until return log expected looks, two look satisficing log expected looks.

Process models that predict a full time distribution, fitted jointly to choices and times on the training participants by maximum likelihood, with each person's times divided by that person's geometric mean time (so person speed does not enter the fit). The trial prediction is the model's expected log time, computed by numerical integration.

- LBA, linear ballistic accumulators with drift v_i = d0 + d1 x_i, drift standard deviation 1, start point uniform on [0, A], threshold A + k, nondecision time t0.
- MLBA LBA, as LBA with MLBA style drift for one attribute, d0 + g sum_j w_ij (x_i - x_j) / (n - 1), w = exp(-l |x_i - x_j|).
- Wald race, racing diffusions (inverse Gaussian first passage) with drift d0 + d1 x_i (kept positive), threshold b, noise 1, t0.
- Relative Wald, as the Wald race with drift v / (1 + exp(-tau (x_i - max of the others))), the GLAM form without gaze.
- Sample until return timing, time = t0 + the sum of L look durations, each gamma with shape kappa and scale theta, with L the number of looks of sample until return at the sensitivity fitted to choices. P(L > k) = k! e_k(q).

Ceiling. Gradient boosted trees (scikit learn HistGradientBoostingRegressor, random_state 0) on a rich set of trial features (log2 n, the five largest values, gap, mean and spread of the rest, count of options within 0.1 of the best, CL entropy at three fixed sensitivities), trained on centered log time. A model's share of the ceiling's R2 is reported as completeness for time, an upper bound in the sense of Fudenberg and colleagues.

## Satisfactory

Fixed now, before any fit. A model is satisfactory when, on the vault, (1) its R2 is clearly above S1 in every set where S2 is clearly above S1, and (2) it is not clearly below S2 in any set. Clear means the 95 percent percentile and basic intervals from a participant bootstrap of the vault (B = 1000, numpy.random.default_rng(130500 + b), no refitting) both exclude zero. The share of the ceiling is reported beside it.

## Rounds

Round 1 runs every model above. Later rounds may change or add models only after looking at development scores, and each round, with what changed and why and every development score, is written to `docs/time_models_log.md` before the next round starts. Simulation recovery checks precede real fits for each process model. The final list of models is frozen in the log, and only then is the vault opened. The run is repeated for an identical second run and key numbers are recomputed by separate code.

## Reporting

Exploratory throughout, labelled as such, every model and every set reported.
