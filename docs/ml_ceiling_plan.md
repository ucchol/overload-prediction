# A machine learning ceiling for completeness on the open datasets (exploratory, fixed before the run)

Written 2 November 2026, before any of the code below was run on real data. Exploratory, so it is not an amendment and carries no decision rule.

Why. In our own study every display was seen by many people, so the irreducible error is the entropy of the choice shares and completeness is exact. In the open datasets each display is seen once and no lookup table exists. Fudenberg, Kleinberg, Liang and Mullainathan (2022) then use a flexible machine learning model as the benchmark. Its error is at least the irreducible error, so the completeness it gives is an upper bound.

Datasets. Snacks at a glance, snacks in three seconds, and the four panels of Caplin, Dean and Martin, loaded as in Amendment 8.

Validation. Ten folds of participants, numpy.random.default_rng(20261014) as in Amendment 6, within each panel. Every model is fitted on nine folds and scored on the tenth. The score is the mean log2 probability of the chosen option minus that of a uniform guess, in bits per choice.

Features of each option, all computed within its trial. Value scaled by the dataset's value range, value minus the best, value minus the mean, rank from the best scaled to 0 to 1, whether it is the best, the value gap to the best other option, its position in the data's order scaled to 0 to 1, the share of options within one unit of it, and log2 of the set size.

Benchmarks.
1. A neural network score (one hidden layer of 16 tanh units) passed through a softmax over the trial's options. Its L2 penalty is chosen from 0.001, 0.01, 0.1 and 1 by five inner folds of the training participants, three random starts, seeds 160000 plus fold.
2. Gradient boosted trees (scikit learn HistGradientBoostingClassifier, default settings, random_state 0) predicting whether each option was chosen, renormalized within the trial.
The ceiling of a panel is the better of the two in held out bits.

Compared. CL and sample until return, each with its one sensitivity fitted on the same training folds.

Reported. Bits per choice for every model, and the share of the ceiling that CL and sample until return reach (their bits divided by the ceiling's), an upper bound on their completeness. Intervals come from a participant bootstrap of the held out trial scores (no refitting), B = 1000, numpy.random.default_rng(170000 + b). The ceiling is chosen once on the full data and kept fixed in every replicate.

Checks. Run twice for identity, and the CL and ceiling scores recomputed by separate code.
