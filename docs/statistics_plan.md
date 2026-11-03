# Statistics layer, plan (written 3 November 2026, before any of it was computed)

Asked by the study's author on 3 November 2026, who chose four additions for every main result. Mixed effects models, effect sizes, Bayes factors for results that are not supported, and the smallest effect each test could detect. The confirmatory decisions already made (bootstrap intervals, the clear rule, Holm) stay as they are. This layer helps readers interpret them and is labelled as secondary in the paper. Every number is computed twice and the key ones by separate code.

## Results covered

| Result | Unit of analysis | Mixed effects model | Effect size |
|---|---|---|---|
| H1 and H3 growth of misses, each dataset (Amendment 8, snack exploratory) | Trial, miss of the best (0 or 1) | Logistic GLMM, miss on log2 n with a random intercept and a random slope of log2 n per participant | Odds ratio of a miss per doubling, and the model implied change in miss probability per doubling |
| H3 excess growth beyond each model (O1) | Trial | The same GLMM with the model's held out log odds of a miss as an offset. The log2 n coefficient is growth beyond the model's forecast | Odds ratio per doubling beyond the forecast |
| H3 premonition (O2) | Trial within set size | Logistic GLMM of a miss on the model's held out log odds and on the near best count, both standardized within set size, random intercept per participant | Odds ratio per standard deviation of each predictor, and the AUC already reported |
| H1 within, lotteries (Amendment 10) | Choice | No participant identifier, so a logistic regression with menu clustered standard errors, miss on log2 n, cost and their interaction | Odds ratio per doubling at each cost, and the ratio of odds ratios high against low |
| H2 return rule (Amendment 11) | Trial | Linear mixed model of the per trial held out log score difference (return model minus dwell model, and return model minus CL), random intercept per participant | Mean difference in bits per choice and Cohen's d over participants (mean of per person differences over their standard deviation) |
| Decision time from structure (Amendment 12) | Trial | Linear mixed model of log time on the four structure features, random intercept and random slope of log2 n per participant | Standardized coefficients, marginal and conditional R2 (Nakagawa and Schielzeth), percent change in time per doubling |
| Machine learning ceiling | Trial | Linear mixed model of the per trial held out log score difference, CL minus the ceiling, random intercept per participant | Share of the ceiling, already reported |
| Decision time models (docs/time_models_plan.md) | Trial | Linear mixed model of the per trial difference in squared prediction error between two models, random intercept per participant | Held out R2 and its difference |

Mixed models are fitted on all participants of a dataset. Logistic GLMMs use the Laplace approximation (as lme4 with nAGQ = 1), written in `src/overload/glmm.py` and checked on simulated data with known parameters before use. Linear mixed models use statsmodels MixedLM by REML. Wald z or t statistics and p values are reported, with likelihood ratio tests for the key coefficient as a check. Within each family of the original plan, p values carry Holm's correction.

## Bayes factors for results that were not supported

For every confirmatory test that was not supported, and for every T3 and every unsupported T1 and T2 of Amendment 12, a Bayes factor BF01 for no effect against the hypothesized effect, from the normal likelihood of the estimate (point value and bootstrap standard error) and a half normal prior in the predicted direction (Dienes 2014). The prior scale is the effect the plan's own simulation produced under the hypothesis where the amendment reports one (H1w +0.036, H2 return minus dwell +1.07 and +0.38 bits), otherwise the effect seen where the same test was supported in another dataset. Each Bayes factor is also given at half and at double that scale. BF01 above 3 is read as moderate evidence for no effect, above 10 as strong, between one third and 3 as inconclusive. For tests whose estimate lies on the wrong side of zero, the Bayes factor is still reported, since it then says how strongly the data count against the hypothesis.

## Sensitivity

For every confirmatory test, the smallest effect it could detect with 80 percent power at a two sided 0.05 level, 2.80 times the bootstrap standard error, beside the estimate and the hypothesized effect. A test whose detectable effect exceeds the hypothesized effect is said to be underpowered for it.

## Reporting

Tables in `results/statistics/`, one row per result, with the original decision unchanged. Any result whose mixed model disagrees with the bootstrap decision is reported and discussed, not reconciled.
