# Completeness, positioned against Fudenberg, Kleinberg, Liang and Mullainathan (2022)

Fudenberg, Kleinberg, Liang and Mullainathan (2022, Journal of Political Economy 130(4), doi 10.1086/718371) define the completeness of a model class as

(E_naive − E_model) / (E_naive − E_best)

the share of the reducible error that the model removes. E_naive is the error of a naive rule, E_model that of the best model in the class, and E_best the irreducible error. They estimate E_best with a lookup table, which maps each feature vector to its best prediction under tenfold cross validation, when every feature vector has many observations, and with black box machine learning otherwise. In the second case the ratio is only an upper bound on completeness.

Our measure (ANALYSIS_PLAN.md Section 5) is the same ratio, with three choices fixed.

1. Error is log loss in bits per choice.
2. The naive rule is a uniform guess, so E_naive − E_model is our L_c, the log score above a guess.
3. Every display was seen by many people, so the lookup table is exact. The best prediction for a display is its observed choice shares, and E_naive − E_best is log2 n_c − H(p_c). The plug in entropy of the shares understates the irreducible error, so we subtract the Miller (1955) correction (m_c − 1) / (2 N_c ln 2) analytically instead of cross validating the table.

What this gives the paper.

- The term and its reading (0 is the naive rule, 100 removes all but the irreducible error) come from Fudenberg and colleagues and are cited as such. Our contribution is not the measure. It is applying it to process models of choice under growing set sizes.
- Their finding that completeness depends on which experiments are run matches ours. Laptop and file displays give very different completeness for the same rule.
- Where each display is seen once (the open datasets), no lookup table exists. Their fallback, a flexible machine learning benchmark, is the route to an upper bound there. We do not use it in the main paper, and we report bits per choice instead.

Text for the paper (Methods). "We measure how much of the predictable variation in choices each model captures with completeness (Fudenberg et al. 2022), the share of the reducible log loss between a uniform guess and the irreducible error that a model removes. Because every display was seen by many participants, the irreducible error is the entropy of the observed choice shares, which we correct for sample size (Miller 1955)."
