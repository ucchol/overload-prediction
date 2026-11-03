# A machine learning ceiling for completeness (exploratory)

Held out bits per choice above a uniform guess, ten folds of participants. The ceiling is the better of the neural network and the boosted trees.

| Dataset | CL | Sample until return | Neural network | Boosted trees | CL, percent of ceiling | Sample until return, percent of ceiling |
|---|---|---|---|---|---|---|
| Snacks at a glance | 1.935 [1.728, 2.130] | 1.928 [1.719, 2.125] | 1.955 [1.753, 2.144] | 1.922 [1.724, 2.111] | 99 [98, 100] | 99 [97, 100] |
| Snacks in three seconds | 0.711 [0.598, 0.824] | 0.703 [0.589, 0.815] | 0.732 [0.623, 0.841] | 0.712 [0.607, 0.818] | 97 [95, 99] | 96 [93, 99] |
| Sums worked out | 1.951 [1.682, 2.205] | 1.954 [1.681, 2.210] | 2.153 [1.884, 2.411] | 1.645 [1.112, 2.115] | 91 [88, 93] | 91 [89, 93] |

Brackets give 95% percentile intervals from a participant bootstrap of held out trial scores (B = 1000, seeds 170000 + b, no refitting).
A share of the ceiling is an upper bound on completeness (Fudenberg et al. 2022), because the ceiling's own error is at least the irreducible error.
