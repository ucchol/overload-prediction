# Draft, Amendment 10. H1 within one study, on the data of Aguiar, Boccardi, Kashaev and Kim (2023)

Status. Draft for the study's author to review. Nothing below has been run on any recorded choice.

What has been seen. The replication package `QE1861_code_and_data.zip` (SHA-256 4081c63fac9b8510f64379ca003a30f9a5ac5558c0b2e5c75546e23fedba31ee), its two README files, the file names, the column names and row counts of the four data files (`x1`, `x2`, about 4100 rows in each cost file, 12,298 pooled), and the authors' Julia code that reads them (`x1` indexes the 32 subsets of the five lotteries in the order of `powerset(1:5)`, the first being empty, and `x2` is the chosen option, 0 for the sure amount). No choice, frequency or table of the package has been opened, and the results sections of the paper and appendix have not been read.

Design of the data. Each menu is a subset of five lotteries over prizes (50, 48, 30, 14, 12, 10, 0 tokens), and the sure 12 tokens (option 0) is always available. Prizes are shown as sums or differences that must be worked out, with 1, 3 or 5 operations per prize (low, medium and high cost). Expected values are, option 0, 12. Lottery 1, 25. Lottery 2, 20. Lottery 3, 22.5. Lottery 4, 24.2. Lottery 5, 21.7. The package holds no participant identifier.

Claim. The cost of judging an option decides whether more options hurt. When judging is cheap people compensate, and the growth of misses with set size is small. When judging is costly misses grow faster.

1. Outcome. A miss is a choice whose expected value is below the largest expected value on its menu (sure amount included).
2. Set size. n, the number of options on the menu, from 2 to 6 (lotteries plus the sure amount). Menus with no lottery are excluded.
3. Growth. Within each cost level, the least squares slope of the miss rate per menu size on log2 n, each size weighted by its number of choices.
4. Primary test, H1w. Growth at high cost minus growth at low cost. Supported when clearly positive. The lotteries, prizes and menus are the same at every cost level, so risk attitudes, which push choices away from lottery 1 (the best by expected value and also the riskiest), enter both growths alike and cancel in the difference.
5. Secondary, linked to Amendment 8. Within each cost level, CL and sample until return with expected values, each set size forecast from fits to the other sizes. People's growth minus CL's, compensation when clearly negative and overload when clearly positive. Predicted, compensation or no difference at low cost and overload at high cost.
6. Descriptive. Miss rates by menu size and cost with intervals, the medium cost growth, the growth with the miss defined against a CRRA expected utility fitted by maximum likelihood on all choices of the low cost file (a check that the risk confound does not drive the result), and each model's bits per choice.
7. Resampling. The package has no participant identifier, so choices are resampled with replacement within each menu and cost level (B = 1000, numpy.random.default_rng(100000 + b)). Each person contributed at most about two choices, so this understates uncertainty only slightly. This is stated as a limit.
8. Checks before freezing. The code is run on choices simulated over these menus from CL with a fixed sensitivity at every cost level (H1w should not be supported) and from CL whose sensitivity falls with cost (H1w should be supported).

Open points for the study's author.
- License. The package states none beyond the Econometric Society's terms of use. It is free to download. Confirm that this counts as open for the paper.
- The best option by expected value is also the riskiest. The primary test cancels this, but readers may still ask. The CRRA check addresses it descriptively.
- Set size varies mostly between people, and there are no participant identifiers. Both are limits of this dataset.
