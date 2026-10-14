# Search model, confirmatory tests on the snack data (Amendment 5)

## S1, search grows sublinearly

| | 9 | 16 | 25 | 36 | Elasticity |
|---|---|---|---|---|---|
| Distinct items looked at, observed | 7.42 | 10.38 | 12.97 | 14.89 | 0.51 [0.46, 0.55] * |
| Predicted by sample until return | 2.13 | 2.81 | 3.49 | 4.17 | 0.48 [0.46, 0.51] * |
| Predicted by evaluating every item | 9 | 16 | 25 | 36 | 1 |
| Observed over predicted by sample until return | 3.48 | 3.69 | 3.72 | 3.57 | |

- Observed elasticity below one by 0.49 [0.45, 0.54] *.
- Predicted minus observed elasticity -0.02 [-0.07, 0.03], within the margin of 0.1 True. Supported True.

## S2, time held out (mean absolute error of log response time)

| Model | Error | 9 | 16 | 25 | 36 | Sample until return minus model |
|---|---|---|---|---|---|---|
| CL | 0.543 [0.498, 0.586] * | 0.485 | 0.523 | 0.576 | 0.589 | +0.009 [-0.003, 0.021] |
| DbS_avg | 0.550 [0.506, 0.592] * | 0.495 | 0.529 | 0.582 | 0.594 | +0.002 [-0.010, 0.015] |
| MLBA_avg | 0.550 [0.506, 0.592] * | 0.495 | 0.529 | 0.582 | 0.594 | +0.002 [-0.010, 0.015] |
| SR | 0.552 [0.504, 0.596] * | 0.500 | 0.521 | 0.582 | 0.607 |  |

Supported False.

## S3, premonition of slow decisions

| Forecast | Mean within participant rank correlation with response time |
|---|---|
| looks_SR | +0.084 [0.039, 0.129] * |
| entropy_CL | +0.043 [-0.004, 0.092] |
| entropy_DbS_avg | -0.040 [-0.083, 0.002] |
| entropy_MLBA_avg | -0.108 [-0.137, -0.078] * |
| top_rating | -0.322 [-0.364, -0.276] * |
| rating_sd | -0.174 [-0.210, -0.136] * |
| top_count | -0.036 [-0.084, 0.011] |

- Sample until return minus CL entropy +0.041 [0.030, 0.053] *. Supported True.

## S4, premonition of missing the best (bits per trial)

| Model | Log score | Minus constant |
|---|---|---|
| CL | -0.6651 [-0.7447, -0.5799] * | +0.0411 [0.0166, 0.0644] * |
| DbS_avg | -0.7704 [-0.8477, -0.6832] * | -0.0642 [-0.1060, -0.0227] * |
| MLBA_avg | -1.1472 [-1.2958, -1.0012] * | -0.4411 [-0.5842, -0.2936] * |
| SR | -0.6640 [-0.7445, -0.5771] * | +0.0422 [0.0169, 0.0663] * |
| constant | -0.7062 [-0.7799, -0.6235] * |  |

- Sample until return minus CL +0.0011 [-0.0017, 0.0037].

## S5, choices, held out likelihood of the chosen item relative to chance

- Sample until return minus CL -0.3 [-0.7, 0.0] percent (basic -0.7 to 0.0). Non inferior within 2 percent True, better False, worse False. Supported True.

## Secondary

Choices of every model, held out likelihood of the chosen item relative to chance.

| Model | Times chance | 9 | 16 | 25 | 36 | Sample until return minus model, % |
|---|---|---|---|---|---|---|
| CL | 3.84 [3.30, 4.45] | 2.86 | 3.64 | 4.37 | 4.80 | -0.3 [-0.7, 0.0] |
| DbS_avg | 3.55 [3.09, 4.07] | 2.64 | 3.48 | 4.03 | 4.28 | +8.0 [5.1, 11.0] * |
| MLBA_avg | 2.62 [2.35, 2.93] | 2.06 | 2.62 | 2.91 | 3.02 | +46.1 [30.6, 64.2] * |
| SR | 3.83 [3.29, 4.44] | 2.85 | 3.63 | 4.35 | 4.80 |  |

- Elasticity of CL's sensitivity implied by sample until return -0.010 [-0.032, 0.014] (observed in Amendment 3, +0.18).

Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile and basic intervals both exclude zero.
