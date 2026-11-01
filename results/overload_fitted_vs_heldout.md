# Fitted and held out forecasts of human overload, every model (exploratory)

Growth is the rise in the share of choices that miss the best option per doubling of the set size. Held out forecasts each set size from fits to the others, fitted uses one fit to every set size. Error is the distance from people's growth. Bits per choice is the mean log2 probability of the chosen option (higher is better). Brackets give 95% percentile intervals from the participant bootstrap, a star marks a value whose percentile and basic intervals both exclude zero.

## Caplin, Dean and Martin

People's growth +0.156 [+0.126, +0.183] *.

| Model | Growth, held out | Growth, fitted | Error, held out | Error, fitted | Bits, held out | Bits, fitted |
|---|---|---|---|---|---|---|
| CL | -0.011 [-0.022, +0.002] | +0.036 [+0.033, +0.039] * | 0.167 | 0.120 | -2.588 | -2.515 |
| CL with compensation | -0.082 [-0.094, -0.067] * | -0.015 [-0.018, -0.009] * | 0.238 | 0.170 | -2.720 | -2.561 |
| Sample until return | -0.018 [-0.030, -0.006] * | +0.031 [+0.028, +0.034] * | 0.174 | 0.125 | -2.595 | -2.509 |
| Mixed logit | -0.004 [-0.017, +0.008] | +0.037 [+0.034, +0.039] * | 0.159 | 0.119 | -2.503 | -2.464 |
| DbS | +0.234 [+0.217, +0.244] * | +0.220 [+0.206, +0.226] * | 0.080 | 0.065 | -2.526 | -2.503 |
| MLBA | +0.215 [+0.201, +0.229] * | +0.225 [+0.209, +0.239] * | 0.061 | 0.069 | -3.278 | -3.255 |

O1 verdict held out: overload. Fitted: overload.

## Reutskaja and colleagues

People's growth +0.099 [+0.075, +0.122] *.

| Model | Growth, held out | Growth, fitted | Error, held out | Error, fitted | Bits, held out | Bits, fitted |
|---|---|---|---|---|---|---|
| CL | +0.111 [+0.092, +0.128] * | +0.113 [+0.097, +0.127] * | 0.012 | 0.014 | -2.338 | -2.338 |
| CL with compensation | +0.057 [+0.038, +0.074] * | +0.076 [+0.061, +0.090] * | 0.042 | 0.023 | -2.352 | -2.344 |
| Sample until return | +0.098 [+0.080, +0.115] * | +0.105 [+0.089, +0.119] * | 0.001 | 0.006 | -2.346 | -2.345 |
| Mixed logit | +0.101 [+0.083, +0.119] * | +0.106 [+0.090, +0.120] * | 0.002 | 0.007 | -2.323 | -2.323 |
| DbS | +0.208 [+0.184, +0.231] * | +0.183 [+0.165, +0.200] * | 0.109 | 0.084 | -2.401 | -2.387 |
| MLBA | +0.209 [+0.187, +0.231] * | +0.180 [+0.161, +0.199] * | 0.111 | 0.081 | -2.565 | -2.541 |

O1 verdict held out: overload. Fitted: overload.

## Snacks

People's growth +0.010 [-0.003, +0.026].

| Model | Growth, held out | Growth, fitted | Error, held out | Error, fitted | Bits, held out | Bits, fitted |
|---|---|---|---|---|---|---|
| CL | +0.062 [+0.046, +0.078] * | +0.048 [+0.035, +0.061] * | 0.051 | 0.038 | -2.284 | -2.279 |
| CL with compensation | +0.004 [-0.009, +0.018] | +0.004 [-0.006, +0.016] | 0.006 | 0.006 | -2.272 | -2.272 |
| Sample until return | +0.051 [+0.036, +0.068] * | +0.040 [+0.027, +0.053] * | 0.041 | 0.030 | -2.289 | -2.285 |
| Mixed logit | +0.056 [+0.044, +0.070] * | +0.044 [+0.034, +0.056] * | 0.046 | 0.034 | -2.270 | -2.266 |
| DbS | +0.134 [+0.113, +0.153] * | +0.109 [+0.091, +0.126] * | 0.124 | 0.099 | -2.398 | -2.378 |
| MLBA | +0.133 [+0.112, +0.154] * | +0.113 [+0.091, +0.133] * | 0.123 | 0.102 | -2.832 | -2.811 |

O1 verdict held out: compensation. Fitted: compensation.

## Development data

See `overload_dev/table_overload_dev.md` for every model's fitted and held out forecasts of misses, regret and felt overload by display, in file selection and laptop purchase.

