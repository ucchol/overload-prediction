# Two look satisficing on the snack data (exploratory)

## E1, time held out (mean absolute error of log response time)

| Model | Error | 9 | 16 | 25 | 36 |
|---|---|---|---|---|---|
| CL | 0.543 [0.498, 0.586] * | 0.485 | 0.523 | 0.576 | 0.589 |
| SR | 0.552 [0.504, 0.596] * | 0.500 | 0.521 | 0.582 | 0.607 |
| TLS | 0.528 [0.484, 0.568] * | 0.460 | 0.497 | 0.562 | 0.593 |

- Two look satisficing minus CL -0.0153 [-0.0336, 0.0024], minus sample until return -0.0245 [-0.0441, -0.0058] *.

## E2, slow decisions (mean within participant rank correlation with response time)

| Forecast | Correlation |
|---|---|
| looks_TLS | +0.278 [0.241, 0.315] * |
| looks_SR | +0.084 [0.039, 0.129] * |
| top_rating | -0.322 [-0.364, -0.276] * |

- Two look satisficing minus sample until return +0.194 [0.156, 0.233] *. Two look satisficing plus best rating -0.044 [-0.072, -0.020] * (zero would mean the model carries all of the best rating's link with time).

## E3, distinct snacks looked at

| | 9 | 16 | 25 | 36 | Elasticity |
|---|---|---|---|---|---|
| People | 7.42 | 10.38 | 12.97 | 14.89 | 0.51 [0.46, 0.55] * |
| Two look satisficing | 3.74 | 5.00 | 6.25 | 7.41 | 0.50 [0.47, 0.51] * |
| Sample until return | 2.13 | 2.81 | 3.49 | 4.17 | |

- Predicted minus observed elasticity -0.01 [-0.06, 0.03]. People over model, 1.99, 2.08, 2.08, 2.01.

## Sensitivity, attention that ignores value (point estimates)

- Time error 0.543, rank correlation with time +0.303, distinct snacks 6.98, 10.66, 14.49, 18.20, elasticity 0.69.

Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile and basic intervals both exclude zero. Exploratory, these data suggested the model.
