# Forecasting choice overload (Amendment 8)

## cdm

| Measure | Value |
|---|---|
| Growth of misses per doubling of the set, observed | +0.156 [0.126, 0.183] * |
| Growth of misses per doubling of the set, CL | -0.011 [-0.022, 0.002] |
| Growth of misses per doubling of the set, CL_comp | -0.082 [-0.094, -0.067] * |
| Growth of misses per doubling of the set, SR | -0.018 [-0.030, -0.006] * |
| Error of the forecast growth, CL | 0.167 [0.129, 0.202] * |
| Error of the forecast growth, CL_comp | 0.238 [0.196, 0.272] * |
| Error of the forecast growth, SR | 0.174 [0.136, 0.209] * |
| O1, people's growth minus compensation's | +0.238 [0.196, 0.272] * |
| O1, people's growth minus CL's | +0.167 [0.129, 0.202] * |
| Compensation's error minus CL's (descriptive) | +0.071 [0.066, 0.075] * |
| O1 verdict | overload |
| Within set size, area under the curve, CL | 0.554 [0.521, 0.593] * |
| Within set size, area under the curve, CL_comp | 0.553 [0.520, 0.591] * |
| Within set size, area under the curve, SR | 0.554 [0.521, 0.594] * |
| Within set size, area under the curve, near_best | 0.561 [0.536, 0.590] * |
| Within set size, area under the curve, minus_gap | 0.552 [0.524, 0.585] * |
| O2, SR minus the near best count | -0.007 [-0.029, 0.017] |
| SR minus one half (descriptive) | +0.054 [0.021, 0.094] * |
| O2 supported | False |

## reutskaja

| Measure | Value |
|---|---|
| Growth of misses per doubling of the set, observed | +0.099 [0.075, 0.122] * |
| Growth of misses per doubling of the set, CL | +0.111 [0.092, 0.128] * |
| Growth of misses per doubling of the set, CL_comp | +0.057 [0.038, 0.074] * |
| Growth of misses per doubling of the set, SR | +0.098 [0.080, 0.115] * |
| Error of the forecast growth, CL | 0.012 [0.001, 0.039] |
| Error of the forecast growth, CL_comp | 0.042 [0.015, 0.071] * |
| Error of the forecast growth, SR | 0.001 [0.001, 0.032] |
| O1, people's growth minus compensation's | +0.042 [0.015, 0.071] * |
| O1, people's growth minus CL's | -0.012 [-0.039, 0.018] |
| Compensation's error minus CL's (descriptive) | +0.030 [-0.024, 0.055] |
| O1 verdict | overload |
| Within set size, area under the curve, CL | 0.702 [0.674, 0.730] * |
| Within set size, area under the curve, CL_comp | 0.701 [0.672, 0.729] * |
| Within set size, area under the curve, SR | 0.702 [0.675, 0.731] * |
| Within set size, area under the curve, near_best | 0.609 [0.585, 0.632] * |
| Within set size, area under the curve, minus_gap | 0.599 [0.577, 0.619] * |
| O2, SR minus the near best count | +0.093 [0.073, 0.116] * |
| SR minus one half (descriptive) | +0.202 [0.175, 0.231] * |
| O2 supported | True |

## Miss rates by set size (original sample)

- cdm_standard_3, sizes [10, 20, 40]
  - observed: 0.068, 0.220, 0.288
  - CL: 0.257, 0.358, 0.241
  - CL_comp: 0.341, 0.399, 0.202
  - SR: 0.278, 0.381, 0.249
- cdm_standard_7, sizes [10, 20, 40]
  - observed: 0.236, 0.561, 0.654
  - CL: 0.533, 0.638, 0.471
  - CL_comp: 0.619, 0.678, 0.399
  - SR: 0.549, 0.644, 0.471
- cdm_process_3, sizes [10, 20, 40]
  - observed: 0.114, 0.267, 0.379
  - CL: 0.278, 0.376, 0.271
  - CL_comp: 0.362, 0.409, 0.226
  - SR: 0.296, 0.392, 0.278
- cdm_process_7, sizes [10, 20, 40]
  - observed: 0.465, 0.587, 0.809
  - CL: 0.574, 0.684, 0.571
  - CL_comp: 0.652, 0.718, 0.486
  - SR: 0.588, 0.686, 0.566
- reutskaja, sizes [4, 9, 16]
  - observed: 0.277, 0.412, 0.472
  - CL: 0.305, 0.447, 0.524
  - CL_comp: 0.359, 0.452, 0.471
  - SR: 0.326, 0.457, 0.520
  - misses whose best option was never fixated: 0.248, 0.568, 0.636

Brackets give 95% percentile intervals from 1000 participant resamples. A star marks a value whose percentile and basic intervals both exclude zero.
