# H1, narrowing with set size (development data)

## file

### A1a, set size scaling, held out completeness (%)

| Model | beta | Completeness | 8 | 16 | 32 | Overlap % | Minus beta 0 | Overlap minus beta 0 |
|---|---|---|---|---|---|---|---|---|
| CL | -1.0 | 28.4 [20.0, 37.0] * | 61.9 | 3.0 | 21.1 | 57.6 | -1.1 [-5.5, 2.8] | -0.6 [-2.3, 1.6] |
| CL | -0.5 | 29.9 [21.1, 39.1] * | 61.7 | -5.8 | 32.5 | 58.8 | 0.4 [-2.0, 2.5] | 0.5 [-1.0, 1.6] |
| CL | +0.0 | 29.5 [20.2, 39.2] * | 54.0 | -7.7 | 39.3 | 58.2 |  |  |
| CL | +0.5 | 26.6 [17.8, 35.5] * | 38.0 | 0.3 | 38.4 | 56.4 | -3.0 [-5.4, -0.8] * | -1.8 [-2.4, -0.4] * |
| CL | +1.0 | 22.2 [14.3, 30.4] * | 21.9 | 6.8 | 35.1 | 54.6 | -7.3 [-12.1, -3.2] * | -3.6 [-4.5, -1.5] * |
| DbS_avg | -1.0 | 27.8 [19.3, 37.0] * | 53.3 | 12.9 | 18.5 | 56.8 | -4.3 [-9.3, 0.3] | -2.0 [-3.9, -0.2] * |
| DbS_avg | -0.5 | 31.3 [22.1, 41.0] * | 55.9 | 8.3 | 29.4 | 58.6 | -0.8 [-3.5, 1.5] | -0.1 [-1.7, 0.7] |
| DbS_avg | +0.0 | 32.1 [22.4, 42.2] * | 52.4 | 5.8 | 36.5 | 58.7 |  |  |
| DbS_avg | +0.5 | 29.0 [19.8, 38.4] * | 39.2 | 10.0 | 35.9 | 56.9 | -3.1 [-5.7, -0.8] * | -1.8 [-2.3, -0.2] * |
| DbS_avg | +1.0 | 23.9 [16.1, 32.0] * | 23.3 | 13.4 | 33.1 | 54.7 | -8.2 [-13.0, -4.0] * | -4.0 [-4.8, -1.7] * |
| MLBA_avg | -1.0 | 25.8 [17.7, 34.2] * | 45.7 | 16.3 | 16.8 | 55.6 | -5.6 [-10.9, -0.9] * | -2.7 [-4.6, -0.9] * |
| MLBA_avg | -0.5 | 30.1 [21.3, 39.5] * | 49.8 | 13.8 | 26.9 | 57.8 | -1.2 [-3.9, 1.2] | -0.6 [-1.9, 0.3] |
| MLBA_avg | +0.0 | 31.4 [22.2, 41.3] * | 48.3 | 11.6 | 33.3 | 58.4 |  |  |
| MLBA_avg | +0.5 | 28.1 [19.3, 37.0] * | 36.6 | 14.5 | 32.0 | 56.4 | -3.3 [-5.8, -0.8] * | -1.9 [-2.3, -0.3] * |
| MLBA_avg | +1.0 | 22.6 [15.0, 30.2] * | 21.8 | 16.2 | 28.5 | 54.1 | -8.8 [-13.7, -4.4] * | -4.2 [-5.0, -1.7] * |

Primary tests:

- CL|beta=+0.5 minus beta 0, predicted positive. Clear in that direction False, Holm p 0.024, overlap the same way False, supports H1 False.
- CL|beta=-0.5 minus beta 0, predicted negative. Clear in that direction False, Holm p 0.726, overlap the same way False, supports H1 False.
- A1a supports H1 in this task: False

### A1b, elimination by aspects against CL

- EBA. Completeness -2.7 [-8.1, 1.9], minus CL -32.2 [-44.1, -22.0] *. Held out bits minus CL at 8, 16, 32 options -0.774, +0.142, -0.390. 32 minus 8 0.384 [0.192, 0.606] *, supports H1 True.
- EBA_best. Completeness 22.8 [15.1, 31.2] *, minus CL -6.7 [-11.0, -2.9] *. Held out bits minus CL at 8, 16, 32 options -0.289, +0.157, -0.081. 32 minus 8 0.208 [0.088, 0.355] *, supports H1 True.

### B1, concentration gap (observed minus CL, effective options as a share of the set)

| Trial | Options | Observed | CL held out | Gap |
|---|---|---|---|---|
| T1_L8 | 8 | 41.4 | 80.7 | -39.3 |
| T2_H8 | 8 | 56.0 | 79.0 | -23.0 |
| T3_L16 | 16 | 55.5 | 73.2 | -17.7 |
| T4_H16 | 16 | 40.9 | 64.2 | -23.3 |
| T5_L32 | 32 | 42.3 | 62.7 | -20.4 |
| T6_H32 | 32 | 36.6 | 67.1 | -30.5 |

- Mean gap -25.7 [-29.7, -21.8] * points, slope per doubling of options 2.9 [-1.9, 6.9] points, supports H1 False.

## laptop

### A1a, set size scaling, held out completeness (%)

| Model | beta | Completeness | 8 | 16 | 32 | Overlap % | Minus beta 0 | Overlap minus beta 0 |
|---|---|---|---|---|---|---|---|---|
| CL | -1.0 | 24.1 [16.2, 31.7] * | -2.7 | 37.9 | 26.3 | 55.6 | -31.3 [-41.7, -21.9] * | -9.0 [-10.8, -6.3] * |
| CL | -0.5 | 41.0 [31.3, 51.2] * | 22.4 | 44.8 | 46.5 | 59.2 | -14.4 [-19.4, -9.6] * | -5.4 [-6.2, -3.7] * |
| CL | +0.0 | 55.4 [43.1, 67.9] * | 52.6 | 44.6 | 63.9 | 64.6 |  |  |
| CL | +0.5 | 59.2 [45.7, 73.6] * | 59.0 | 44.9 | 69.0 | 67.2 | 3.8 [0.4, 7.5] * | 2.5 [0.6, 4.4] * |
| CL | +1.0 | 54.2 [40.6, 68.8] * | 43.9 | 44.2 | 65.5 | 64.9 | -1.2 [-7.0, 5.2] | 0.3 [-1.9, 4.1] |
| DbS_avg | -1.0 | 33.0 [25.0, 41.3] * | 24.9 | 42.7 | 29.9 | 57.9 | -26.3 [-35.8, -17.4] * | -9.0 [-10.4, -6.0] * |
| DbS_avg | -0.5 | 48.4 [37.6, 59.6] * | 51.7 | 43.9 | 49.9 | 62.1 | -10.9 [-15.3, -6.9] * | -4.9 [-5.7, -3.3] * |
| DbS_avg | +0.0 | 59.3 [46.0, 72.6] * | 72.2 | 40.3 | 66.6 | 67.0 |  |  |
| DbS_avg | +0.5 | 62.0 [47.4, 76.7] * | 69.5 | 41.8 | 72.4 | 68.8 | 2.7 [-0.4, 6.0] | 1.9 [0.0, 3.8] |
| DbS_avg | +1.0 | 57.9 [43.4, 73.3] * | 50.0 | 44.7 | 70.4 | 66.7 | -1.3 [-7.1, 4.8] | -0.3 [-2.1, 3.6] |
| MLBA_avg | -1.0 | 40.8 [31.9, 50.3] * | 53.3 | 43.7 | 33.4 | 60.4 | -18.9 [-26.9, -11.4] * | -7.9 [-9.5, -4.7] * |
| MLBA_avg | -0.5 | 52.6 [40.7, 64.8] * | 75.9 | 38.5 | 52.1 | 64.5 | -7.1 [-10.7, -3.9] * | -3.9 [-4.7, -1.9] * |
| MLBA_avg | +0.0 | 59.7 [46.1, 73.6] * | 84.8 | 33.9 | 66.4 | 68.4 |  |  |
| MLBA_avg | +0.5 | 60.7 [46.3, 75.3] * | 73.1 | 37.7 | 70.9 | 68.6 | 0.9 [-1.8, 3.8] | 0.3 [-1.2, 2.3] |
| MLBA_avg | +1.0 | 56.9 [42.6, 71.7] * | 50.2 | 43.3 | 69.0 | 66.4 | -2.8 [-8.3, 3.0] | -2.0 [-3.5, 1.5] |

Primary tests:

- CL|beta=+0.5 minus beta 0, predicted positive. Clear in that direction True, Holm p 0.032, overlap the same way True, supports H1 True.
- CL|beta=-0.5 minus beta 0, predicted negative. Clear in that direction True, Holm p 0.000, overlap the same way True, supports H1 True.
- A1a supports H1 in this task: True

### A1b, elimination by aspects against CL

- EBA. Completeness 14.0 [8.0, 20.6] *, minus CL -41.4 [-54.3, -29.3] *. Held out bits minus CL at 8, 16, 32 options -0.080, -0.563, -0.711. 32 minus 8 -0.631 [-0.923, -0.360] *, supports H1 False.
- EBA_best. Completeness 41.0 [29.0, 53.4] *, minus CL -14.4 [-23.8, -5.9] *. Held out bits minus CL at 8, 16, 32 options -0.314, +0.309, -0.466. 32 minus 8 -0.152 [-0.424, 0.094], supports H1 False.

### B1, concentration gap (observed minus CL, effective options as a share of the set)

| Trial | Options | Observed | CL held out | Gap |
|---|---|---|---|---|
| LT1_L8 | 8 | 59.4 | 53.8 | +5.6 |
| LT2_H8 | 8 | 62.1 | 67.4 | -5.2 |
| LT3_L16 | 16 | 43.2 | 75.1 | -31.9 |
| LT4_H16 | 16 | 46.4 | 50.7 | -4.3 |
| LT5_L32 | 32 | 28.1 | 57.7 | -29.6 |
| LT6_H32 | 30 | 34.2 | 53.5 | -19.3 |

- Mean gap -14.1 [-17.3, -10.9] * points, slope per doubling of options -12.8 [-16.6, -8.7] * points, supports H1 True.

Brackets give 95% intervals from 1000 participant resamples, a star marks a value whose percentile and basic intervals both exclude zero. Completeness holds each trial's benchmark at its full sample value. B1 intervals are centered as in Amendment 2.
