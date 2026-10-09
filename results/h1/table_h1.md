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
| CL | -1.0 | 21.6 [14.8, 28.2] * | -0.2 | 32.9 | 23.4 | 54.5 | -28.7 [-38.3, -19.5] * | -8.7 [-10.5, -5.8] * |
| CL | -0.5 | 36.7 [28.0, 46.2] * | 18.6 | 40.5 | 42.0 | 57.9 | -13.6 [-18.4, -8.9] * | -5.3 [-6.2, -3.4] * |
| CL | +0.0 | 50.3 [38.9, 62.0] * | 47.3 | 41.6 | 57.5 | 63.2 |  |  |
| CL | +0.5 | 52.3 [40.1, 65.5] * | 52.6 | 41.5 | 59.5 | 64.1 | 2.0 [-1.6, 5.5] | 0.9 [-0.9, 3.3] |
| CL | +1.0 | 45.6 [33.9, 58.1] * | 37.7 | 38.8 | 53.8 | 61.0 | -4.6 [-11.2, 1.4] | -2.2 [-4.0, 1.6] |
| DbS_avg | -1.0 | 29.7 [22.4, 37.5] * | 21.1 | 37.9 | 27.9 | 56.4 | -25.6 [-34.7, -16.7] * | -9.1 [-10.7, -5.9] * |
| DbS_avg | -0.5 | 44.6 [34.6, 55.4] * | 44.1 | 41.0 | 47.2 | 60.3 | -10.8 [-15.2, -6.5] * | -5.2 [-6.1, -3.2] * |
| DbS_avg | +0.0 | 55.3 [42.7, 68.7] * | 64.8 | 39.3 | 62.1 | 65.5 |  |  |
| DbS_avg | +0.5 | 56.1 [43.1, 70.4] * | 61.5 | 40.6 | 64.3 | 66.1 | 0.8 [-2.7, 4.1] | 0.6 [-1.0, 2.8] |
| DbS_avg | +1.0 | 49.6 [36.4, 63.1] * | 42.3 | 40.3 | 59.2 | 63.1 | -5.7 [-12.1, 0.5] | -2.3 [-3.8, 1.6] |
| MLBA_avg | -1.0 | 37.6 [29.4, 46.8] * | 45.4 | 39.8 | 32.7 | 59.2 | -19.6 [-27.8, -11.7] * | -8.2 [-10.2, -4.8] * |
| MLBA_avg | -0.5 | 50.0 [38.8, 62.1] * | 67.8 | 36.7 | 51.4 | 63.2 | -7.1 [-10.9, -3.6] * | -4.2 [-5.2, -1.9] * |
| MLBA_avg | +0.0 | 57.2 [43.8, 70.7] * | 77.2 | 34.3 | 64.0 | 67.4 |  |  |
| MLBA_avg | +0.5 | 56.3 [42.9, 70.6] * | 64.6 | 38.0 | 65.1 | 66.7 | -0.9 [-4.3, 2.4] | -0.7 [-1.8, 1.9] |
| MLBA_avg | +1.0 | 50.0 [36.8, 64.1] * | 42.2 | 39.7 | 60.5 | 64.1 | -7.1 [-13.7, -1.1] * | -3.3 [-4.4, 0.6] |

Primary tests:

- CL|beta=+0.5 minus beta 0, predicted positive. Clear in that direction False, Holm p 0.268, overlap the same way True, supports H1 False.
- CL|beta=-0.5 minus beta 0, predicted negative. Clear in that direction True, Holm p 0.000, overlap the same way True, supports H1 True.
- A1a supports H1 in this task: False

### A1b, elimination by aspects against CL

- EBA. Completeness 16.7 [11.0, 23.2] *, minus CL -33.6 [-43.8, -24.0] *. Held out bits minus CL at 8, 16, 32 options -0.037, -0.601, -0.460. 32 minus 8 -0.423 [-0.634, -0.225] *, supports H1 False.
- EBA_best. Completeness 38.7 [27.3, 50.6] *, minus CL -11.6 [-20.3, -3.6] *. Held out bits minus CL at 8, 16, 32 options -0.302, +0.326, -0.403. 32 minus 8 -0.101 [-0.347, 0.105], supports H1 False.

### B1, concentration gap (observed minus CL, effective options as a share of the set)

| Trial | Options | Observed | CL held out | Gap |
|---|---|---|---|---|
| LT1_L8 | 8 | 59.4 | 57.4 | +2.0 |
| LT2_H8 | 8 | 62.1 | 68.3 | -6.2 |
| LT3_L16 | 16 | 43.2 | 77.2 | -34.1 |
| LT4_H16 | 16 | 46.4 | 57.8 | -11.4 |
| LT5_L32 | 32 | 28.1 | 60.0 | -31.9 |
| LT6_H32 | 30 | 34.2 | 52.7 | -18.4 |

- Mean gap -16.7 [-20.1, -13.1] * points, slope per doubling of options -12.0 [-16.0, -8.0] * points, supports H1 True.

Brackets give 95% intervals from 1000 participant resamples, a star marks a value whose percentile and basic intervals both exclude zero. Completeness holds each trial's benchmark at its full sample value. B1 intervals are centered as in Amendment 2.
