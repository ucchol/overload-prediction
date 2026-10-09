# Story

Living record of the paper's story, the status of every hypothesis and every result. Updated after each step.

## The story in one paragraph

When displays grow in the number of options and in the attributes shown for each option, people do not integrate everything shown. They adapt (Payne, Bettman and Johnson, 1993), screening options on a few salient cues and choosing among what survives, so their choices narrow rather than become noisier. Choice models are formal statements of how choices are computed, and held out prediction decides between them. Each hypothesis predicts a behavioral signature and the model structure that should predict better where that signature appears. The intruder's file choice and the buyer's laptop choice are the development data, and open datasets give the confirmatory tests.

## Hypotheses and status

| | Behavior | Algorithm |
|---|---|---|
| H1 Narrowing with set size | B1 concentration gap. Level known (S20), slope to run | A1a set size scaling to run (beta = 1 against 0 known for comparison models, confirmed in R0), A1b screening to run |
| H2 Comparison under sparse information | B2 revealed weights, known for laptops | A2 known on development data, confirmatory dataset to fix |
| H3 Inputs over rules | B3 known | A3a ladder known, A3b independent cue to fix and run |
| H4 Cost depends on cue validity | to specify | to specify |

## Plan versions

| Version | Date | SHA-256 of ANALYSIS_PLAN.md |
|---|---|---|
| 1.0 | 9 Oct 2026 | 6bc0e96185501c9ae5bf2fcd14a6a2fb617466ee3c740d505e4eb773e5c75cd8 |
| 1.0 with Amendment 1 | 9 Oct 2026 | 2953b6ecaf8420cacc3e7112d8effd153e986e4cae1c34887a2d1130729e6ed0 |

## Results log

### 9 Oct 2026. Pipeline rebuilt and verified

The package reproduces the earlier pipeline from the raw data. Rubric shares match exactly, and per trial held out and fitted log scores and overlaps of QR, CL, DbS and MLBA match within 1e-6 bits (largest difference 2.5e-8, from the earlier optimizer's tolerance), for the original sample and for bootstrap replicates 0, 1, 499 and 999. Published completeness values round to the same first decimal.

### 9 Oct 2026. R0, the baseline (descriptive, development data)

Held out completeness, 95 percent intervals, 1000 participant resamples (`results/baseline`).

| | QR | CL | DbS_avg | MLBA_avg | DbS_sum | MLBA_sum |
|---|---|---|---|---|---|---|
| Files | 21.9 | 29.5 [20.2, 39.2] | 32.1 | 31.4 | 23.9 | 22.6 |
| Laptops | 47.2 | 50.3 [38.9, 62.0] | 55.3 | 57.2 | 49.6 | 50.0 |

- Averaged comparison scores predicted unseen trials at least as well as CL in both tasks. DbS_avg exceeded CL by 2.6 points [0.7, 4.4] for files and 5.1 [3.1, 7.3] for laptops, MLBA_avg by 1.8 [-0.9, 4.6] and 6.9 [3.4, 11.0]. For laptops, overlap rose with them, by 2.3 and 4.2 points, both clear.
- Summing the comparisons over every rival cost 5.6 and 6.9 points against CL for files and nothing clear for laptops, so how sensitivity scales with set size matters, which is test A1a of H1.
- Every published interval of the earlier pipeline is reproduced once the benchmark is held at its full sample value (Amendment 1).

Story so far. Comparing options with each other, averaged over rivals, predicts at least as well as valuing each option on its own. The open question for H1 is what happens to that comparison as sets grow.

## Setup facts checked against the data

- The data file holds recruitment platform identifiers (a Prolific ID column and Prolific screening questions) and a nine question security knowledge quiz (SQL injection, SSL, firewall, proxy, DDoS, IDS, honeypot, TCP, Nmap). The Study Design should describe recruitment and screening from these.

## Open decisions

- H2 confirmatory dataset. Confirm that Hensher (2006) is openly downloadable, otherwise choose another open dataset that varies the attributes shown for the same options. Fix by amendment before inspecting outcomes.
- H3 independent cue. Choose the external source of the filename cue before running A3b.
- H4. Specify cue validity, the regret contrast and the screening model before running.
