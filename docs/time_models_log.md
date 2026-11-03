# Decision time from choice models, log of rounds (exploratory)

Plan in `docs/time_models_plan.md`. Each round is written here before the next starts. Development scores only, the vault stays sealed until the final list is frozen.

## Simulation checks, 3 November 2026, before any real fit

`scripts/time_models_sim.py` on the displays of snacks in three seconds (values only, 30 people), times and choices simulated from known parameters with a speed factor per person, half the people fitted and the other half predicted.

| Model | Fitted vs true expected log time, R2 | Held out R2, fitted law | Held out R2, true law | Numeric vs Monte Carlo, R2 |
|---|---|---|---|---|
| LBA | 0.983 | 0.292 | 0.289 | 0.986 |
| MLBA LBA | 0.962 | 0.197 | 0.201 | 0.977 |
| Wald race | 0.982 | 0.212 | 0.207 | 0.983 |
| Relative Wald | 0.949 | 0.086 | 0.078 | 0.954 |
| Sample until return timing | 0.924 | 0.097 | 0.085 | 0.927 |

Every process model recovers the true trial by trial expected time and reaches the held out R2 of the true law. The numerical expectation agrees with Monte Carlo up to Monte Carlo noise (100 to 200 draws per trial). Parameters are in relative time units, so they are compared through predictions, not one by one.
