# Data

Nothing in this folder is committed, so participant data never enter the repository.

## Development data

The original study's file `Updated_Choice_Overload.csv`, one row per respondent. The chosen options of the six file selection trials are in columns `ci_co_fl_1` to `ci_co_fl_6` and those of the six laptop purchase trials in `em_co_lp_1` to `em_co_lp_6`. The option definitions and rubric scores of every trial are in `src/overload/stimuli`.

Pass the file's path to the experiments with `--data`, or set `OVERLOAD_DATA` for the tests and the notebook. The analysis reads only the choice columns and never writes identifiers or participant counts.

## Confirmatory data

```bash
sh scripts/fetch_snack_data.sh data/snack
```

This downloads the four aggregate files of Thomas, Molter and Krajbich (2021) from [athms/many-item-choice](https://github.com/athms/many-item-choice) at the commit fixed in Amendment 3 of the plan, together with their MIT license and the commit hash.
