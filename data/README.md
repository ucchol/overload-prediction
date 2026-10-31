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

```bash
sh scripts/fetch_gaze_data.sh data/gaze
```

This downloads three prepared datasets of choices between two or three foods from [athms/gaze-bias-differences](https://github.com/athms/gaze-bias-differences) (Thomas, Molter, Krajbich, Heekeren and Mohr, 2019, MIT license) at the commit fixed in Amendment 6, with the original file of Folke and colleagues (2016) for its confidence ratings, the license and the commit hash.

### Search and satisficing data (manual download)

The replication package of Caplin, Dean and Martin (2011), openICPSR project 116100, version 1 (CC BY 4.0), needs a free openICPSR sign in, so it is not fetched by a script. Download the project as a zip from [openicpsr.org/openicpsr/project/116100/version/V1/view](https://www.openicpsr.org/openicpsr/project/116100/version/V1/view), then run

```bash
sh scripts/prepare_cdm_data.sh PATH_TO/116100-V1.zip data/cdm
```

which checks the click file against the hash fixed in Amendment 7 and copies it with the license to `data/cdm`.

### Eye tracking under time pressure (manual download)

The replication package of Reutskaja, Nagel, Camerer and Rangel (2011), openICPSR project 112415, version 1 (CC BY 4.0), also needs a free sign in. Download it as a zip from [openicpsr.org/openicpsr/project/112415](https://www.openicpsr.org/openicpsr/project/112415), then run

```bash
sh scripts/prepare_reutskaja_data.sh PATH_TO/112415-V1.zip data/reutskaja
```

which checks the trial file against the hash fixed in Amendment 8.

