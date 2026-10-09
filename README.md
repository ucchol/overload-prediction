# overload

Held out prediction tests of how people choose when displays grow in options and in the attributes shown for each option.

The question, hypotheses, models, measures and decision rules are fixed in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) before the tests run. The running story and every result, with the plan digest it was produced under, are kept in [`STORY.md`](STORY.md).

## Layout

```
ANALYSIS_PLAN.md          frozen plan, amendments appended with date and reason
STORY.md                  living story, hypothesis status and results log
src/overload/
  coding.py               attribute coding of the original study's options
  data.py                 tasks, trials, rubric shares and the choice panel
  models.py               QR, CL, DbS, MLBA (summed and averaged), set size scaling, elimination by aspects
  fit.py                  one parameter maximum likelihood, fitted, held out and alone predictions
  metrics.py              log score, attainable score, completeness, overlap, effective number
  bootstrap.py            participant resampling, intervals, Holm correction
  engine.py               one replicate of an experiment, provenance
  experiments/            one module per result of the plan
  stimuli/                option definitions and rubric scores (study materials, no participant data)
tests/                    unit tests and reproduction tests against the earlier pipeline
notebooks/                Colab notebooks, one per result, in writing, code and findings cells
results/                  outputs, one folder per result
```

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10 or later. The versions behind reported results are in `requirements-lock.txt`.

## Data

Participant data never enter the repository. Place the study's CSV file anywhere and pass its path, or set `OVERLOAD_DATA` for the tests. See `data/README.md`. Confirmatory datasets are open and are downloaded by the experiment that uses them.

## Tests

```bash
ruff check src tests && ruff format --check src tests
OVERLOAD_DATA=/path/to/Updated_Choice_Overload.csv pytest
```

Without `OVERLOAD_DATA` the reproduction tests are skipped and the unit tests still run.

## Results

```bash
python -m overload.experiments.baseline --data /path/to/Updated_Choice_Overload.csv --out results/baseline --replicates 1000 --jobs 4
```

Each result folder holds `summary.json` (estimates, intervals, provenance with the plan digest), `point.json` (per trial values of the original sample), tables in Markdown and CSV, and `replicates.jsonl` (per trial values of every bootstrap replicate, aggregate values only, not committed).

## Reporting rule

No participant counts appear in any output. Shares are reported as percentages.
