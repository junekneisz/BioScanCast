# BioScanCast Evaluation Stage

This folder evaluates probabilistic forecasts and compares forecasting sources such as human forecasts, BioScanCast, and other LLM baselines.

## What it measures

The evaluation stage scores each question/source pair using:

- **Brier score**: probability error; lower is better.
- **Log score**: how much probability was assigned to the true answer; lower is better.
- **Accuracy**: whether the top predicted bucket matches the true bucket; higher is better.
- **RPS**: Ranked Probability Score for ordered buckets; lower is better.
- **Top probability**: the largest probability in the forecast; higher means sharper forecasts.
- **Normalized entropy**: forecast uncertainty; lower means more concentrated predictions.
- **True probability**: probability assigned to the actual resolved outcome; higher is better.

## Comparing sources

If you pass more than one forecast CSV, the pipeline compares them question-by-question and produces:

- paired scatter plots
- per-question difference plots
- win-rate plots
- significance tests (paired t-test, Wilcoxon signed-rank, McNemar)

## How to run

Single source:

```bash
python -m bioscancast.stages.eval_stage.main \
  --forecasts bioscancast/stages/eval_stage/bioscancast_forecasts.csv
```

Human vs BioScanCast:

```bash
python -m bioscancast.stages.eval_stage.main \
  --forecasts bioscancast/stages/eval_stage/mock_forecasts/human_forecasts.csv \
              bioscancast/stages/eval_stage/mock_forecasts/bioscancast_forecasts.csv
```

## Outputs

The pipeline writes results to `bioscancast/stages/eval_stage/outputs/`.

Key files:

- `summary_metrics.csv`
- `summary_metrics_by_question_type.csv`
- `question_level_metrics.csv`
- `pairwise_comparison.csv`
- `significance_tests.csv`
- `source_comparison.png`
- `brier_boxplot.png`
- `log_boxplot.png`
- `rps_boxplot.png`
- `calibration_<source>.png`
- `scatter_<metric>.png`
- `differences_<metric>.png`
- `win_rate_<metric>.png`

## How to explain it quickly

The stage does three things:

1. **Scores each forecast** against the resolved answer.
2. **Summarizes performance** by source.
3. **Compares sources directly** on the same questions and checks whether the differences are likely real or just random noise.
