# SMX V1 Evaluation Report

Generated (UTC): 2026-02-26T05:04:43+00:00

## Configuration
- holdout_start: `2022-01-01`
- wf_min_train: `1000`
- wf_test_h: `63`
- wf_max_folds: `8`
- threshold: `0.5`
- grid_size: `12`

## Dataset
- dev_rows: `6782`
- holdout_rows: `999`
- folds: `8`

## Baseline
- majority logloss: `0.677166`
- majority brier: `0.242051`
- majority acc: `0.589590`

## Family Comparison

| family | features | logloss | brier | acc | d_logloss_vs_majority | d_acc_vs_majority |
|---|---:|---:|---:|---:|---:|---:|
| full | 57 | 0.677042 | 0.241989 | 0.589590 | -0.000124 | +0.000000 |
| price_return | 17 | 0.673388 | 0.240304 | 0.589590 | -0.003779 | +0.000000 |
| technical | 31 | 0.676521 | 0.241735 | 0.589590 | -0.000646 | +0.000000 |
| calendar | 9 | 0.678033 | 0.242461 | 0.596597 | +0.000867 | +0.007007 |
