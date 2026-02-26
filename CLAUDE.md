# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stock Movement Predictor (SMX) — predicts next-day SPY log returns using XGBoost regression with an ARIMA baseline. Uses MLflow for experiment tracking/model registry and DVC for data versioning.

## Commands

```bash
# Install (editable mode)
pip install -e .
pip install -e ".[dev]"    # includes pytest, black, ruff

# Data pipeline
python -m stock_movement_predictor.jobs.ingest_spy    # fetch SPY data → data/raw/spy.parquet
python -m stock_movement_predictor.jobs.tune          # hyperparameter grid search
python -m stock_movement_predictor.jobs.bootstrap     # train final model, register to MLflow

# Serving
uvicorn stock_movement_predictor.serve.app:app --reload   # FastAPI on :8000
streamlit run app/streamlit_app.py                         # Streamlit UI

# MLflow UI
mlflow ui   # http://localhost:5000

# Testing & linting
pytest
black src/ app/
ruff check src/ app/
```

## Architecture

Source is in `src/stock_movement_predictor/`. Package is installed via `pyproject.toml` with `package-dir = src`.

**Pipeline flow:** `ingest_spy` → `spy.parquet` → `features/build.py` → models → MLflow → `serve/app.py`

Key modules:
- **features/build.py**: Builds 50+ technical indicators (returns, trend, volatility, momentum, volume, calendar features) from raw OHLCV data. Supports optional z-score normalization (252-day rolling) and winsorization. Target is next-day log return.
- **models/xgboost.py**: XGBoost regression with `fit_xgb_reg()`, `tune_xgb_reg()` (grid search), and `backtest_xgb_reg()` (expanding-window walk-forward). Uses early stopping (50 rounds).
- **models/arima.py**: Wrapper around `pmdarima.auto_arima` as baseline.
- **eval/backtest.py**: `rolling_windows()` generates expanding-window CV folds (min 1500 train samples, 125-sample test blocks).
- **jobs/bootstrap.py**: Runs backtest, logs to MLflow experiment "SMX-SPY", registers model as `SMX_SPY_XGB` with Production alias.
- **jobs/tune.py**: Grid search with hardcoded train/val split (pre-2019 train, 2019-2021 val).
- **serve/app.py**: FastAPI with `/predict` endpoint, loads model from `models:/SMX_SPY_XGB/Production`.
- **medium_code/**: Research scripts for trading simulation (equity curves, Sharpe, drawdown) and Optuna walk-forward tuning.

## Key Design Decisions

- All feature engineering is centralized in `features/build.py` — add new features there.
- Time-series cross-validation only: no random splits. Backtesting uses expanding windows.
- MLflow model registry manages model versions and Production/Staging aliases.
- DVC tracks `data/raw/spy.parquet` but no remote storage is configured.
- Data source is Yahoo Finance via `yfinance`.
