# Stock Movement Predictor (V1)

Clean v1 scope:
- Ingest SPY data
- Evaluate model families with a canonical report artifact
- Train/register/promote XGBoost model in MLflow using report-driven gating
- Serve champion model behind FastAPI
- Keep fast sanity tests for pre-push checks

## Project Entry Points
- Ingest data:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.ingest_spy`
- Generate canonical evaluation report:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.report_v1`
- Train + register + promote:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.train_register_promote`
  - uses `reports/v1_report.json` and `REPORT_FAMILY` from `train_register_promote.py`
- Run ablation evaluation:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.ablation_compare`
- Start API:
  - `PYTHONPATH=src venv/bin/python -m uvicorn stock_movement_predictor.serve.app:app --host 0.0.0.0 --port 8000`
- One-shot retrain then serve:
  - `./scripts/retrain_and_serve.sh`
- Pre-push sanity check:
  - `./scripts/prepush_sanity.sh`

## Evaluation and Promotion Flow
1. `report_v1` runs the shared evaluation pipeline and writes:
   - `reports/v1_report.json`
   - `reports/v1_report.md`
2. `train_register_promote` selects one feature family from the report, retrains a model, registers a new MLflow model version, and updates aliases:
   - `candidate`
   - `champion` (if promotion gate passes)

## Tests
- Fast local tests:
  - `PYTHONPATH=src venv/bin/python -m pytest -q`
- Current suite:
  - `tests/test_feat_engineer.py`
  - `tests/test_xgboost_metrics.py`
  - `tests/test_report_v1.py`

## Core Files
- Feature engineering: `src/stock_movement_predictor/features/feat_engineer.py`
- Model training/eval helpers: `src/stock_movement_predictor/models/xgboost.py`
- Retrain/promotion pipeline: `src/stock_movement_predictor/jobs/train_register_promote.py`
- Ablation CLI: `src/stock_movement_predictor/jobs/ablation_compare.py`
- Official report job: `src/stock_movement_predictor/jobs/report_v1.py`
- Shared eval utilities: `src/stock_movement_predictor/eval/v1_evaluation.py`
- API service: `src/stock_movement_predictor/serve/app.py`
- Production checklist: `docs/production_checklist.md`
