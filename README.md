# Stock Movement Predictor (V1)

Clean v1 scope:
- Ingest SPY data
- Train/evaluate/register/promote XGBoost model in MLflow
- Serve champion model behind FastAPI
- Keep one ablation job for evaluation governance

## Project Entry Points
- Ingest data:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.ingest_spy`
- Train + register + promote:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.train_register_promote`
- Run ablation evaluation:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.ablation_compare`
- Generate official v1 report artifacts:
  - `PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.report_v1`
- Start API:
  - `PYTHONPATH=src venv/bin/python -m uvicorn stock_movement_predictor.serve.app:app --host 0.0.0.0 --port 8000`
- One-shot retrain then serve:
  - `./scripts/retrain_and_serve.sh`

## Core Files
- Feature engineering: `src/stock_movement_predictor/features/feat_engineer.py`
- Model training/eval helpers: `src/stock_movement_predictor/models/xgboost.py`
- Retrain/promotion pipeline: `src/stock_movement_predictor/jobs/train_register_promote.py`
- Ablation evaluator: `src/stock_movement_predictor/jobs/ablation_compare.py`
- Official report job: `src/stock_movement_predictor/jobs/report_v1.py`
- Shared eval utilities: `src/stock_movement_predictor/eval/v1_evaluation.py`
- API service: `src/stock_movement_predictor/serve/app.py`
- Production checklist: `docs/production_checklist.md`
