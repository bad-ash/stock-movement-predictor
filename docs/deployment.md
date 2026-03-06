# Deployment Guide (Portfolio Production)

This guide sets up:
- MLflow tracking/registry service
- FastAPI inference service
- Scheduled retraining via GitHub Actions
- R2-backed model artifacts

## 1) Provision services

1. Create a Postgres database (Neon/Supabase/etc).
2. Create an R2 bucket for MLflow artifacts.
3. Create two Fly apps:
   - `smx-mlflow` (from `fly.mlflow.toml`)
   - `smx-api` (from `fly.api.toml`)

## 2) Deploy MLflow service

Required Fly secrets for `smx-mlflow`:
- `BACKEND_STORE_URI` (example: `postgresql+psycopg2://...`)
- `ARTIFACTS_DESTINATION` (example: `s3://smx-mlflow-artifacts`)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION=auto`
- `MLFLOW_S3_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com`

Deploy:
```bash
flyctl deploy -c fly.mlflow.toml
```

## 3) Deploy API service

Required Fly secrets for `smx-api`:
- `MLFLOW_TRACKING_URI` (your mlflow URL, e.g. `https://smx-mlflow.fly.dev`)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION=auto`
- `MLFLOW_S3_ENDPOINT_URL`
- `REGISTERED_MODEL_NAME=SMX_SPY_XGB`
- `MODEL_ALIAS=champion`

Deploy:
```bash
flyctl deploy -c fly.api.toml
```

## 4) Configure scheduled retraining (GitHub Actions)

Set repository secrets:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

Optional repository variables:
- `REPORT_FAMILY`
- `REGISTERED_MODEL_NAME`
- `EXPERIMENT_NAME`

Workflow file:
- `.github/workflows/retrain.yml`

## 5) First production bootstrap

Run once (locally or via workflow dispatch):
```bash
PYTHONPATH=src python -m stock_movement_predictor.jobs.ingest_spy
PYTHONPATH=src python -m stock_movement_predictor.jobs.report_v1
PYTHONPATH=src python -m stock_movement_predictor.jobs.train_register_promote
```

Then API should load:
- `models:/SMX_SPY_XGB@champion`

## 6) Smoke checks

- API health:
  - `GET /health`
- Model info:
  - `GET /model-info`
- Local sanity:
  - `./scripts/prepush_sanity.sh`
