# SMX Production Checklist

## 1. Model Lifecycle (Registry + Promotion)
- [x] Train/evaluate/register pipeline entrypoint: `python -m src.stock_movement_predictor.jobs.train_register_promote`
- [x] Registry aliases used for deployment (`candidate`, `champion`)
- [x] Promotion gate based on holdout metric (`holdout_logloss` + minimum improvement)
- [ ] Add model rollback command/runbook
- [ ] Add model deprecation/archive process

## 2. Inference Service
- [x] FastAPI app wired to registry alias (`models:/SMX_SPY_XGB@champion`)
- [x] Request contract enforces feature schema
- [x] `/health` and `/model-info` endpoints
- [ ] Add request/latency/error structured logging
- [ ] Add auth/rate limiting for non-local usage
- [ ] Add Docker image + container startup healthcheck

## 3. Data + Feature Reliability
- [x] Shared feature engineering code for training and evaluation
- [x] Leakage-prone global winsorization disabled
- [ ] Add input schema validation for raw OHLCV ingestion
- [ ] Add data freshness/staleness checks
- [ ] Add feature drift monitoring (basic PSI/stat checks)

## 4. Evaluation + Governance
- [x] Walk-forward CV + holdout + baseline comparison jobs
- [ ] Add one canonical evaluation report artifact (JSON + markdown table)
- [ ] Define and store promotion policy thresholds in config file
- [ ] Add confidence intervals/variance across multiple holdout windows

## 5. Scheduled Retraining
- [x] Retrain/promote script entrypoint exists
- [ ] Add scheduler integration (cron/GitHub Actions/systemd timer)
- [ ] Persist retrain logs and alerts (success/failure)
- [ ] Add idempotency guard for repeated same-day retrains

## 6. CI/CD + Quality Gates
- [ ] Add unit tests for feature pipeline and model-serving contract
- [ ] Add smoke test: train -> register -> serve -> predict
- [ ] Add lint/test workflow on PR
- [ ] Add release workflow for deployment artifact

## 7. Security + Ops
- [ ] Secrets/config via env vars (no hardcoded URIs/credentials)
- [ ] API dependency and vulnerability scanning
- [ ] Basic observability dashboard (request volume, errors, drift, model version)
- [ ] Incident runbook (rollback + retrain + hotfix flow)
