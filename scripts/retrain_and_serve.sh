#!/usr/bin/env bash
set -euo pipefail

# 1) Train/evaluate/register/promote by alias gate.
PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.train_register_promote

# 2) Start API with champion alias model.
exec PYTHONPATH=src venv/bin/python -m uvicorn stock_movement_predictor.serve.app:app --host 0.0.0.0 --port 8000
