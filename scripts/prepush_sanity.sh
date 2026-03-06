#!/usr/bin/env bash
set -euo pipefail

# Fast local sanity checks before pushing.
PYTHONPATH=src venv/bin/python -m pytest -q
PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.report_v1
PYTHONPATH=src venv/bin/python -m stock_movement_predictor.jobs.train_register_promote

echo "prepush sanity checks passed"
