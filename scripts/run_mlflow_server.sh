#!/usr/bin/env bash
set -euo pipefail

: "${BACKEND_STORE_URI:?BACKEND_STORE_URI is required}"
: "${ARTIFACTS_DESTINATION:?ARTIFACTS_DESTINATION is required}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-1}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-*}"
MLFLOW_CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-*}"

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --workers "${MLFLOW_WORKERS}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
  --cors-allowed-origins "${MLFLOW_CORS_ALLOWED_ORIGINS}" \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --serve-artifacts
