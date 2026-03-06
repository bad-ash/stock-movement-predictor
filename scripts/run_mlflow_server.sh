#!/usr/bin/env bash
set -euo pipefail

: "${BACKEND_STORE_URI:?BACKEND_STORE_URI is required}"
: "${ARTIFACTS_DESTINATION:?ARTIFACTS_DESTINATION is required}"

exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}" \
  --serve-artifacts
