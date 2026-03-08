"""FastAPI inference service for the current champion model.

The API loads the model by MLflow alias and validates incoming feature keys
against the exact training schema stored in model run metadata.
"""

from __future__ import annotations

import os
from typing import Dict

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI(title="SMX Inference API", version="0.1.0")
client = MlflowClient()

_model = None
_cached_expected_features: list[str] | None = None
_model_error: str | None = None


def _fetch_expected_features() -> list[str]:
    """Fetch the champion model's feature list from MLflow run params."""
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(mv.run_id)
    feature_csv = run.data.params.get("features", "")
    feats = [f for f in feature_csv.split(",") if f]
    if not feats:
        raise RuntimeError("No feature list found in model run params.")
    return feats


def _ensure_model_loaded() -> tuple[object, list[str]]:
    """Load and cache model + schema on first use.

    Keeps API process alive even if model loading fails, so /health can
    surface the underlying error instead of refusing all connections.
    """
    global _model, _cached_expected_features, _model_error
    if _model is not None and _cached_expected_features is not None:
        return _model, _cached_expected_features

    try:
        _model = mlflow.sklearn.load_model(MODEL_URI)
        _cached_expected_features = _fetch_expected_features()
        _model_error = None
        return _model, _cached_expected_features
    except Exception as exc:  # pragma: no cover - startup/runtime environment dependent
        _model_error = str(exc)
        _model = None
        _cached_expected_features = None
        raise RuntimeError(_model_error) from exc


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature map keyed by training feature name.")


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _ensure_model_loaded()
    except Exception as exc:
        return {"status": "degraded", "model_uri": MODEL_URI, "error": str(exc)}
    return {"status": "ok", "model_uri": MODEL_URI}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    try:
        _, expected_features = _ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "feature_count": len(expected_features),
        "features": expected_features,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, float | int]:
    """Return probability of up move plus binary direction at threshold 0.5."""
    try:
        model, expected_features = _ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    incoming = payload.features
    missing = [f for f in expected_features if f not in incoming]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing[:10]}")

    X = pd.DataFrame([{f: incoming[f] for f in expected_features}])
    prob_up = float(model.predict_proba(X)[:, 1][0])
    direction = int(prob_up >= 0.5)
    return {"prob_up": prob_up, "direction": direction}
