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

from stock_movement_predictor.features.feat_engineer import make_inference_features

app = FastAPI(title="SMX Inference API", version="0.1.0")

_model = None
_cached_expected_features: list[str] | None = None
_cached_model_uri: str | None = None
_model_error: str | None = None


def _fetch_expected_features() -> list[str]:
    """Fetch the champion model's feature list from MLflow run params."""
    client = _mlflow_client()
    mv = client.get_model_version_by_alias(_registered_model_name(), _model_alias())
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
    global _model, _cached_expected_features, _cached_model_uri, _model_error
    model_uri = _model_uri()
    if (
        _model is not None
        and _cached_expected_features is not None
        and _cached_model_uri == model_uri
    ):
        return _model, _cached_expected_features

    try:
        _model = mlflow.sklearn.load_model(model_uri)
        _cached_expected_features = _fetch_expected_features()
        _cached_model_uri = model_uri
        _model_error = None
        return _model, _cached_expected_features
    except Exception as exc:  # pragma: no cover - startup/runtime environment dependent
        _model_error = str(exc)
        _model = None
        _cached_expected_features = None
        _cached_model_uri = None
        raise RuntimeError(_model_error) from exc


def reset_model_cache() -> None:
    """Clear cached model state so tests can switch MLflow settings."""
    global _model, _cached_expected_features, _cached_model_uri, _model_error
    _model = None
    _cached_expected_features = None
    _cached_model_uri = None
    _model_error = None


def _registered_model_name() -> str:
    return os.getenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB")


def _model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "champion")


def _model_uri() -> str:
    return f"models:/{_registered_model_name()}@{_model_alias()}"


def _mlflow_client() -> MlflowClient:
    return MlflowClient()


class PredictRequest(BaseModel):
    features: Dict[str, float] | None = Field(
        default=None,
        description="Optional precomputed feature map keyed by training feature name.",
    )
    ohlcv: list[dict[str, float | str]] | None = Field(
        default=None,
        description="Optional raw OHLCV rows. Each row should include date/open/high/low/close/volume.",
    )


@app.get("/health")
def health() -> dict[str, str]:
    try:
        _ensure_model_loaded()
    except Exception as exc:
        return {"status": "degraded", "model_uri": _model_uri(), "error": str(exc)}
    return {"status": "ok", "model_uri": _model_uri()}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    try:
        _, expected_features = _ensure_model_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc

    return {
        "registered_model": _registered_model_name(),
        "alias": _model_alias(),
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

    if payload.ohlcv is not None:
        raw = pd.DataFrame(payload.ohlcv)
        if raw.empty:
            raise HTTPException(
                status_code=400, detail="ohlcv must contain at least one row."
            )
        try:
            features_df, _ = make_inference_features(raw)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid ohlcv payload: {exc}"
            ) from exc
        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough history to compute features. Send more OHLCV rows.",
            )
        latest = features_df.iloc[-1]
        missing = [f for f in expected_features if f not in latest.index]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing derived features: {missing[:10]}"
            )
        X = pd.DataFrame([{f: float(latest[f]) for f in expected_features}])
    elif payload.features is not None:
        incoming = payload.features
        missing = [f for f in expected_features if f not in incoming]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Missing features: {missing[:10]}"
            )
        X = pd.DataFrame([{f: incoming[f] for f in expected_features}])
    else:
        raise HTTPException(
            status_code=400, detail="Provide either 'ohlcv' or 'features'."
        )

    prob_up = float(model.predict_proba(X)[:, 1][0])
    direction = int(prob_up >= 0.5)
    return {"prob_up": prob_up, "direction": direction}
