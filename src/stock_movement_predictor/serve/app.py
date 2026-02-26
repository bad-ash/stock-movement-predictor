"""FastAPI inference service for the current champion model.

The API loads the model by MLflow alias and validates incoming feature keys
against the exact training schema stored in model run metadata.
"""

from __future__ import annotations

from typing import Dict

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


REGISTERED_MODEL_NAME = "SMX_SPY_XGB"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

app = FastAPI(title="SMX Inference API", version="0.1.0")
client = MlflowClient()
model = mlflow.sklearn.load_model(MODEL_URI)


def _expected_features() -> list[str]:
    """Fetch the champion model's feature list from MLflow run params."""
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    run = client.get_run(mv.run_id)
    feature_csv = run.data.params.get("features", "")
    feats = [f for f in feature_csv.split(",") if f]
    if not feats:
        raise RuntimeError("No feature list found in model run params.")
    return feats


EXPECTED_FEATURES = _expected_features()


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Feature map keyed by training feature name.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "feature_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, float | int]:
    """Return probability of up move plus binary direction at threshold 0.5."""
    incoming = payload.features
    missing = [f for f in EXPECTED_FEATURES if f not in incoming]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing[:10]}")

    X = pd.DataFrame([{f: incoming[f] for f in EXPECTED_FEATURES}])
    prob_up = float(model.predict_proba(X)[:, 1][0])
    direction = int(prob_up >= 0.5)
    return {"prob_up": prob_up, "direction": direction}
