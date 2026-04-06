"""Generate a simple feature and prediction drift report from local repo data."""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from stock_movement_predictor.features.feat_engineer import make_inference_features
from stock_movement_predictor.validation import (
    DEFAULT_FRESHNESS_DAYS,
    validate_market_data,
)

OUT_PATH = Path("reports/drift_report.json")


def main() -> None:
    raw_path = os.getenv("RAW_PATH", "data/raw/spy.parquet")
    out_path = Path(os.getenv("DRIFT_REPORT_PATH", str(OUT_PATH)))
    current_date = _validation_current_date()
    freshness_window = timedelta(
        days=int(os.getenv("FRESHNESS_DAYS", str(DEFAULT_FRESHNESS_DAYS)))
    )

    validated = validate_market_data(
        raw_path, current_date=current_date, freshness_window=freshness_window
    )
    earlier, later = _split_halves(validated)

    payload = {
        "window_summary": {
            "earlier_row_count": len(earlier),
            "later_row_count": len(later),
        },
        "feature_drift": _feature_drift(earlier, later),
        "prediction_drift": _prediction_drift(earlier, later),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def _split_halves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    midpoint = len(df) // 2
    return df.iloc[:midpoint].copy(), df.iloc[midpoint:].copy()


def _feature_drift(
    earlier: pd.DataFrame, later: pd.DataFrame
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = {
            "reference_mean": float(earlier[col].mean()),
            "recent_mean": float(later[col].mean()),
        }
    return out


def _prediction_drift(
    earlier: pd.DataFrame, later: pd.DataFrame
) -> dict[str, float] | dict[str, str] | None:
    try:
        model = mlflow.sklearn.load_model(_model_uri())
        expected_features = _expected_features()
        earlier_prob = _predict_window(model, expected_features, earlier)
        later_prob = _predict_window(model, expected_features, later)
    except Exception as exc:
        return {"status": f"prediction drift not computable: {exc}"}

    return {
        "earlier_mean_prob": float(earlier_prob),
        "later_mean_prob": float(later_prob),
    }


def _predict_window(
    model: object, expected_features: list[str], window: pd.DataFrame
) -> float:
    features, _ = make_inference_features(window)
    if features.empty:
        raise ValueError("not enough history to compute inference features")
    latest = features.iloc[-1]
    X = pd.DataFrame([{name: float(latest[name]) for name in expected_features}])
    return float(model.predict_proba(X)[:, 1][0])


def _expected_features() -> list[str]:
    mv = MlflowClient().get_model_version_by_alias(
        _registered_model_name(), _model_alias()
    )
    run = MlflowClient().get_run(mv.run_id)
    feature_csv = run.data.params.get("features", "")
    features = [name for name in feature_csv.split(",") if name]
    if not features:
        raise RuntimeError("No feature list found in model run params.")
    return features


def _registered_model_name() -> str:
    return os.getenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB")


def _model_alias() -> str:
    return os.getenv("MODEL_ALIAS", "champion")


def _model_uri() -> str:
    return f"models:/{_registered_model_name()}@{_model_alias()}"


def _validation_current_date() -> date | None:
    configured = os.getenv("VALIDATION_CURRENT_DATE")
    return date.fromisoformat(configured) if configured else None


if __name__ == "__main__":
    main()
