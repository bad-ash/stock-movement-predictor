from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_movement_predictor.jobs import train_register_promote
from stock_movement_predictor.serve import app as serve_app

FIXTURE_PATH = Path("tests/fixtures/spy_smoke.csv")


def test_train_then_serve_then_predict_from_committed_fixture(
    tmp_path, monkeypatch
) -> None:
    tracking_dir = tmp_path / "mlruns"
    report_path = tmp_path / "reports" / "v1_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_candidate_report()))

    monkeypatch.setenv("RAW_PATH", str(FIXTURE_PATH))
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("VALIDATION_CURRENT_DATE", "2021-08-09")
    monkeypatch.setenv("FRESHNESS_DAYS", "3")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_dir.resolve().as_uri())
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", tracking_dir.resolve().as_uri())
    monkeypatch.setenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB_SMOKE")
    monkeypatch.setenv("EXPERIMENT_NAME", "smoke-test")

    train_register_promote.main()

    rows = pd.read_csv(FIXTURE_PATH).to_dict(orient="records")
    serve_app.reset_model_cache()
    payload = serve_app.predict(serve_app.PredictRequest(ohlcv=rows))

    assert 0.0 <= float(payload["prob_up"]) <= 1.0
    assert int(payload["direction"]) in {0, 1}


def _candidate_report() -> dict[str, object]:
    return {
        "families": [
            {
                "family": "price_return",
                "best_params": {
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 1.0,
                    "min_child_weight": 1,
                    "n_estimators": 12,
                },
                "holdout_metrics": {"logloss": 0.61},
            }
        ]
    }
