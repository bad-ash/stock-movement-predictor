from __future__ import annotations

import json
from pathlib import Path

from stock_movement_predictor.jobs import drift_report, train_register_promote

FIXTURE_PATH = Path("tests/fixtures/spy_smoke.csv")


def test_drift_report_writes_feature_and_prediction_sections(
    tmp_path, monkeypatch
) -> None:
    tracking_dir = tmp_path / "mlruns"
    report_path = tmp_path / "reports" / "v1_report.json"
    drift_path = tmp_path / "reports" / "drift_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_candidate_report()))

    common_env = {
        "RAW_PATH": str(FIXTURE_PATH),
        "VALIDATION_CURRENT_DATE": "2021-08-09",
        "FRESHNESS_DAYS": "3",
        "MLFLOW_TRACKING_URI": tracking_dir.resolve().as_uri(),
        "MLFLOW_REGISTRY_URI": tracking_dir.resolve().as_uri(),
        "REGISTERED_MODEL_NAME": "SMX_SPY_XGB_DRIFT",
        "EXPERIMENT_NAME": "drift-test",
    }
    for key, value in common_env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("REPORT_PATH", str(report_path))

    train_register_promote.main()

    monkeypatch.setenv("DRIFT_REPORT_PATH", str(drift_path))
    drift_report.main()

    payload = json.loads(drift_path.read_text())
    assert "feature_drift" in payload
    assert payload["feature_drift"]
    assert "prediction_drift" in payload
    assert isinstance(payload["prediction_drift"]["earlier_mean_prob"], float)
    assert isinstance(payload["prediction_drift"]["later_mean_prob"], float)


def test_drift_report_still_writes_when_prediction_drift_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    drift_path = tmp_path / "reports" / "drift_report.json"
    monkeypatch.setenv("RAW_PATH", str(FIXTURE_PATH))
    monkeypatch.setenv("VALIDATION_CURRENT_DATE", "2021-08-09")
    monkeypatch.setenv("FRESHNESS_DAYS", "3")
    monkeypatch.setenv("DRIFT_REPORT_PATH", str(drift_path))
    monkeypatch.setenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB_MISSING")

    drift_report.main()

    payload = json.loads(drift_path.read_text())
    assert "feature_drift" in payload
    assert "prediction_drift" in payload
    assert (
        payload["prediction_drift"] is None or "status" in payload["prediction_drift"]
    )


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
