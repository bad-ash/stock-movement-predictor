from __future__ import annotations

import json

import pandas as pd
import pytest

from stock_movement_predictor.jobs import report_v1, train_register_promote
from stock_movement_predictor.validation import MarketDataValidationError


def test_report_job_fails_before_writing_outputs_on_invalid_input(
    tmp_path, monkeypatch
) -> None:
    raw_path = tmp_path / "invalid.parquet"
    _missing_volume_frame().to_parquet(raw_path, index=False)

    monkeypatch.setenv("RAW_PATH", str(raw_path))
    monkeypatch.setenv("VALIDATION_CURRENT_DATE", "2024-01-16")
    monkeypatch.setenv("FRESHNESS_DAYS", "5")
    monkeypatch.setattr(report_v1, "OUT_DIR", tmp_path / "reports")
    monkeypatch.setattr(report_v1, "OUT_JSON", tmp_path / "reports" / "v1_report.json")
    monkeypatch.setattr(report_v1, "OUT_MD", tmp_path / "reports" / "v1_report.md")

    with pytest.raises(MarketDataValidationError):
        report_v1.main()

    assert not report_v1.OUT_JSON.exists()
    assert not report_v1.OUT_MD.exists()


def test_train_job_fails_before_mlflow_work_on_invalid_input(
    tmp_path, monkeypatch
) -> None:
    raw_path = tmp_path / "invalid.parquet"
    _missing_volume_frame().to_parquet(raw_path, index=False)
    report_path = tmp_path / "reports" / "v1_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"families": []}))

    monkeypatch.setenv("RAW_PATH", str(raw_path))
    monkeypatch.setenv("REPORT_PATH", str(report_path))
    monkeypatch.setenv("VALIDATION_CURRENT_DATE", "2024-01-16")
    monkeypatch.setenv("FRESHNESS_DAYS", "5")

    with pytest.raises(MarketDataValidationError):
        train_register_promote.main()


def _missing_volume_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100 + i for i in range(len(dates))],
            "High": [101 + i for i in range(len(dates))],
            "Low": [99 + i for i in range(len(dates))],
            "Close": [100.5 + i for i in range(len(dates))],
            "AdjClose": [100.5 + i for i in range(len(dates))],
        }
    )
