from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from stock_movement_predictor.validation import (
    MarketDataValidationError,
    validate_market_data,
)


def test_validation_rejects_missing_required_column(tmp_path) -> None:
    df = _sample_frame().drop(columns=["Volume"])
    path = tmp_path / "missing_volume.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(MarketDataValidationError):
        validate_market_data(
            path, current_date=date(2024, 3, 1), freshness_window=timedelta(days=5)
        )


def test_validation_rejects_duplicate_dates(tmp_path) -> None:
    df = _sample_frame()
    duplicated = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicate_dates.parquet"
    duplicated.to_parquet(path, index=False)

    with pytest.raises(MarketDataValidationError):
        validate_market_data(
            path, current_date=date(2024, 3, 1), freshness_window=timedelta(days=5)
        )


def test_validation_sorts_unique_unsorted_input(tmp_path) -> None:
    df = _sample_frame().sample(frac=1.0, random_state=42).reset_index(drop=True)
    path = tmp_path / "unsorted.parquet"
    df.to_parquet(path, index=False)

    validated = validate_market_data(
        path, current_date=date(2024, 1, 16), freshness_window=timedelta(days=5)
    )

    assert validated["date"].is_monotonic_increasing


def test_validation_rejects_missing_business_day(tmp_path) -> None:
    df = _sample_frame().drop(index=3).reset_index(drop=True)
    path = tmp_path / "missing_day.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(MarketDataValidationError):
        validate_market_data(
            path, current_date=date(2024, 3, 1), freshness_window=timedelta(days=5)
        )


def test_validation_rejects_stale_data(tmp_path) -> None:
    path = tmp_path / "stale.parquet"
    _sample_frame().to_parquet(path, index=False)

    with pytest.raises(MarketDataValidationError):
        validate_market_data(
            path, current_date=date(2024, 3, 20), freshness_window=timedelta(days=5)
        )


def _sample_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=10)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100 + i for i in range(len(dates))],
            "High": [101 + i for i in range(len(dates))],
            "Low": [99 + i for i in range(len(dates))],
            "Close": [100.5 + i for i in range(len(dates))],
            "AdjClose": [100.5 + i for i in range(len(dates))],
            "Volume": [1_000_000 + i * 1_000 for i in range(len(dates))],
        }
    )
