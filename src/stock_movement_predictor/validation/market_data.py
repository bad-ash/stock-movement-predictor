"""Shared validation for repo-local OHLCV market data."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USThanksgivingDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay

from stock_movement_predictor.features.feat_engineer import _canonicalize_ohlcv_frame

DEFAULT_FRESHNESS_DAYS = 30


class _UsMarketHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday(
            "Juneteenth",
            month=6,
            day=19,
            start_date="2021-06-19",
            observance=nearest_workday,
        ),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


_US_BUSINESS_DAY = CustomBusinessDay(calendar=_UsMarketHolidayCalendar())


class MarketDataValidationError(ValueError):
    """Raised when raw market data is invalid for downstream processing."""


def read_market_data(raw_path: str | Path) -> pd.DataFrame:
    """Read repo-local market data from parquet or CSV."""
    path = Path(raw_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def validate_market_data(
    raw_path: str | Path,
    current_date: date | datetime | pd.Timestamp | None = None,
    freshness_window: timedelta | pd.Timedelta = timedelta(days=DEFAULT_FRESHNESS_DAYS),
) -> pd.DataFrame:
    """Load and validate market data, returning a sorted canonical frame."""
    try:
        work = _canonicalize_ohlcv_frame(read_market_data(raw_path)).copy()
    except ValueError as exc:
        raise MarketDataValidationError(str(exc)) from exc
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None)
    if work["date"].isna().any():
        raise MarketDataValidationError("Date column contains null or invalid values.")

    duplicate_dates = work["date"].duplicated(keep=False)
    if duplicate_dates.any():
        dupes = work.loc[duplicate_dates, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise MarketDataValidationError(f"Duplicate date rows found: {dupes[:10]}")

    work = work.sort_values("date").reset_index(drop=True)

    missing_dates = _missing_business_dates(work["date"])
    if missing_dates:
        preview = ", ".join(day.isoformat() for day in missing_dates[:5])
        raise MarketDataValidationError(
            f"Missing business-day rows detected: {preview}"
        )

    latest_date = work["date"].dt.date.iloc[-1]
    reference_date = (
        _as_date(current_date) if current_date is not None else date.today()
    )
    freshness_delta = pd.Timedelta(freshness_window)
    if pd.Timestamp(reference_date) - pd.Timestamp(latest_date) > freshness_delta:
        raise MarketDataValidationError(
            "Data is stale: latest record is "
            f"{latest_date.isoformat()} but current date is {reference_date.isoformat()} "
            f"(freshness window: {freshness_delta})"
        )

    return work


def _missing_business_dates(dates: pd.Series) -> list[date]:
    if dates.empty:
        raise MarketDataValidationError("Market data is empty.")

    ordered = dates.drop_duplicates().sort_values()
    expected = pd.date_range(ordered.iloc[0], ordered.iloc[-1], freq=_US_BUSINESS_DAY)
    missing = expected.difference(pd.DatetimeIndex(ordered))
    return [ts.date() for ts in missing]


def _as_date(value: date | datetime | pd.Timestamp) -> date:
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None).date() if value.tzinfo else value.date()
    if isinstance(value, datetime):
        return value.date()
    return value
