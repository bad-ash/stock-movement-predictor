import pandas as pd

from stock_movement_predictor.features.feat_engineer import _canonicalize_ohlcv_frame


def test_canonicalize_maps_ingested_columns() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Open": [1.0, 1.1],
            "High": [1.2, 1.3],
            "Low": [0.9, 1.0],
            "Close": [1.1, 1.2],
            "AdjClose": [1.05, 1.15],
            "Volume": [100, 120],
        }
    )

    out = _canonicalize_ohlcv_frame(df)

    assert "date" in out.columns
    for col in ["open", "high", "low", "close", "adjclose", "volume"]:
        assert col in out.columns

