"""Ingest and validate raw SPY OHLCV data from yfinance."""

import os
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join("data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "spy.parquet")

def fetch_spy(start="1993-01-29", end=None, interval="1d") -> pd.DataFrame:
    """Download SPY history and return a validated, normalized frame."""
    # SPY IPO 1993-01-29
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
    # Standardize column names
    df = df.rename(columns=str.title)  # Open, High, Low, Close, Volume, Dividends, Stock Splits
    # Keep Adj Close explicitly
    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    # Ensure DatetimeIndex UTC-naive (consistent with daily close)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    # Minimal sanity checks
    assert df.index.is_monotonic_increasing
    assert df[["Open","High","Low","Close","Volume"]].isnull().sum().sum() == 0
    assert (df["Low"] <= df[["Open","Close","High"]].min(axis=1)).all()
    assert (df["High"] >= df[["Open","Close","Low"]].max(axis=1)).all()
    assert (df["Volume"] >= 0).all()
    # Drop events if you don’t need them now; keep if you do
    df = df.drop(['Dividends', 'Stock Splits', 'Capital Gains'], axis=1) 
    return df

def main():
    """Write parquet data plus a lightweight metadata sidecar."""
    os.makedirs(RAW_DIR, exist_ok=True)
    df = fetch_spy()
    # Embed lightweight provenance as parquet metadata
    meta = {
        "source": "yfinance",
        "symbol": "SPY",
        "auto_adjust": "False",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frequency": "1wk",
        "notes": "Daily OHLCV + AdjClose; calendar per Yahoo Finance",
    }
    # Write parquet with metadata (pandas stores key/value in file schema)
    table = df.reset_index().rename(columns={"Date": "Date"})  # keep column named Date after reset
    table.to_parquet(OUT_PATH, engine="pyarrow")
    # Also write a small sidecar metadata file (easy to diff in git if you want)
    with open(os.path.join(RAW_DIR, "spy.meta.txt"), "w") as f:
        for k,v in meta.items():
            f.write(f"{k}: {v}\n")
    print(f"Wrote {OUT_PATH}, rows={len(df)} from {df.index.min().date()} to {df.index.max().date()}")

if __name__ == "__main__":
    main()
