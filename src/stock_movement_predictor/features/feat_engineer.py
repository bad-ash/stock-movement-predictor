"""Feature engineering for the v1 directional XGBoost model.

Design notes:
- Produce one consistent feature table for training/evaluation/serving.
- Keep target creation here so every job uses the same horizon definition.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np

# Tunable windows
ATR_WIN = 14
RSI_WIN = 14
CCI_WIN = 20
SMA_WINS = (20, 50, 100)
EMA_WINS = (20, 50)
KAMA_WINS = (10, 20)
ROLL_HV_WINS = (20, 63)
BOLL_WIN = 20
BOLL_K = 2
ROLL_Z_WIN = 252  # set None to disable
WINSOR_Q = None  # disabled to avoid global quantile leakage across time splits
TARGET_HORIZON_DAYS = 5  # classify whether close is higher after this many trading days


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Build training features and binary target for model training/evaluation."""
    out, raw_adjclose_by_date = _make_feature_frame(df, require_adjclose=True)

    # Binary target: whether adjusted close is higher after TARGET_HORIZON_DAYS.
    next_adjclose = raw_adjclose_by_date.shift(-TARGET_HORIZON_DAYS)
    target = (next_adjclose > raw_adjclose_by_date).where(next_adjclose.notna())
    target = target.loc[out.index].dropna().astype(int)
    out = out.loc[target.index]

    return out, target, list(out.columns)


def make_inference_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build serving-time features only (no target construction)."""
    out, _ = _make_feature_frame(df, require_adjclose=False)
    return out, list(out.columns)


def _make_feature_frame(
    df: pd.DataFrame, require_adjclose: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create the canonical feature frame and aligned adjclose series."""
    work = _canonicalize_ohlcv_frame(df)

    # Dates and required columns
    work["date"] = pd.to_datetime(work["date"])

    # Returns
    work["log_ret_1"] = np.log(work["close"]).diff(1)  # type: ignore
    work["log_ret_5"] = np.log(work["close"]).diff(5)  # type: ignore
    work["roc_5"] = work["close"].pct_change(5)
    work["roc_10"] = work["close"].pct_change(10)
    work["roc_20"] = work["close"].pct_change(20)

    # Trend (SMA, EMA, KAMA) and distances
    for w in SMA_WINS:
        work[f"sma_{w}"] = work["close"].rolling(w).mean()
    for w in EMA_WINS:
        work[f"ema_{w}"] = work["close"].ewm(span=w, adjust=False).mean()
    for w in KAMA_WINS:
        work[f"kama_{w}"] = _kama(work["close"], window=w)
        work[f"kama_{w}_slope_5"] = work[f"kama_{w}"].diff(5)
    work["close_over_sma20"] = work["close"] / work["sma_20"]
    work["ema20_over_ema50"] = work["ema_20"] / work["ema_50"]

    # Bollinger bands
    sma_b = work["close"].rolling(BOLL_WIN).mean()
    std_b = work["close"].rolling(BOLL_WIN).std()
    work["bb_upper"] = sma_b + BOLL_K * std_b
    work["bb_lower"] = sma_b - BOLL_K * std_b
    work["bb_bandwidth"] = (work["bb_upper"] - work["bb_lower"]) / sma_b
    work["bb_percent_b"] = (work["close"] - work["bb_lower"]) / (
        work["bb_upper"] - work["bb_lower"]
    )

    # Volatility (ATR, HV, Parkinson, Rogers-Satchell)
    hl = work["high"] - work["low"]
    hc = (work["high"] - work["close"].shift()).abs()
    lc = (work["low"] - work["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WIN).mean()
    work["atr_pct"] = (atr / work["close"]) * 100
    for win in ROLL_HV_WINS:
        work[f"hv_{win}"] = work["close"].pct_change().rolling(win).std() * np.sqrt(252)
    work["parkinson_20"] = _parkinson_vol(work["high"], work["low"], win=20)
    work["rs_20"] = _rogers_satchell_vol(
        work["open"], work["high"], work["low"], work["close"], win=20
    )

    # Momentum (RSI, Stoch, Williams %R, CCI, MACD)
    delta = work["close"].diff()
    gain = delta.clip(lower=0).rolling(RSI_WIN).mean()
    loss = -delta.clip(upper=0).rolling(RSI_WIN).mean()
    rs = gain / loss.replace(0, np.nan)
    work["rsi_14"] = 100 - (100 / (1 + rs))

    ll14 = work["low"].rolling(14).min()
    hh14 = work["high"].rolling(14).max()
    work["stoch_k"] = 100 * (work["close"] - ll14) / (hh14 - ll14)
    work["stoch_d"] = work["stoch_k"].rolling(3).mean()
    work["williams_r"] = -100 * (hh14 - work["close"]) / (hh14 - ll14)

    tp = (work["high"] + work["low"] + work["close"]) / 3
    work["cci_20"] = (tp - tp.rolling(CCI_WIN).mean()) / (
        0.015 * tp.rolling(CCI_WIN).std()
    )

    ema12 = work["close"].ewm(span=12, adjust=False).mean()
    ema26 = work["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    work["macd"] = macd
    work["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    work["macd_hist"] = work["macd"] - work["macd_signal"]

    # Volume and flow
    work["dollar_vol"] = work["close"] * work["volume"]
    work["vroc_10"] = work["volume"].pct_change(10)
    work["vol_z_252"] = (work["volume"] - work["volume"].rolling(252).mean()) / work[
        "volume"
    ].rolling(252).std()
    work["obv"] = (np.sign(work["close"].diff()) * work["volume"]).fillna(0).cumsum()  # type: ignore

    # Money flow (CMF, ADL, MFI)
    denom = (work["high"] - work["low"]).replace(0, np.nan)
    mfm = ((work["close"] - work["low"]) - (work["high"] - work["close"])) / denom
    mfv = mfm * work["volume"]
    work["cmf_20"] = mfv.rolling(20).sum() / work["volume"].rolling(20).sum()

    clv = ((work["close"] - work["low"]) - (work["high"] - work["close"])) / denom
    work["adl"] = (clv * work["volume"]).fillna(0).cumsum()

    mf = tp * work["volume"]
    pos_mf = np.where(tp > tp.shift(), mf, 0.0)
    neg_mf = np.where(tp < tp.shift(), mf, 0.0)
    pos_roll = pd.Series(pos_mf, index=work.index).rolling(14).sum()
    neg_roll = pd.Series(neg_mf, index=work.index).rolling(14).sum()
    mfr = pos_roll / neg_roll.replace(0, np.nan)
    work["mfi"] = 100 - (100 / (1 + mfr))

    # Drop early rows created by rolling windows
    work = work.dropna().reset_index(drop=True)
    if "adjclose" not in work.columns:
        if require_adjclose:
            raise ValueError(
                "Expected 'AdjClose'/'adjclose' column for target construction."
            )
        raw_adjclose_by_date = pd.Series(
            work["close"].to_numpy(), index=work["date"], name="adjclose"
        ).sort_index()
    else:
        raw_adjclose_by_date = pd.Series(
            work["adjclose"].to_numpy(), index=work["date"], name="adjclose"
        ).sort_index()

    # Calendar features (raw and cyclical)
    work["dow"] = work["date"].dt.weekday  # type: ignore
    work["month"] = work["date"].dt.month  # type: ignore
    work["yday"] = work["date"].dt.dayofyear  # type: ignore
    work["dow_sin"] = np.sin(2 * np.pi * work["dow"] / 7)
    work["dow_cos"] = np.cos(2 * np.pi * work["dow"] / 7)
    work["mon_sin"] = np.sin(2 * np.pi * work["month"] / 12)
    work["mon_cos"] = np.cos(2 * np.pi * work["month"] / 12)
    work["yday_sin"] = np.sin(2 * np.pi * work["yday"] / 365.25)
    work["yday_cos"] = np.cos(2 * np.pi * work["yday"] / 365.25)

    # Optional tail clipping
    if WINSOR_Q is not None:
        for c in work.columns:
            if c != "date":
                work[c] = _winsorize(work[c], q=WINSOR_Q)

    # Optional rolling z-scores (still same-day)
    if ROLL_Z_WIN:
        for c in list(work.columns):
            if c == "date":
                continue
            mu = work[c].rolling(ROLL_Z_WIN).mean()
            sd = work[c].rolling(ROLL_Z_WIN).std().replace(0, np.nan)
            work[c + "_z"] = (work[c] - mu) / sd

    # Clean up unused fields
    work = work.drop(
        columns=[
            c
            for c in ["adjclose", "vwap", "change", "changepercent"]
            if c in work.columns
        ],
        errors="ignore",
    )

    # Choose compact set; prefer z-versions if present
    prefer = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_vol",
        "log_ret_1",
        "log_ret_5",
        "roc_5",
        "roc_10",
        "roc_20",
        "rsi_14",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "cci_20",
        "macd",
        "macd_signal",
        "macd_hist",
        "sma_20",
        "sma_50",
        "sma_100",
        "ema_20",
        "ema_50",
        "kama_10",
        "kama_20",
        "kama_10_slope_5",
        "kama_20_slope_5",
        "close_over_sma20",
        "ema20_over_ema50",
        "atr_pct",
        "hv_20",
        "hv_63",
        "parkinson_20",
        "rs_20",
        "bb_bandwidth",
        "bb_percent_b",
        "vroc_10",
        "vol_z_252",
        "obv",
        "cmf_20",
        "adl",
        "mfi",
        "dow",
        "month",
        "yday",
        "dow_sin",
        "dow_cos",
        "mon_sin",
        "mon_cos",
        "yday_sin",
        "yday_cos",
    ]
    keep = ["date"]
    for base in prefer:
        zname = base + "_z"
        keep.append(zname if zname in work.columns else base)
    keep = [c for c in keep if c in work.columns]

    # Always keep raw OHLCV for downstream steps
    for base in ["open", "high", "low", "close", "volume"]:
        if base in work.columns and base not in keep:
            keep.append(base)

    # Finalize feature table
    out = work[keep].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    out = out.set_index("date").sort_index()

    return out, raw_adjclose_by_date


def _winsorize(s: pd.Series, q=0.001):
    """Clip a series to lower/upper quantiles to reduce tail outliers."""
    if q is None:
        return s
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)


def _canonicalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw OHLCV data into one expected shape:
    - `date` column
    - lowercase columns (`open`, `high`, `low`, `close`, `volume`, optional `adjclose`)
    """
    work = df.copy()

    if "Date" in work.columns:
        work = work.rename(columns={"Date": "date"})
    elif "date" not in work.columns:
        # Fallback for frames where time is stored in the index.
        if not isinstance(work.index, pd.DatetimeIndex):
            raise ValueError("Expected a 'Date'/'date' column or a DatetimeIndex.")
        work = work.reset_index().rename(columns={work.index.name or "index": "date"})

    work = work.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "AdjClose": "adjclose",
        }
    )

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return work


def _kama(series, window=10, pow1=2, pow2=30):
    """Compute a simple Kaufman Adaptive Moving Average approximation."""
    # Simple KAMA approximation for trend and slope
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    er = change / volatility
    sc = (er * (2 / (pow1 + 1) - 2 / (pow2 + 1)) + 2 / (pow2 + 1)) ** 2
    out = [np.nan] * len(series)
    if len(series) > window:
        out[window] = series.iloc[window]
        for i in range(window + 1, len(series)):
            out[i] = out[i - 1] + sc.iloc[i] * (series.iloc[i] - out[i - 1])
    return pd.Series(out, index=series.index)


def _parkinson_vol(high, low, win=20):
    """Estimate volatility from high/low ranges using Parkinson's estimator."""
    # High-low volatility estimator
    with np.errstate(divide="ignore"):
        rs = (np.log(high / low)) ** 2
    return np.sqrt((1.0 / (4.0 * np.log(2))) * rs.rolling(win).mean())


def _rogers_satchell_vol(open_, high, low, close, win=20):
    """Estimate volatility with the Rogers-Satchell directional estimator."""
    # Directional volatility estimator
    u = np.log(high / close)
    d = np.log(low / close)
    cu = np.log(high / open_)
    cd = np.log(low / open_)
    rs = (u * cu + d * cd).rolling(win).mean().clip(lower=0)
    return np.sqrt(rs)
