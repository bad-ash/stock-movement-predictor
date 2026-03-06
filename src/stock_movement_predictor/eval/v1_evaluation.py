"""Reusable v1 evaluation pipeline.

This module centralizes ablation logic so all jobs (CLI printouts, report
generation, and promotion gating) share the same evaluation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from stock_movement_predictor.eval.backtest import rolling_windows
from stock_movement_predictor.features.feat_engineer import make_features
from stock_movement_predictor.models.xgboost import (
    Params,
    TuneResult,
    evaluate_classification,
    fit_xgb_class,
)


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for one v1 ablation/evaluation run."""

    raw_path: str = "data/raw/spy.parquet"
    holdout_start: str = "2022-01-01"
    wf_min_train: int = 1000
    wf_test_h: int = 63
    wf_max_folds: int | None = 8
    threshold: float = 0.5
    seed: int = 42


def run_ablation(config: EvalConfig) -> dict[str, object]:
    """Run v1 family ablation and return a structured result payload.

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary containing run config, dataset stats, majority baseline,
        and per-family tuning/holdout results.
    """
    X, y = load_xy(config.raw_path)
    X_dev, y_dev, X_holdout, y_holdout = split_dev_holdout(X, y, config.holdout_start)
    splits = list(
        rolling_windows(
            len(y_dev),
            min_train=config.wf_min_train,
            test_h=config.wf_test_h,
            max_folds=config.wf_max_folds,
        )
    )
    if not splits:
        raise ValueError("No walk-forward folds produced; adjust settings.")

    families = make_feature_families(X.columns)
    grid = make_ablation_grid()
    majority_holdout = score_majority_baseline(y_dev, y_holdout, threshold=config.threshold)

    family_results: list[dict[str, object]] = []
    for name, selector in families.items():
        cols = selector(X.columns)
        X_dev_f = X_dev[cols]
        X_holdout_f = X_holdout[cols]
        cv_results = walkforward_tune_xgb(
            X_dev_f,
            y_dev,
            grid,
            splits,
            threshold=config.threshold,
            seed=config.seed,
        )
        best = cv_results[0]
        holdout = fit_xgb_on_dev_then_score_holdout(
            X_dev_f,
            y_dev,
            X_holdout_f,
            y_holdout,
            best.params,
            threshold=config.threshold,
            seed=config.seed,
        )
        family_results.append(
            {
                "family": name,
                "feature_count": len(cols),
                "best_params": best.params,
                "holdout_metrics": holdout,
                "delta_vs_majority": {
                    "logloss": float(holdout["logloss"] - majority_holdout["logloss"]),
                    "acc": float(holdout["acc"] - majority_holdout["acc"]),
                },
            }
        )

    return {
        "config": {
            "raw_path": config.raw_path,
            "holdout_start": config.holdout_start,
            "wf_min_train": config.wf_min_train,
            "wf_test_h": config.wf_test_h,
            "wf_max_folds": config.wf_max_folds,
            "threshold": config.threshold,
            "seed": config.seed,
            "grid_size": len(grid),
        },
        "dataset": {
            "dev_rows": len(X_dev),
            "holdout_rows": len(X_holdout),
            "folds": len(splits),
        },
        "majority_baseline": majority_holdout,
        "families": family_results,
    }


def load_xy(raw_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw parquet data and build model-ready features/target.

    Args:
        raw_path: Path to parquet file with OHLCV data.

    Returns:
        Tuple of `(X, y)` where `X` is feature matrix and `y` is binary target.
    """
    df = pd.read_parquet(raw_path).set_index("Date")
    X, y, _ = make_features(df)
    return X, y


def split_dev_holdout(
    X: pd.DataFrame, y: pd.Series, holdout_start: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create time-based development and holdout splits.

    Args:
        X: Feature matrix indexed by date.
        y: Target series indexed by date.
        holdout_start: First date (inclusive) for holdout region.

    Returns:
        `(X_dev, y_dev, X_holdout, y_holdout)`.
    """
    holdout_mask = X.index >= pd.Timestamp(holdout_start)
    X_dev, y_dev = X.loc[~holdout_mask], y.loc[~holdout_mask]
    X_holdout, y_holdout = X.loc[holdout_mask], y.loc[holdout_mask]
    return X_dev, y_dev, X_holdout, y_holdout


def make_feature_families(columns: Iterable[str]) -> Dict[str, Callable[[Iterable[str]], List[str]]]:
    """Build family selectors used by ablation.

    Args:
        columns: Available feature columns.

    Returns:
        Mapping from family name to selector function.
    """
    price_return = {
        "open", "high", "low", "close", "volume", "dollar_vol",
        "log_ret_1", "log_ret_5", "roc_5", "roc_10", "roc_20", "vroc_10",
    }
    technical = {
        "rsi_14", "stoch_k", "stoch_d", "williams_r", "cci_20",
        "macd", "macd_signal", "macd_hist",
        "sma_20", "sma_50", "sma_100", "ema_20", "ema_50",
        "kama_10", "kama_20", "kama_10_slope_5", "kama_20_slope_5",
        "close_over_sma20", "ema20_over_ema50",
        "atr_pct", "hv_20", "hv_63", "parkinson_20", "rs_20",
        "bb_bandwidth", "bb_percent_b",
        "vol_z_252", "obv", "cmf_20", "adl", "mfi",
    }
    calendar = {
        "dow", "month", "yday", "dow_sin", "dow_cos", "mon_sin", "mon_cos", "yday_sin", "yday_cos",
    }
    return {
        "full": lambda cols: list(cols),
        "price_return": lambda cols: _select_family(cols, price_return),
        "technical": lambda cols: _select_family(cols, technical),
        "calendar": lambda cols: _select_family(cols, calendar),
    }


def make_ablation_grid() -> List[Params]:
    """Build a small, stable hyperparameter grid for v1 ablation.

    Returns:
        List of XGBoost parameter dictionaries.
    """
    grid: List[Params] = []
    for max_depth in [2, 3, 4]:
        for learning_rate in [0.03, 0.05]:
            for subsample in [0.7, 0.9]:
                grid.append(
                    {
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "subsample": subsample,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.0,
                        "min_child_weight": 1,
                        "n_estimators": 3000,
                    }
                )
    return grid


def walkforward_tune_xgb(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    param_grid: Iterable[Params],
    splits: List[tuple[np.ndarray, np.ndarray]],
    threshold: float,
    seed: int,
) -> List[TuneResult]:
    """Tune parameters with walk-forward CV across development folds.

    Args:
        X_dev: Development feature matrix.
        y_dev: Development target.
        param_grid: Hyperparameter configs to evaluate.
        splits: Walk-forward `(train_idx, test_idx)` folds.
        threshold: Probability threshold for threshold-based metrics.
        seed: Random seed.

    Returns:
        Sorted list of tuning results.
    """
    results: List[TuneResult] = []
    for params in param_grid:
        fold_metrics: List[dict[str, float]] = []
        for train_window, test_window in splits:
            tr_idx, val_idx = _train_val_split_from_train_window(train_window)
            model = fit_xgb_class(
                X_dev.iloc[tr_idx],
                y_dev.iloc[tr_idx],
                X_dev.iloc[val_idx],
                y_dev.iloc[val_idx],
                params=params,
                seed=seed,
            )
            y_prob = model.predict_proba(X_dev.iloc[test_window])[:, 1]
            fold_metrics.append(
                evaluate_classification(
                    y_dev.iloc[test_window].to_numpy(),
                    y_prob,
                    threshold=threshold,
                )
            )
        results.append(TuneResult(params=params, metrics=_avg_metrics(fold_metrics)))
    results.sort(key=lambda r: (r.metrics["logloss"], r.metrics["brier"], -r.metrics["acc"]))
    return results


def fit_xgb_on_dev_then_score_holdout(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    best_params: Params,
    threshold: float,
    seed: int,
) -> dict[str, float]:
    """Fit one dev model with early stopping and score on holdout.

    Args:
        X_dev: Development features.
        y_dev: Development target.
        X_holdout: Holdout features.
        y_holdout: Holdout target.
        best_params: Chosen model hyperparameters.
        threshold: Probability threshold for threshold-based metrics.
        seed: Random seed.

    Returns:
        Holdout metric dictionary.
    """
    n_dev = len(y_dev)
    val_len = max(25, int(n_dev * 0.10))
    tr_idx = np.arange(0, n_dev - val_len)
    val_idx = np.arange(n_dev - val_len, n_dev)
    model = fit_xgb_class(
        X_dev.iloc[tr_idx],
        y_dev.iloc[tr_idx],
        X_dev.iloc[val_idx],
        y_dev.iloc[val_idx],
        params=best_params,
        seed=seed,
    )
    y_prob = model.predict_proba(X_holdout)[:, 1]
    return evaluate_classification(y_holdout.to_numpy(), y_prob, threshold=threshold)


def score_majority_baseline(y_dev: pd.Series, y_holdout: pd.Series, threshold: float) -> dict[str, float]:
    """Score a constant-probability majority baseline on holdout.

    Args:
        y_dev: Development target used to estimate base positive rate.
        y_holdout: Holdout target used for scoring.
        threshold: Probability threshold for threshold-based metrics.

    Returns:
        Holdout metric dictionary.
    """
    p_up = float(y_dev.mean())
    y_prob = np.full(len(y_holdout), p_up, dtype=float)
    return evaluate_classification(y_holdout.to_numpy(), y_prob, threshold=threshold)


def _train_val_split_from_train_window(
    train_window: np.ndarray,
    val_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one train window into early-train and tail-validation indices."""
    val_len = max(25, int(len(train_window) * val_frac))
    tr_idx = train_window[:-val_len]
    val_idx = train_window[-val_len:]
    return tr_idx, val_idx


def _avg_metrics(metric_rows: List[dict[str, float]]) -> dict[str, float]:
    """Average each metric key across a list of metric dictionaries."""
    keys = metric_rows[0].keys()
    return {k: float(np.mean([row[k] for row in metric_rows])) for k in keys}


def _select_family(columns: Iterable[str], allowed: set[str]) -> List[str]:
    """Select columns whose base feature name belongs to `allowed`."""
    return [c for c in columns if _base_name(c) in allowed]


def _base_name(col: str) -> str:
    """Strip `_z` suffix so family matching works for z-scored features."""
    return col[:-2] if col.endswith("_z") else col
