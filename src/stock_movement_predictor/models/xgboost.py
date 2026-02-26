"""XGBoost model helpers used by training, evaluation, and backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


MetricDict = Dict[str, float]
Params = Dict[str, float | int]


@dataclass(frozen=True)
class TuneResult:
    """One hyperparameter trial result with validation metrics."""

    params: Params
    metrics: MetricDict
    best_iteration: Optional[int] = None


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Treat 0 as 0 sign; fine for this use case.
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def fit_xgb_reg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Params] = None,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """
    Fit an XGBoost classifier. If validation data is provided, uses early stopping.

    params: hyperparameters for the model. If None, uses reasonable defaults.
    """
    default_params: Params = {
        "n_estimators": 2000,        # large; early stopping picks effective number
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "gamma": 0.0,
    }
    if params:
        default_params.update(params)

    # Auto-balance positive class weight from the provided training split.
    # If caller explicitly passes scale_pos_weight in params, keep that value.
    if "scale_pos_weight" not in default_params:
        y_np = y_train.to_numpy()
        pos = float((y_np == 1).sum())
        neg = float((y_np == 0).sum())
        if pos > 0:
            default_params["scale_pos_weight"] = max(neg / pos, 1.0)

    model = xgb.XGBClassifier(
        **default_params,
        random_state=seed,
        n_jobs=-1,
        objective='binary:logistic',
        early_stopping_rounds=50,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def evaluate_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> MetricDict:
    # Threshold-dependent metric.
    y_pred = (y_prob >= threshold).astype(int)
    acc = float((y_true == y_pred).mean())
    logloss = float(log_loss(y_true, y_prob))
    brier = float(np.mean((y_prob - y_true) ** 2))
    dir_acc = _directional_accuracy(y_true, y_pred)
    return {"acc": acc, "logloss": logloss, "brier": brier, "dir_acc": dir_acc}


def tune_xgb_reg(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    param_grid: Iterable[Params],
    seed: int = 42,
    threshold: float = 0.5,
) -> List[TuneResult]:
    """
    Manual-but-structured tuning.

    Trains models on training indices and early-stops on validation indices.
    Does not touch test/backtest data.

    Args:
        X: Feature matrix.
        y: Target series.
        train_idx: Row indices for training.
        val_idx: Row indices for validation.
        param_grid: Iterable of parameter dicts.
        seed: Random seed.

    Returns:
        A list of tuning results sorted by validation MAE (ascending).
    """
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    results: List[TuneResult] = []

    for params in param_grid:
        model = fit_xgb_reg(X_train, y_train, X_val, y_val, params=params, seed=seed)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = evaluate_classification(y_val.to_numpy(), y_prob, threshold=threshold)
        best_iter = getattr(model, "best_iteration", None)
        results.append(TuneResult(params=params, metrics=metrics, best_iteration=best_iter))
        best_iter0 = model.best_iteration          # 0-based
        effective = best_iter0 + 1                 # number of trees actually used
        print(f"effective trees used: {effective}")
        print(model.best_score)

    # lower/lower/higher is better
    results.sort(key=lambda r: (r.metrics["logloss"], r.metrics["brier"], -r.metrics["acc"]))
    return results


def backtest_xgb_reg(
    X: pd.DataFrame,
    y: pd.Series,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    params: Optional[Params] = None,
    seed: int = 42,
    val_frac_within_train: float = 0.10,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, MetricDict]:
    """
    Rolling/expanding-window evaluation.

    For each fold:
    - train on tr
    - take the *last* val_frac_within_train portion of tr as validation (for early stopping)
    - test on te (never used in training or early stopping)

    Returns:
    - preds aligned to y (NaN where not predicted)
    - aggregate metrics computed over all predicted points

    IMPORTANT: Hyperparameters must be supplied (params). This function does not tune.
    """
    n = len(y)
    probs = np.full(n, np.nan, dtype=float)

    for tr, te in splits:
        if len(tr) < 50:
            continue

        # validation slice is the last chunk of the training window
        val_len = max(25, int(len(tr) * val_frac_within_train))
        tr_idx = tr[:-val_len]
        val_idx = tr[-val_len:]

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_te = X.iloc[te]

        model = fit_xgb_reg(X_tr, y_tr, X_val, y_val, params=params, seed=seed)
        probs[te] = model.predict_proba(X_te)[:, 1]

    mask = ~np.isnan(probs)
    metrics = evaluate_classification(y.to_numpy()[mask], probs[mask], threshold=threshold)
    return probs, metrics


def make_small_param_grid() -> List[Params]:
    """
    A small starter grid you can run manually from terminal.
    Keep it small at first; expand later.
    """
    grid: List[Params] = []
    for max_depth in [2, 3, 4]:
        for learning_rate in [0.02, 0.03, 0.05]:
            for subsample in [0.7, 0.8, 0.9]:
                    grid.append(
                        {
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": 0.8,
                            "reg_lambda": 1.0,
                            "min_child_weight": 1,
                            "n_estimators": 3000,  # early stopping will cut it down                            
                        }
                    )
    return grid
