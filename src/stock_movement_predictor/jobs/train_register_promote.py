"""Train, register, and promote models using the canonical v1 report.

Flow:
1) Load candidate metrics/params from reports/v1_report.json (generate if missing).
2) Retrain candidate on all available data.
3) Register model version in MLflow and update aliases.
4) Promote to champion if report-based gate is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path

import mlflow
from mlflow import sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from stock_movement_predictor.eval.v1_evaluation import EvalConfig, run_ablation
from stock_movement_predictor.features.feat_engineer import make_features
from stock_movement_predictor.models.xgboost import fit_xgb_class
from stock_movement_predictor.validation import (
    DEFAULT_FRESHNESS_DAYS,
    read_market_data,
    validate_market_data,
)

SEED = 42


@dataclass(frozen=True)
class EvalResult:
    """Candidate performance and params selected for promotion decision.

    Attributes:
        metrics: Holdout metrics loaded from the canonical evaluation report.
        params: Hyperparameters associated with the selected candidate family.
    """

    metrics: dict[str, float]
    params: dict[str, float | int]


def main() -> None:
    """Execute train -> register -> promote flow using report-driven gating.

    Steps:
    1) Load (or generate) canonical report payload.
    2) Select candidate family metrics/params from report.
    3) Retrain model on full dataset and register new model version.
    4) Update `candidate` alias and promote to `champion` if gate passes.
    """
    raw_path = os.getenv("RAW_PATH", EvalConfig.raw_path)
    report_path = Path(os.getenv("REPORT_PATH", "reports/v1_report.json"))
    report_family = os.getenv("REPORT_FAMILY", "price_return")
    experiment_name = os.getenv("EXPERIMENT_NAME", "SMX-SPY")
    registered_model_name = os.getenv("REGISTERED_MODEL_NAME", "SMX_SPY_XGB")
    champion_alias = os.getenv("CHAMPION_ALIAS", "champion")
    candidate_alias = os.getenv("CANDIDATE_ALIAS", "candidate")
    holdout_start = os.getenv("HOLDOUT_START", EvalConfig.holdout_start)
    promotion_min_improvement = float(
        os.getenv("PROMOTION_MIN_LOGLOSS_IMPROVEMENT", "0.0005")
    )
    current_date = _validation_current_date()
    freshness_window = timedelta(
        days=int(os.getenv("FRESHNESS_DAYS", str(DEFAULT_FRESHNESS_DAYS)))
    )

    validate_market_data(
        raw_path, current_date=current_date, freshness_window=freshness_window
    )

    report = _load_or_generate_report(report_path, raw_path, holdout_start)
    eval_result = _candidate_from_report(report, report_family, report_path)
    X_all, y_all, _ = _get_data(raw_path)
    holdout_logloss = float(eval_result.metrics["logloss"])

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    with mlflow.start_run(run_name="train-register-promote") as run:
        for k, v in eval_result.params.items():
            mlflow.log_param(k, v)
        for k, v in eval_result.metrics.items():
            mlflow.log_metric(f"holdout_{k}", v)
        mlflow.log_param("target_horizon_days", 5)
        mlflow.log_param("holdout_start", holdout_start)
        mlflow.log_param("report_path", str(report_path))
        mlflow.log_param("report_family", report_family)
        mlflow.log_param("promotion_metric", "holdout_logloss")
        mlflow.log_param("promotion_min_improvement", promotion_min_improvement)
        mlflow.log_param("features", ",".join(X_all.columns))

        # Retrain on all available data before registration.
        n_all = len(y_all)
        val_len = max(25, int(n_all * 0.10))
        tr_idx = np.arange(0, n_all - val_len)
        val_idx = np.arange(n_all - val_len, n_all)
        final_model = fit_xgb_class(
            X_all.iloc[tr_idx],
            y_all.iloc[tr_idx],
            X_all.iloc[val_idx],
            y_all.iloc[val_idx],
            params=eval_result.params,
            seed=SEED,
        )

        mlflow_sklearn.log_model(
            final_model,
            name="model",
            registered_model_name=registered_model_name,
        )

        candidate_version = _resolve_candidate_version(
            client, run.info.run_id, registered_model_name
        )
        client.set_registered_model_alias(
            registered_model_name, candidate_alias, candidate_version
        )

        champion_ll = _champion_logloss(client, registered_model_name, champion_alias)
        if _should_promote(holdout_logloss, champion_ll, promotion_min_improvement):
            client.set_registered_model_alias(
                registered_model_name, champion_alias, candidate_version
            )
            promoted = "yes"
        else:
            promoted = "no"

        print(
            f"run_id={run.info.run_id} version={candidate_version} "
            f"holdout_logloss={holdout_logloss:.6f} champion_prev={champion_ll} promoted={promoted}"
        )


def _load_or_generate_report(
    report_path: Path,
    raw_path: str,
    holdout_start: str,
) -> dict[str, object]:
    """Load `reports/v1_report.json`, generating it if missing.

    Returns:
        Parsed report payload as a dictionary.
    """
    if report_path.exists():
        return json.loads(report_path.read_text())

    payload = _generate_report_payload(raw_path, holdout_start)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _candidate_from_report(
    report: dict[str, object], family: str, report_path: Path
) -> EvalResult:
    """Extract candidate metrics/params for one feature family from report data.

    Args:
        report: Parsed report payload.
        family: Feature family name to select (for example, `price_return`).

    Returns:
        `EvalResult` containing holdout metrics and best params for that family.

    Raises:
        ValueError: If report structure is invalid or family is not present.
    """
    rows = report.get("families", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid report format: expected list in 'families'.")

    for row in rows:
        if row.get("family") == family:
            return EvalResult(
                metrics=dict(row["holdout_metrics"]),
                params=dict(row["best_params"]),
            )
    raise ValueError(f"Family '{family}' not found in {report_path}.")


def _should_promote(
    candidate_logloss: float,
    champion_logloss: float | None,
    promotion_min_improvement: float,
) -> bool:
    """Decide whether candidate should become champion.

    Args:
        candidate_logloss: Candidate holdout logloss.
        champion_logloss: Existing champion holdout logloss, or `None`.

    Returns:
        `True` when candidate should be promoted.
    """
    if champion_logloss is None:
        return True
    return candidate_logloss <= (champion_logloss - promotion_min_improvement)


def _get_data(raw_path: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load full feature/label data used for final model retraining.

    Returns:
        Tuple of `(X, y, feature_names)` from the raw parquet source.
    """
    df = read_market_data(raw_path)
    if "Date" in df.columns:
        df = df.set_index("Date")
    elif "date" in df.columns:
        df = df.set_index("date")
    else:
        raise ValueError("Expected a 'Date' or 'date' column in market data.")
    df.index = pd.to_datetime(df.index)
    X, y, feats = make_features(df)
    return X, y, feats


def _generate_report_payload(raw_path: str, holdout_start: str) -> dict[str, object]:
    """Run canonical v1 evaluation and build a report payload.

    Returns:
        Dictionary with report metadata and evaluation output.
    """
    return {
        "report_version": "v1",
        **run_ablation(EvalConfig(raw_path=raw_path, holdout_start=holdout_start)),
    }


def _resolve_candidate_version(
    client: MlflowClient, run_id: str, registered_model_name: str
) -> str:
    """Resolve the registered model version created by the current MLflow run.

    Args:
        client: MLflow tracking client.
        run_id: Active MLflow run id.

    Returns:
        Model version string associated with this run.

    Raises:
        RuntimeError: If no registered version is found for the run.
    """
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    by_run = [v for v in versions if v.run_id == run_id]
    if not by_run:
        raise RuntimeError("Could not resolve registered model version for this run.")
    return max(by_run, key=lambda v: int(v.version)).version


def _champion_logloss(
    client: MlflowClient,
    registered_model_name: str,
    champion_alias: str,
) -> float | None:
    """Fetch current champion holdout logloss, if available.

    Args:
        client: MLflow tracking client.

    Returns:
        Champion holdout logloss, or `None` when alias/metric is missing.
    """
    try:
        champion_mv = client.get_model_version_by_alias(
            registered_model_name, champion_alias
        )
    except Exception:
        return None
    run = client.get_run(champion_mv.run_id)
    if "holdout_logloss" not in run.data.metrics:
        return None
    return float(run.data.metrics["holdout_logloss"])


def _validation_current_date() -> date | None:
    configured = os.getenv("VALIDATION_CURRENT_DATE")
    return date.fromisoformat(configured) if configured else None


if __name__ == "__main__":
    main()
