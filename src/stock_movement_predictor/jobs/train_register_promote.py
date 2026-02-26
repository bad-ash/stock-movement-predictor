"""Train, register, and promote models using the canonical v1 report.

Flow:
1) Load candidate metrics/params from reports/v1_report.json (generate if missing).
2) Retrain candidate on all available data.
3) Register model version in MLflow and update aliases.
4) Promote to champion if report-based gate is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from stock_movement_predictor.eval.v1_evaluation import EvalConfig, run_ablation
from stock_movement_predictor.features.feat_engineer import make_features
from stock_movement_predictor.models.xgboost import fit_xgb_reg


RAW_PATH = "data/raw/spy.parquet"
REPORT_PATH = Path("reports/v1_report.json")
REPORT_FAMILY = "price_return"
EXPERIMENT_NAME = "SMX-SPY"
REGISTERED_MODEL_NAME = "SMX_SPY_XGB"
CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"
HOLDOUT_START = "2022-01-01"
PROMOTION_MIN_LOGLOSS_IMPROVEMENT = 0.0005
SEED = 42


@dataclass(frozen=True)
class EvalResult:
    """Candidate performance and params selected for promotion decision."""

    metrics: dict[str, float]
    params: dict[str, float | int]


def _get_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load all features/labels used for final model retraining."""
    df = pd.read_parquet(RAW_PATH).set_index("Date")
    X, y, feats = make_features(df)
    return X, y, feats


def _generate_report_payload() -> dict[str, object]:
    """Run evaluation and return report payload matching report_v1 format."""
    return {
        "report_version": "v1",
        **run_ablation(EvalConfig()),
    }


def _load_or_generate_report() -> dict[str, object]:
    """Load v1 report JSON, generating it if missing."""
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text())

    payload = _generate_report_payload()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _candidate_from_report(report: dict[str, object], family: str) -> EvalResult:
    """Extract selected family metrics/params from canonical report artifact."""
    rows = report.get("families", [])
    if not isinstance(rows, list):
        raise ValueError("Invalid report format: expected list in 'families'.")

    for row in rows:
        if row.get("family") == family:
            return EvalResult(
                metrics=dict(row["holdout_metrics"]),
                params=dict(row["best_params"]),
            )
    raise ValueError(f"Family '{family}' not found in {REPORT_PATH}.")


def _resolve_candidate_version(client: MlflowClient, run_id: str) -> str:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    by_run = [v for v in versions if v.run_id == run_id]
    if not by_run:
        raise RuntimeError("Could not resolve registered model version for this run.")
    return max(by_run, key=lambda v: int(v.version)).version


def _champion_logloss(client: MlflowClient) -> float | None:
    try:
        champion_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None
    run = client.get_run(champion_mv.run_id)
    if "holdout_logloss" not in run.data.metrics:
        return None
    return float(run.data.metrics["holdout_logloss"])


def _should_promote(candidate_logloss: float, champion_logloss: float | None) -> bool:
    if champion_logloss is None:
        return True
    return candidate_logloss <= (champion_logloss - PROMOTION_MIN_LOGLOSS_IMPROVEMENT)


def main() -> None:
    report = _load_or_generate_report()
    eval_result = _candidate_from_report(report, REPORT_FAMILY)
    X_all, y_all, _ = _get_data()
    holdout_logloss = float(eval_result.metrics["logloss"])

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    with mlflow.start_run(run_name="train-register-promote") as run:
        for k, v in eval_result.params.items():
            mlflow.log_param(k, v)
        for k, v in eval_result.metrics.items():
            mlflow.log_metric(f"holdout_{k}", v)
        mlflow.log_param("target_horizon_days", 5)
        mlflow.log_param("holdout_start", HOLDOUT_START)
        mlflow.log_param("report_path", str(REPORT_PATH))
        mlflow.log_param("report_family", REPORT_FAMILY)
        mlflow.log_param("promotion_metric", "holdout_logloss")
        mlflow.log_param("promotion_min_improvement", PROMOTION_MIN_LOGLOSS_IMPROVEMENT)
        mlflow.log_param("features", ",".join(X_all.columns))

        # Retrain on all available data before registration.
        n_all = len(y_all)
        val_len = max(25, int(n_all * 0.10))
        tr_idx = np.arange(0, n_all - val_len)
        val_idx = np.arange(n_all - val_len, n_all)
        final_model = fit_xgb_reg(
            X_all.iloc[tr_idx],
            y_all.iloc[tr_idx],
            X_all.iloc[val_idx],
            y_all.iloc[val_idx],
            params=eval_result.params,
            seed=SEED,
        )

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        candidate_version = _resolve_candidate_version(client, run.info.run_id)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CANDIDATE_ALIAS, candidate_version)

        champion_ll = _champion_logloss(client)
        if _should_promote(holdout_logloss, champion_ll):
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, candidate_version)
            promoted = "yes"
        else:
            promoted = "no"

        print(
            f"run_id={run.info.run_id} version={candidate_version} "
            f"holdout_logloss={holdout_logloss:.6f} champion_prev={champion_ll} promoted={promoted}"
        )


if __name__ == "__main__":
    main()
