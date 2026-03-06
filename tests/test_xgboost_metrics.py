import numpy as np

from stock_movement_predictor.models.xgboost import evaluate_classification


def test_evaluate_classification_keys_and_range() -> None:
    y_true = np.array([0, 1, 0, 1], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.4, 0.8], dtype=float)

    metrics = evaluate_classification(y_true, y_prob, threshold=0.5)

    assert set(metrics.keys()) == {"acc", "logloss", "brier", "dir_acc"}
    assert 0.0 <= metrics["acc"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 <= metrics["dir_acc"] <= 1.0

