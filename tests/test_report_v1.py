import json
from pathlib import Path

import stock_movement_predictor.jobs.report_v1 as report_v1


def test_report_v1_main_writes_json_and_markdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_v1, "OUT_DIR", tmp_path)
    monkeypatch.setattr(report_v1, "OUT_JSON", tmp_path / "v1_report.json")
    monkeypatch.setattr(report_v1, "OUT_MD", tmp_path / "v1_report.md")

    fake_result = {
        "config": {
            "holdout_start": "2022-01-01",
            "wf_min_train": 1000,
            "wf_test_h": 63,
            "wf_max_folds": 8,
            "threshold": 0.5,
            "grid_size": 12,
        },
        "dataset": {"dev_rows": 100, "holdout_rows": 20, "folds": 2},
        "majority_baseline": {"logloss": 0.69, "brier": 0.25, "acc": 0.5, "dir_acc": 0.5},
        "families": [
            {
                "family": "full",
                "feature_count": 5,
                "holdout_metrics": {"logloss": 0.68, "brier": 0.24, "acc": 0.55, "dir_acc": 0.55},
                "delta_vs_majority": {"logloss": -0.01, "acc": 0.05},
            }
        ],
    }
    monkeypatch.setattr(report_v1, "run_ablation", lambda config: fake_result)

    report_v1.main()

    assert report_v1.OUT_JSON.exists()
    assert report_v1.OUT_MD.exists()
    payload = json.loads(report_v1.OUT_JSON.read_text())
    assert payload["report_version"] == "v1"
    assert payload["families"][0]["family"] == "full"
    assert "# SMX V1 Evaluation Report" in report_v1.OUT_MD.read_text()

