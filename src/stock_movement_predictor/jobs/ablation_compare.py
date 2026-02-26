"""CLI wrapper for quick ablation output in terminal."""

from __future__ import annotations

from stock_movement_predictor.eval.v1_evaluation import EvalConfig, run_ablation


def main() -> None:
    """Print concise baseline and family comparison metrics."""
    result = run_ablation(EvalConfig())
    base = result["majority_baseline"]
    dataset = result["dataset"]
    config = result["config"]

    print(
        "Dev rows={} Holdout rows={} Folds={} Grid={}".format(
            dataset["dev_rows"],
            dataset["holdout_rows"],
            dataset["folds"],
            config["grid_size"],
        )
    )
    print(
        "majority      | logloss={:.6f} brier={:.6f} acc={:.6f}".format(
            base["logloss"],
            base["brier"],
            base["acc"],
        )
    )
    print("\nAblation holdout comparison:")
    for row in result["families"]:
        m = row["holdout_metrics"]
        d = row["delta_vs_majority"]
        print(
            "{:13} | logloss={:.6f} brier={:.6f} acc={:.6f} d_logloss={:+.6f} d_acc={:+.6f}".format(
                row["family"],
                m["logloss"],
                m["brier"],
                m["acc"],
                d["logloss"],
                d["acc"],
            )
        )


if __name__ == "__main__":
    main()
