"""Generate the canonical v1 evaluation report artifacts (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from stock_movement_predictor.eval.v1_evaluation import EvalConfig, run_ablation


OUT_DIR = Path("reports")
OUT_JSON = OUT_DIR / "v1_report.json"
OUT_MD = OUT_DIR / "v1_report.md"


def main() -> None:
    """Run evaluation and write report artifacts under ./reports."""
    result = run_ablation(EvalConfig())
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "report_version": "v1",
        **result,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))
    OUT_MD.write_text(_render_markdown(payload))

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


def _render_markdown(result: dict[str, object]) -> str:
    """Render a human-readable report table from structured evaluation output."""
    config = result["config"]
    dataset = result["dataset"]
    base = result["majority_baseline"]
    families = result["families"]

    lines = [
        "# SMX V1 Evaluation Report",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        f"- holdout_start: `{config['holdout_start']}`",
        f"- wf_min_train: `{config['wf_min_train']}`",
        f"- wf_test_h: `{config['wf_test_h']}`",
        f"- wf_max_folds: `{config['wf_max_folds']}`",
        f"- threshold: `{config['threshold']}`",
        f"- grid_size: `{config['grid_size']}`",
        "",
        "## Dataset",
        f"- dev_rows: `{dataset['dev_rows']}`",
        f"- holdout_rows: `{dataset['holdout_rows']}`",
        f"- folds: `{dataset['folds']}`",
        "",
        "## Baseline",
        f"- majority logloss: `{base['logloss']:.6f}`",
        f"- majority brier: `{base['brier']:.6f}`",
        f"- majority acc: `{base['acc']:.6f}`",
        "",
        "## Family Comparison",
        "",
        "| family | features | logloss | brier | acc | d_logloss_vs_majority | d_acc_vs_majority |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in families:
        m = row["holdout_metrics"]
        d = row["delta_vs_majority"]
        lines.append(
            "| {family} | {feature_count} | {logloss:.6f} | {brier:.6f} | {acc:.6f} | {dlog:+.6f} | {dacc:+.6f} |".format(
                family=row["family"],
                feature_count=row["feature_count"],
                logloss=m["logloss"],
                brier=m["brier"],
                acc=m["acc"],
                dlog=d["logloss"],
                dacc=d["acc"],
            )
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
