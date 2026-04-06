from pathlib import Path


def test_ci_workflow_exists_and_runs_repo_checks() -> None:
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.exists()

    text = workflow.read_text()
    assert "push:" in text
    assert "pull_request:" in text
    assert "pytest -q" in text
    assert "ruff check ." in text
