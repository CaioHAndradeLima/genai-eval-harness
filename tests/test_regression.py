"""Regression tests — full eval pipeline vs committed baseline."""

from pathlib import Path

import pytest

from eval_harness.config import load_experiment_matrix
from eval_harness.leaderboard import build_leaderboard
from eval_harness.regression import assert_regression, check_regression
from eval_harness.runner import ExperimentRunner
from eval_harness.scorers import score_matrix_run

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/matrices/rag_qa_baseline.yaml"
BASELINE = ROOT / "examples/baseline_metrics.json"


def _run_scored_matrix():
    matrix = load_experiment_matrix(MATRIX)
    runner = ExperimentRunner(ROOT, use_mock_model=True, show_progress=False)
    result = runner.run_matrix(matrix, MATRIX)
    return score_matrix_run(matrix, MATRIX, result)


def test_full_matrix_regression_against_baseline():
    """CI gate: mock eval metrics must stay within tolerance of baseline."""
    if not BASELINE.exists():
        pytest.skip("baseline file not committed yet")
    result = _run_scored_matrix()
    assert_regression(result, BASELINE)


def test_regression_detects_drop(tmp_path: Path):
    result = _run_scored_matrix()
    # Write an impossibly high baseline
    bad = tmp_path / "bad.json"
    bad.write_text(
        """
{
  "matrix": "rag-qa-baseline",
  "tolerance": 0.0,
  "variants": {
    "bm25-only": {"recall_at_k": 999.0}
  }
}
""",
        encoding="utf-8",
    )
    failures = check_regression(result, bad)
    assert failures


def test_leaderboard_built_after_scoring():
    result = _run_scored_matrix()
    rows = build_leaderboard(result)
    assert len(rows) == 2
    assert all(row.metrics for row in rows)
