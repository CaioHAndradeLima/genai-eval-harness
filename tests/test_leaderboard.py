"""Tests for leaderboard generation."""

from eval_harness.leaderboard import build_leaderboard, leaderboard_to_csv
from eval_harness.models import MatrixRunResult, MetricScore, VariantRunResult


def test_build_leaderboard_ranks_by_recall():
    result = MatrixRunResult(
        matrix_name="test",
        baseline="a",
        variant_results=[
            VariantRunResult(
                variant_name="a",
                matrix_name="test",
                dataset_name="d",
                dataset_version="1",
                aggregate_metrics=[MetricScore(name="recall_at_k", value=0.8)],
            ),
            VariantRunResult(
                variant_name="b",
                matrix_name="test",
                dataset_name="d",
                dataset_version="1",
                aggregate_metrics=[MetricScore(name="recall_at_k", value=1.0)],
            ),
        ],
    )
    rows = build_leaderboard(result)
    assert rows[0].variant_name == "b"
    assert rows[0].rank == 1
    assert rows[1].is_baseline is True


def test_leaderboard_csv_has_header():
    csv_text = leaderboard_to_csv([])
    assert "rank" in csv_text
