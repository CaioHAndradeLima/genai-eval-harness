"""Tests for evaluation scorers."""

from eval_harness.models import EvalSample, SampleResult, ScorerConfig
from eval_harness.scorers.engine import score_sample
from eval_harness.scorers.generation import exact_match, faithfulness
from eval_harness.scorers.retrieval import mrr, recall_at_k


def test_recall_at_k_hit():
    retrieved = ["MLflow Tracking logs metrics", "unrelated"]
    refs = ["MLflow Tracking is a component"]
    assert recall_at_k(retrieved, refs, k=2) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["cooking recipes"], ["MLflow Tracking"], k=1) == 0.0


def test_mrr_first_rank():
    retrieved = ["MLflow Tracking logs parameters", "other"]
    refs = ["parameters code versions metrics"]
    assert mrr(retrieved, refs) == 1.0


def test_exact_match_substring():
    assert exact_match("The answer is mlflow.log_metric", "mlflow.log_metric") == 1.0
    assert exact_match("no", "mlflow.log_metric") == 0.0


def test_faithfulness():
    score = faithfulness("mlflow tracking logs metrics", ["MLflow Tracking logs metrics and params"])
    assert score > 0.5


def test_score_sample_populates_metrics():
    sample = EvalSample(
        id="1",
        input="q",
        expected_output="mlflow tracking",
        reference_contexts=["MLflow Tracking logs metrics"],
    )
    result = SampleResult(
        sample_id="1",
        input="q",
        output="mlflow tracking logs metrics",
        retrieved_contexts=["MLflow Tracking logs metrics and parameters"],
    )
    scorers = ScorerConfig(
        retrieval=["recall_at_k", "mrr"],
        generation=["exact_match", "faithfulness"],
        ops=["latency_ms"],
    )
    metrics = score_sample(sample, result, scorers, top_k=3)
    names = {m.name for m in metrics}
    assert "recall_at_k" in names
    assert "faithfulness" in names
