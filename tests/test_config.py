"""Tests for config loading."""

from pathlib import Path

from eval_harness.config import load_dataset, load_experiment_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_load_experiment_matrix():
    matrix = load_experiment_matrix(ROOT / "configs/matrices/rag_qa_baseline.yaml")
    assert matrix.name == "rag-qa-baseline"
    assert len(matrix.variants) == 2
    assert matrix.baseline == "bm25-only"


def test_load_dataset():
    dataset = load_dataset(ROOT / "datasets/rag/mlflow_qa.json")
    assert dataset.name == "mlflow-qa"
    assert dataset.size == 5
    assert dataset.samples[0].id == "mlflow-001"
