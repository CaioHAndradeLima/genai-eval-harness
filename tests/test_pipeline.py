"""Tests for the pipeline layer."""

from pathlib import Path

import pytest

from eval_harness.config import load_dataset, load_experiment_matrix, resolve_path
from eval_harness.pipeline.chunking import chunk_text
from eval_harness.pipeline.factory import build_rag_pipeline
from eval_harness.pipeline.prompt import PromptRenderer
from eval_harness.pipeline.retriever import BM25Retriever, create_retriever
from eval_harness.models import RetrieverConfig, RetrieverType

ROOT = Path(__file__).resolve().parents[1]


def test_chunk_text_overlap():
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_bm25_retriever_finds_relevant_chunk():
    chunks = [
        "MLflow Tracking logs parameters and metrics.",
        "Python is a programming language.",
        "Unrelated content about cooking recipes.",
    ]
    retriever = BM25Retriever(chunks)
    hits = retriever.retrieve("How do I log metrics in MLflow?", top_k=1)
    assert "MLflow Tracking" in hits[0]


def test_prompt_renderer():
    renderer = PromptRenderer("prompts/rag/qa_v1.jinja", base_dir=ROOT)
    prompt = renderer.render(question="What is MLflow?", contexts=["MLflow is an ML platform."])
    assert "What is MLflow?" in prompt
    assert "MLflow is an ML platform." in prompt


def test_rag_pipeline_mock_run():
    matrix = load_experiment_matrix(ROOT / "configs/matrices/rag_qa_baseline.yaml")
    variant = matrix.get_variant("bm25-only")
    dataset = load_dataset(resolve_path(ROOT, variant.data.path))
    sample = dataset.samples[0]

    pipeline = build_rag_pipeline(variant, ROOT, use_mock_model=True)
    result = pipeline.run(sample)

    assert result.error is None
    assert result.sample_id == sample.id
    assert result.output.startswith("Mock answer for:")
    assert len(result.retrieved_contexts) > 0
    assert "MLflow" in result.prompt


def test_create_retriever_requires_corpus():
    config = RetrieverConfig(type=RetrieverType.BM25, corpus_path=None)
    with pytest.raises(ValueError, match="corpus_path"):
        create_retriever(config, ROOT)
